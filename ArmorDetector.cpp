#include "ArmorDetector.hpp"

#include <string>

#include "libxr_rw.hpp"
#include "logger.hpp"

ArmorDetector::ArmorDetector(LibXR::HardwareContainer& hw, LibXR::ApplicationManager& app,
                             Config cfg)
    : cfg_cache_(cfg),
      binary_thres_(cfg.binary_thres),
      detect_color_(cfg.detect_color),
      l_(cfg.light),
      a_(cfg.armor),
      classifier_threshold_(cfg.classifier.classifier_threshold),
      ignore_classes_(cfg.classifier.ignore_classes),
      cmd_file_(LibXR::RamFS::CreateFile(name_, CommandFun, this))
{
  XR_LOG_INFO("Starting ArmorDetector (with Config).");

  hw.template FindOrExit<LibXR::RamFS>({"ramfs"})->Add(cmd_file_);

  // 初始化分类器
  InitClassifier();

  // 订阅 camera_info
  auto info_topic = LibXR::Topic(LibXR::Topic::Find("camera_info"));
  auto info_cb = LibXR::Topic::Callback::Create(
      [](bool, ArmorDetector* self, LibXR::RawData& data)
      {
        auto* camera_info = reinterpret_cast<CameraBase::CameraInfo*>(data.addr_);
        static bool inited = false;
        if (!inited)
        {
          XR_LOG_PASS("Got camera info!");
          inited = true;

          self->cam_center_ =
              cv::Point2f(static_cast<float>(camera_info->camera_matrix[2]),
                          static_cast<float>(camera_info->camera_matrix[5]));
          self->cam_info_ = std::make_shared<CameraBase::CameraInfo>(*camera_info);

          ASSERT(camera_info->distortion_model == CameraBase::DistortionModel::PLUMB_BOB);

          auto dist =
              std::array<double, 5>{camera_info->distortion_coefficients.plumb_bob.k1,
                                    camera_info->distortion_coefficients.plumb_bob.k2,
                                    camera_info->distortion_coefficients.plumb_bob.p1,
                                    camera_info->distortion_coefficients.plumb_bob.p2,
                                    camera_info->distortion_coefficients.plumb_bob.k3};

          self->pnp_solver_ =
              std::make_unique<PnPSolver>(camera_info->camera_matrix, dist);
        }
      },
      this);
  info_topic.RegisterCallback(info_cb);

  // 订阅 image_raw
  auto img_topic = LibXR::Topic(LibXR::Topic::Find("image_raw"));
  auto img_cb = LibXR::Topic::Callback::Create(
      [](bool, ArmorDetector* self, LibXR::RawData& data)
      {
        XR_LOG_DEBUG("Got image!");
        auto* img_msg = reinterpret_cast<cv::Mat*>(data.addr_);
        if (self->params_is_changed_ == true)
        {
          self->SetConfig(self->cfg_cache_);
          self->params_is_changed_ = false;
        }
        self->ImageCallback(img_msg);
      },
      this);
  img_topic.RegisterCallback(img_cb);

  app.Register(*this);
}

void ArmorDetector::SetConfig(const Config& cfg)
{
  cfg_cache_ = cfg;

  binary_thres_ = cfg.binary_thres;
  detect_color_ = cfg.detect_color;
  l_ = cfg.light;
  a_ = cfg.armor;
  classifier_threshold_ = cfg.classifier.classifier_threshold;
  ignore_classes_ = cfg.classifier.ignore_classes;

  classifier_.reset();
  InitClassifier();
}

void ArmorDetector::InitClassifier()
{
  classifier_ = std::make_unique<NumberClassifier>(
      DETECTOR_MODEL_PATH, classifier_threshold_, ignore_classes_);
}

cv::Mat ArmorDetector::PreprocessImage(const cv::Mat& rgb_img)
{
  cv::Mat gray;
  cv::cvtColor(rgb_img, gray, cv::COLOR_RGB2GRAY);
  cv::threshold(gray, gray, binary_thres_, 255, cv::THRESH_BINARY);
  return gray;
}

std::vector<Light> ArmorDetector::FindLights(const cv::Mat& rgb_img,
                                             const cv::Mat& binary_img)
{
  using std::vector;
  vector<vector<cv::Point>> contours;
  vector<cv::Vec4i> hierarchy;
  cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  vector<Light> lights;
  lights.reserve(contours.size());

  for (size_t i = 0; i < contours.size(); ++i)
  {
    const auto& contour = contours[i];
    if (contour.size() < 5)
    {
      continue;
    }

    auto r_rect = cv::minAreaRect(contour);
    Light light(r_rect);

    if (!IsLight(light))
    {
      continue;
    }

    const auto RECT = light.boundingRect();
    if (!(0 <= RECT.x && 0 <= RECT.width && RECT.x + RECT.width <= rgb_img.cols &&
          0 <= RECT.y && 0 <= RECT.height && RECT.y + RECT.height <= rgb_img.rows))
    {
      continue;
    }

    const cv::Mat ROI = rgb_img(RECT);

    cv::Mat mask(RECT.size(), CV_8UC1, cv::Scalar(0));
    std::vector<std::vector<cv::Point>> one{contour};
    // 将整图坐标的轮廓偏移到 ROI 坐标系
    cv::drawContours(mask, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED,
                     cv::LINE_8, cv::noArray(), INT_MAX, cv::Point(-RECT.x, -RECT.y));

    const cv::Scalar MEAN_RGB = cv::mean(ROI, mask);
    const double MEAN_R = MEAN_RGB[0];
    const double MEAN_B = MEAN_RGB[2];
    light.color = (MEAN_R > MEAN_B) ? RED : BLUE;

    lights.emplace_back(light);
  }

  return lights;
}

bool ArmorDetector::IsLight(const Light& light)
{
  const double RATIO = light.width / LibXR::max(light.length, 1e-6f);
  const bool RATIO_OK = (l_.min_ratio < RATIO && RATIO < l_.max_ratio);

  const bool ANGLE_OK = (light.tilt_angle < l_.light_max_angle);

  return RATIO_OK && ANGLE_OK;
}

std::vector<Armor> ArmorDetector::MatchLights(const std::vector<Light>& lights)
{
  std::vector<Armor> armors;
  armors.reserve(lights.size());

  for (auto light_1 = lights.begin(); light_1 != lights.end(); ++light_1)
  {
    for (auto light_2 = light_1 + 1; light_2 != lights.end(); ++light_2)
    {
      if (light_1->color != detect_color_ || light_2->color != detect_color_)
      {
        continue;
      }

      if (ContainLight(*light_1, *light_2, lights))
      {
        continue;
      }

      const auto TYPE = IsArmor(*light_1, *light_2);
      if (TYPE != ArmorType::INVALID)
      {
        XR_LOG_DEBUG("Found armor type %d", static_cast<int>(TYPE));
        Armor armor(*light_1, *light_2);
        armor.type = TYPE;
        armors.emplace_back(std::move(armor));
      }
    }
  }
  return armors;
}

bool ArmorDetector::ContainLight(const Light& light_1, const Light& light_2,
                                 const std::vector<Light>& lights)
{
  std::vector<cv::Point2f> points{light_1.top, light_1.bottom, light_2.top,
                                  light_2.bottom};
  const auto BOUNDING_RECT = cv::boundingRect(points);

  for (const auto& test_light : lights)
  {
    if (test_light.center == light_1.center || test_light.center == light_2.center)
    {
      continue;
    }

    if (BOUNDING_RECT.contains(test_light.top) ||
        BOUNDING_RECT.contains(test_light.bottom) ||
        BOUNDING_RECT.contains(test_light.center))
    {
      return true;
    }
  }
  return false;
}

ArmorType ArmorDetector::IsArmor(const Light& light_1, const Light& light_2)
{
  const double LEN1 = light_1.length, LEN2 = light_2.length;
  const double LIGHT_LENGTH_RATION =
      (LEN1 < LEN2) ? (LEN1 / LibXR::max(LEN2, 1e-6f)) : (LEN2 / LibXR::max(LEN1, 1e-6f));
  const bool LIGHT_RATIO_OK = LIGHT_LENGTH_RATION > a_.min_light_ratio;

  const double AVG_LIGHT_LENGTH = (LEN1 + LEN2) / 2.0;
  const double CENTER_DISTANCE =
      cv::norm(light_1.center - light_2.center) / LibXR::max(AVG_LIGHT_LENGTH, 1e-6f);
  const bool CENTER_DISTANCE_OK = (a_.min_small_center_distance <= CENTER_DISTANCE &&
                                   CENTER_DISTANCE < a_.max_small_center_distance) ||
                                  (a_.min_large_center_distance <= CENTER_DISTANCE &&
                                   CENTER_DISTANCE < a_.max_large_center_distance);

  const cv::Point2f DIFF = light_1.center - light_2.center;
  const float ANGLE = std::abs(std::atan2(std::abs(DIFF.y), std::abs(DIFF.x))) * 180.0f /
                      static_cast<float>(CV_PI);
  const bool ANGLE_OK = ANGLE < a_.max_angle;

  const bool IS_ARMOR = LIGHT_RATIO_OK && CENTER_DISTANCE_OK && ANGLE_OK;
  if (!IS_ARMOR)
  {
    return ArmorType::INVALID;
  }

  return (CENTER_DISTANCE > a_.min_large_center_distance) ? ArmorType::LARGE
                                                          : ArmorType::SMALL;
}

std::vector<Armor> ArmorDetector::Detect(const cv::Mat& input)
{
  binary_img_ = PreprocessImage(input);

#if defined(AUTO_AIM_PREVIEW_IMAGE) && AUTO_AIM_PREVIEW_IMAGE
  cv::imshow("binary_img", binary_img_);
  cv::waitKey(1);
#endif
  lights_ = FindLights(input, binary_img_);
  armors_ = MatchLights(lights_);

  if (!lights_.empty())
  {
    XR_LOG_DEBUG("Found %d lights", static_cast<int>(lights_.size()));
  }
  if (!armors_.empty())
  {
    XR_LOG_INFO("Found %d armors", static_cast<int>(armors_.size()));
  }

  if (!armors_.empty() && classifier_)
  {
    classifier_->ExtractNumbers(input, armors_);
    classifier_->Classify(armors_);
  }

  return armors_;
}

// =========== 回调 / 发布 ===========
void ArmorDetector::ImageCallback(cv::Mat* img_msg)
{
  if (pnp_solver_ == nullptr || !cam_info_)
  {
    // 没有相机信息或求解器就直接返回
    return;
  }

  cv::Mat& frame = *img_msg;
  const auto ARMORS = Detect(frame);

  cv::Scalar green(0, 255, 0), red(0, 0, 255);
  cv::drawMarker(frame, cam_center_, green, cv::MARKER_CROSS, 20, 2);  // 主点
  for (const auto& a : ARMORS)
  {
    cv::drawMarker(frame, a.center, red, cv::MARKER_TILTED_CROSS, 18, 2);  // 装甲中心
    cv::line(frame, cam_center_, a.center, red, 1);
  }

#if defined(AUTO_AIM_PREVIEW_IMAGE) && AUTO_AIM_PREVIEW_IMAGE
  cv::imshow("debug_pose_view", frame);
  cv::waitKey(1);
#endif

  armors_msg_.clear();

  const double FX = cam_info_->camera_matrix[0];
  const double FY = cam_info_->camera_matrix[4];
  const double CX = cam_info_->camera_matrix[2];
  const double CY = cam_info_->camera_matrix[5];

  const double K1 = cam_info_->distortion_coefficients.plumb_bob.k1;
  const double K2 = cam_info_->distortion_coefficients.plumb_bob.k2;
  const double P1 = cam_info_->distortion_coefficients.plumb_bob.p1;
  const double P2 = cam_info_->distortion_coefficients.plumb_bob.p2;
  const double K3 = cam_info_->distortion_coefficients.plumb_bob.k3;

  cv::Mat k_use = (cv::Mat_<double>(3, 3) << FX, 0, CX, 0, FY, CY, 0, 0, 1);
  cv::Mat dist_use = (cv::Mat_<double>(1, 5) << K1, K2, P1, P2, K3);

  for (const auto& armor : ARMORS)
  {
    cv::Mat rvec, tvec;
    const bool SUCCESS = pnp_solver_->SolvePnP(armor, rvec, tvec);
    if (!SUCCESS)
    {
      XR_LOG_WARN("PnP failed!");
      continue;
    }

    XR_LOG_DEBUG("Got armor pose!");
    const double Z = std::max(1e-9, tvec.at<double>(2));

    ArmorDetectorResult armor_msg;
    armor_msg.type = armor.type;
    armor_msg.number = armor.number;

    armor_msg.pose.translation.x() = tvec.at<double>(0);
    armor_msg.pose.translation.y() = tvec.at<double>(1);
    armor_msg.pose.translation.z() = Z;

    cv::Mat r;
    cv::Rodrigues(rvec, r);
    armor_msg.pose.rotation = LibXR::RotationMatrix<double>(
        r.at<double>(0, 0), r.at<double>(0, 1), r.at<double>(0, 2), r.at<double>(1, 0),
        r.at<double>(1, 1), r.at<double>(1, 2), r.at<double>(2, 0), r.at<double>(2, 1),
        r.at<double>(2, 2));

    /* 这里还在相机坐标系，Z近似相机到装甲板的距离 */
    const auto EULR = armor_msg.pose.rotation.ToEulerAngle();
    XR_LOG_INFO("Armor Roll: %f, Pitch: %f, Yaw: %f", EULR[0], EULR[1], EULR[2]);
    XR_LOG_INFO("Armor x: %f, y: %f, z: %f", armor_msg.pose.translation.x(),
                armor_msg.pose.translation.y(), armor_msg.pose.translation.z());

    armor_msg.distance_to_image_center =
        pnp_solver_->CalculateDistanceToCenter(armor.center);

    armors_msg_.emplace_back(std::move(armor_msg));
  }

  // 发布
  armors_topic_.Publish(armors_msg_);
}

int ArmorDetector::CommandFun(ArmorDetector* self, int argc, char** argv)
{
  if (argc == 1)
  {
    LibXR::STDIO::Printf("ArmorDetector\n\n");
    LibXR::STDIO::Printf("Usage:\r\n");
    LibXR::STDIO::Printf("  show\r\n");
    LibXR::STDIO::Printf("  binary_thres <value>\r\n");
    LibXR::STDIO::Printf("  detect_color <value>\r\n");
    LibXR::STDIO::Printf("  classifier_threshold <value>\r\n");
    LibXR::STDIO::Printf("  min_ratio <value>\r\n");
    LibXR::STDIO::Printf("  max_ratio <value>\r\n");
    LibXR::STDIO::Printf("  light_max_angle <value>\r\n");
    LibXR::STDIO::Printf("  min_light_ratio <value>\r\n");
    LibXR::STDIO::Printf("  min_small_center_distance <value>\r\n");
    LibXR::STDIO::Printf("  max_small_center_distance <value>\r\n");
    LibXR::STDIO::Printf("  min_large_center_distance <value>\r\n");
    LibXR::STDIO::Printf("  max_large_center_distance <value>\r\n");
    LibXR::STDIO::Printf("  max_angle <value>\r\n");
    XR_LOG_INFO("Show Detector Terminal Parameters");
    return 0;
  }
  else if (argc == 2)
  {
    std::string cmd = argv[1];
    if (cmd == "show")
    {
      // clang-format off
      LibXR::STDIO::Printf("name: ArmorDetector\r\n");
      LibXR::STDIO::Printf("cfg: \r\n");
      LibXR::STDIO::Printf("  classifier: \r\n");
      LibXR::STDIO::Printf("    ignore_classes: \r\n");
      LibXR::STDIO::Printf("      - ArmorNumber::NEGATIVE\r\n");
      LibXR::STDIO::Printf("    classifier_threshold: %f\r\n", self->cfg_cache_.classifier.classifier_threshold);
      LibXR::STDIO::Printf("  detect_color: %d\r\n", self->cfg_cache_.detect_color);
      LibXR::STDIO::Printf("  binary_thres: %d\r\n", self->cfg_cache_.binary_thres);
      LibXR::STDIO::Printf("  light: \r\n");
      LibXR::STDIO::Printf("    min_ratio: %f\r\n", self->cfg_cache_.light.min_ratio);
      LibXR::STDIO::Printf("    max_ratio: %f\r\n", self->cfg_cache_.light.max_ratio);
      LibXR::STDIO::Printf("    light_max_angle: %f\r\n", self->cfg_cache_.light.light_max_angle);
      LibXR::STDIO::Printf("  armor: \r\n");
      LibXR::STDIO::Printf("    min_light_ratio: %f\r\n", self->cfg_cache_.armor.min_light_ratio);
      LibXR::STDIO::Printf("    min_small_center_distance: %f\r\n", self->cfg_cache_.armor.min_small_center_distance);
      LibXR::STDIO::Printf("    max_small_center_distance: %f\r\n", self->cfg_cache_.armor.max_small_center_distance);
      LibXR::STDIO::Printf("    min_large_center_distance: %f\r\n", self->cfg_cache_.armor.min_large_center_distance);
      LibXR::STDIO::Printf("    max_large_center_distance: %f\r\n", self->cfg_cache_.armor.max_large_center_distance);
      LibXR::STDIO::Printf("    max_angle: %f\r\n", self->cfg_cache_.armor.max_angle);
      XR_LOG_INFO("Show Detector Configs");
      // clang-format on
    }
    return 0;
  }
  else if (argc == 3)
  {
    self->params_is_changed_ = false;
    std::string cmd = argv[1];
    if (cmd == "binary_thres")
    {
      self->cfg_cache_.binary_thres = std::stoi(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "detect_color")
    {
      self->cfg_cache_.detect_color = std::stoi(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "classifier_threshold")
    {
      self->cfg_cache_.classifier.classifier_threshold = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "min_ratio")
    {
      self->cfg_cache_.light.min_ratio = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "max_ratio")
    {
      self->cfg_cache_.light.max_ratio = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "light_max_angle")
    {
      self->cfg_cache_.light.light_max_angle = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "min_light_ratio")
    {
      self->cfg_cache_.armor.min_light_ratio = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "min_small_center_distance")
    {
      self->cfg_cache_.armor.min_small_center_distance = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "max_small_center_distance")
    {
      self->cfg_cache_.armor.max_small_center_distance = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "min_large_center_distance")
    {
      self->cfg_cache_.armor.min_large_center_distance = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "max_large_center_distance")
    {
      self->cfg_cache_.armor.max_large_center_distance = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else if (cmd == "max_angle")
    {
      self->cfg_cache_.armor.max_angle = std::stod(argv[2]);
      self->params_is_changed_ = true;
    }
    else
    {
      LibXR::STDIO::Printf("Unknown command: %s\n", argv[1]);
      return -1;
    }
    XR_LOG_INFO("Change Detector Configs");
    return 0;
  }
  LibXR::STDIO::Printf("Unknown command: %s, %s\n", argv[0], argv[1]);
  return -1;
}