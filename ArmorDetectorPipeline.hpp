#pragma once

// 仅供 ArmorDetector.hpp 在类声明之后包含。
template <CameraTypes::CameraInfo CameraInfoV>
ArmorDetector<CameraInfoV>::ArmorDetector(LibXR::HardwareContainer&,
                                          LibXR::ApplicationManager& app,
                                          Config cfg,
                                          Sync& sync)
    : sync_(sync)
{
  SetConfig(cfg);

  sync_frame_thread_.Create(this, SyncFrameThreadFun, "ArmorDetSync",
                            detail::sync_frame_thread_stack_size,
                            LibXR::Thread::Priority::HIGH);

  app.Register(*this);
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorDetector<CameraInfoV>::SetConfig(const Config& cfg)
{
  cfg_ = cfg;
  refined_count_ = 0U;
  discarded_count_ = 0U;
  model_ready_ = false;

  try
  {
    auto model = ov_core_.read_model(ARMOR_DETECTOR_MODEL_PATH);
    ov::preprocess::PrePostProcessor post_processor(model);
    auto& input = post_processor.input();

    input.tensor()
        .set_element_type(ov::element::u8)
        .set_shape(
            ov::PartialShape{1, detail::yolo_input_size, detail::yolo_input_size, 3})
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::BGR);
    input.model().set_layout("NCHW");
    input.preprocess()
        .convert_element_type(ov::element::f32)
        .convert_color(ov::preprocess::ColorFormat::RGB)
        .scale(255.0);

    model = post_processor.build();
    compiled_model_ = ov_core_.compile_model(
        model, "CPU",
        ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    infer_request_ = compiled_model_.create_infer_request();
    model_ready_ = true;
  }
  catch (const std::exception& exception)
  {
    XR_LOG_ERROR("ArmorDetector failed to load YOLOv5 model: %s", exception.what());
    compiled_model_ = ov::CompiledModel();
    infer_request_ = ov::InferRequest();
  }
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorDetector<CameraInfoV>::ProcessImage(const cv::Mat& img_msg,
                                              uint64_t image_timestamp_us)
{
  if (img_msg.empty())
  {
    return;
  }

  latest_timestamp_us_ = image_timestamp_us;

  const auto start_time = std::chrono::steady_clock::now();
  const cv::Mat& bgr_img = img_msg;

  cv::Mat binary_debug;
  cv::Mat* binary_debug_ptr = nullptr;
  if (cfg_.debug.preview && cfg_.debug.show_binary)
  {
    binary_debug_ptr = &binary_debug;
  }

  const auto armors = Detect(bgr_img, binary_debug_ptr);
  const auto detector_finish = std::chrono::steady_clock::now();

  FillResultMessage(armors, bgr_img);
  const auto publish_finish = std::chrono::steady_clock::now();

  ++frame_index_;
  metrics_msg_.frame_index = frame_index_;
  metrics_msg_.image_timestamp_us = latest_timestamp_us_;
  metrics_msg_.armor_count = static_cast<uint32_t>(armors_msg_.results.size());
  metrics_msg_.refined_count = refined_count_;
  metrics_msg_.discarded_count = discarded_count_;
  metrics_msg_.detector_latency_ms =
      std::chrono::duration<double, std::milli>(detector_finish - start_time).count();
  metrics_msg_.publish_latency_ms =
      std::chrono::duration<double, std::milli>(publish_finish - detector_finish).count();

  if (ShouldShowPreview())
  {
    ShowDebugPreview(bgr_img, binary_debug_ptr);
  }

  metrics_topic_.Publish(metrics_msg_);
  armors_topic_.Publish(armors_msg_);

  if ((frame_index_ % detail::metrics_log_period) == 0U)
  {
    XR_LOG_INFO(
        "ArmorDetector frame=%llu armors=%u refined=%u discarded=%u detector_ms=%.2f publish_ms=%.2f",
        static_cast<unsigned long long>(metrics_msg_.frame_index),
        metrics_msg_.armor_count, metrics_msg_.refined_count,
        metrics_msg_.discarded_count, metrics_msg_.detector_latency_ms,
        metrics_msg_.publish_latency_ms);
  }
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorDetector<CameraInfoV>::ProcessSyncedFrame(const SyncedFrame& synced_frame)
{
  const auto* image_frame = synced_frame.GetImageFrame();
  if (image_frame == nullptr)
  {
    XR_LOG_ERROR("ArmorDetector received null synced image");
    return;
  }

  const int cv_type = detail::CvTypeFromEncoding(kCameraInfo.encoding);
  if (cv_type < 0)
  {
    XR_LOG_WARN("ArmorDetector sync frame encoding unsupported: %u",
                static_cast<unsigned>(kCameraInfo.encoding));
    return;
  }

  cv::Mat img(static_cast<int>(kCameraInfo.height), static_cast<int>(kCameraInfo.width),
              cv_type, const_cast<uint8_t*>(image_frame->data.data()),
              static_cast<size_t>(kCameraInfo.step));
  const cv::Mat bgr_img =
      detail::ConvertToBgrWithEncoding(img, kCameraInfo.encoding);
  if (bgr_img.empty())
  {
    return;
  }
  ProcessImage(bgr_img, image_frame->timestamp_us);
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorDetector<CameraInfoV>::SyncFrameThreadFun(ArmorDetector<CameraInfoV>* self)
{
  XR_LOG_INFO("ArmorDetector sync worker starting: image=%s imu=%s",
              self->sync_.ImageTopicName(), self->sync_.ImuTopicName());

  bool attach_logged = false;
  while (true)
  {
    typename Sync::Subscriber subscriber(self->sync_);
    if (!subscriber.Valid())
    {
      if (!attach_logged)
      {
        XR_LOG_WARN("ArmorDetector waiting for sync image topic: %s",
                    self->sync_.ImageTopicName());
        attach_logged = true;
      }
      LibXR::Thread::Sleep(detail::sync_frame_retry_sleep_ms);
      continue;
    }

    XR_LOG_PASS("ArmorDetector attached sync stream: image=%s imu=%s",
                self->sync_.ImageTopicName(), self->sync_.ImuTopicName());
    attach_logged = false;

    SyncedFrame synced_frame;
    while (true)
    {
      const auto wait_ans =
          subscriber.Wait(synced_frame, detail::sync_frame_wait_timeout_ms);
      if (wait_ans == LibXR::ErrorCode::TIMEOUT)
      {
        continue;
      }
      if (wait_ans != LibXR::ErrorCode::OK)
      {
        XR_LOG_WARN("ArmorDetector sync wait failed (err=%d), retrying attach.",
                    static_cast<int>(wait_ans));
        break;
      }

      self->ProcessSyncedFrame(synced_frame);
    }
  }
}

template <CameraTypes::CameraInfo CameraInfoV>
std::vector<typename ArmorDetector<CameraInfoV>::CandidateArmor>
ArmorDetector<CameraInfoV>::Detect(const cv::Mat& raw_img,
                                   cv::Mat* binary_debug)
{
  refined_count_ = 0U;
  discarded_count_ = 0U;

  if (!model_ready_)
  {
    XR_LOG_ERROR("ArmorDetector YOLOv5 model is not ready");
    return {};
  }

  cv::Mat detector_img = raw_img;
  cv::Point2f offset(0.0F, 0.0F);
  cv::Rect clipped_roi(0, 0, raw_img.cols, raw_img.rows);

  // 1. 根据配置裁剪出真正送入网络的图像区域。
  if (cfg_.yolo.use_roi)
  {
    int roi_width = cfg_.yolo.roi_width;
    int roi_height = cfg_.yolo.roi_height;
    if (roi_width < 0)
    {
      roi_width = raw_img.cols;
    }
    if (roi_height < 0)
    {
      roi_height = raw_img.rows;
    }

    const cv::Rect full_roi(0, 0, raw_img.cols, raw_img.rows);
    const cv::Rect roi(cfg_.yolo.roi_x, cfg_.yolo.roi_y, roi_width, roi_height);
    clipped_roi = roi & full_roi;
    if (clipped_roi.empty())
    {
      ++discarded_count_;
      return {};
    }

    detector_img = raw_img(clipped_roi);
    offset = cv::Point2f(static_cast<float>(clipped_roi.x),
                         static_cast<float>(clipped_roi.y));
  }

  // 2. 仅在调试预览时构造二值化图，避免把传统细化的调试逻辑散落到主链路里。
  if (binary_debug != nullptr)
  {
    cv::Mat gray_img;
    cv::cvtColor(detector_img, gray_img, cv::COLOR_BGR2GRAY);
    cv::Mat threshold_img;
    cv::threshold(gray_img, threshold_img, cfg_.traditional.threshold, 255,
                  cv::THRESH_BINARY);

    if (cfg_.yolo.use_roi)
    {
      *binary_debug = cv::Mat::zeros(raw_img.size(), CV_8UC1);
      threshold_img.copyTo((*binary_debug)(clipped_roi));
    }
    else
    {
      *binary_debug = threshold_img;
    }
  }

  const double height_scale =
      static_cast<double>(detail::yolo_input_size) / std::max(1, detector_img.rows);
  const double width_scale =
      static_cast<double>(detail::yolo_input_size) / std::max(1, detector_img.cols);
  const double scale = std::min(height_scale, width_scale);
  const int resized_height =
      std::max(1, static_cast<int>(std::round(detector_img.rows * scale)));
  const int resized_width =
      std::max(1, static_cast<int>(std::round(detector_img.cols * scale)));

  // 3. letterbox 到固定输入尺寸，并复用持久化 infer request。
  cv::Mat input(detail::yolo_input_size, detail::yolo_input_size, CV_8UC3,
                cv::Scalar(0, 0, 0));
  cv::resize(detector_img,
             input(cv::Rect(0, 0, resized_width, resized_height)),
             cv::Size(resized_width, resized_height));

  ov::Tensor input_tensor(
      ov::element::u8,
      ov::Shape{1, static_cast<size_t>(detail::yolo_input_size),
                static_cast<size_t>(detail::yolo_input_size), 3},
      input.data);
  infer_request_.set_input_tensor(input_tensor);
  infer_request_.infer();

  auto output_tensor = infer_request_.get_output_tensor();
  const auto output_shape = output_tensor.get_shape();
  cv::Mat output(static_cast<int>(output_shape[1]), static_cast<int>(output_shape[2]),
                 CV_32F, output_tensor.data<float>());
  auto armors = DecodeOutput(scale, output, detector_img);
  if (armors.empty())
  {
    return armors;
  }

  // 4. 如果网络只看了 ROI，这里再把检测结果平移回整幅原图坐标系。
  if (cfg_.yolo.use_roi)
  {
    for (auto& armor : armors)
    {
      for (auto& point : armor.points)
      {
        point += offset;
      }
      armor.center += offset;
      armor.center_norm = GetNormalizedCenter(raw_img, armor.center);
      armor.box.x += static_cast<int>(offset.x);
      armor.box.y += static_cast<int>(offset.y);
    }
  }

  return armors;
}

template <CameraTypes::CameraInfo CameraInfoV>
std::vector<typename ArmorDetector<CameraInfoV>::CandidateArmor>
ArmorDetector<CameraInfoV>::DecodeOutput(double scale, const cv::Mat& output,
                                         const cv::Mat& bgr_img)
{
  std::vector<NetworkDetection> detections;
  const ArmorColor target_color = detail::detect_color_from_config(cfg_.detect_color);

  for (int row = 0; row < output.rows; ++row)
  {
    const auto detection = DecodeDetection(scale, output, row);
    if (!detection.has_value())
    {
      continue;
    }

    detections.emplace_back(*detection);
  }

  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  boxes.reserve(detections.size());
  confidences.reserve(detections.size());
  for (const auto& detection : detections)
  {
    boxes.emplace_back(detection.box);
    confidences.emplace_back(detection.confidence);
  }
  if (boxes.empty())
  {
    return {};
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, static_cast<float>(cfg_.yolo.score_threshold),
                    static_cast<float>(cfg_.yolo.nms_threshold), indices);

  std::vector<CandidateArmor> armors;
  armors.reserve(indices.size());

  for (const int index : indices)
  {
    CandidateArmor armor = BuildCandidateArmor(detections[index], bgr_img);

    const bool color_mismatch =
        (target_color != ArmorColor::UNKNOWN) && (armor.color != target_color);
    if (color_mismatch ||
        !ArmorNumberIsKnown(armor.number) ||
        armor.confidence < static_cast<float>(cfg_.yolo.min_confidence))
    {
      ++discarded_count_;
      continue;
    }

    if (cfg_.yolo.use_traditional_refine)
    {
      if (RefineArmorCorners(armor, bgr_img))
      {
        ++refined_count_;
      }
    }

    if (ArmorNumberIsLarge(armor.number))
    {
      armor.type = ArmorType::LARGE;
    }
    else if (ArmorNumberIsSmall(armor.number))
    {
      armor.type = ArmorType::SMALL;
    }
    else
    {
      armor.type = InferArmorType(armor);
    }
    if (!ValidateArmorType(armor))
    {
      ++discarded_count_;
      continue;
    }

    armors.emplace_back(std::move(armor));
  }

  return armors;
}

template <CameraTypes::CameraInfo CameraInfoV>
std::optional<typename ArmorDetector<CameraInfoV>::NetworkDetection>
ArmorDetector<CameraInfoV>::DecodeDetection(double scale, const cv::Mat& output,
                                            int row) const
{
  double score = output.at<float>(row, detail::OutputLayout::objectness_index);
  score = 1.0 / (1.0 + std::exp(-score));
  if (score < cfg_.yolo.score_threshold)
  {
    return std::nullopt;
  }

  const cv::Mat color_scores =
      output.row(row).colRange(detail::OutputLayout::color_begin,
                               detail::OutputLayout::color_end);
  const cv::Mat number_scores =
      output.row(row).colRange(detail::OutputLayout::number_begin,
                               detail::OutputLayout::number_end);
  cv::Point color_id_point;
  cv::Point number_id_point;
  cv::minMaxLoc(color_scores, nullptr, nullptr, nullptr, &color_id_point);
  cv::minMaxLoc(number_scores, nullptr, nullptr, nullptr, &number_id_point);

  const std::array<cv::Point2f, 4> unsorted_points = {
      cv::Point2f(output.at<float>(row, 0) / scale, output.at<float>(row, 1) / scale),
      cv::Point2f(output.at<float>(row, 6) / scale, output.at<float>(row, 7) / scale),
      cv::Point2f(output.at<float>(row, 4) / scale, output.at<float>(row, 5) / scale),
      cv::Point2f(output.at<float>(row, 2) / scale, output.at<float>(row, 3) / scale)};

  NetworkDetection detection;
  detection.color = detail::color_from_yolo_id(color_id_point.x);
  detection.number = detail::number_from_yolo_id(number_id_point.x);
  detection.confidence = static_cast<float>(score);
  detection.points = detail::sort_keypoints(unsorted_points);
  detection.box = detail::bounding_rect_from_points(detection.points);
  return detection;
}

template <CameraTypes::CameraInfo CameraInfoV>
typename ArmorDetector<CameraInfoV>::CandidateArmor
ArmorDetector<CameraInfoV>::BuildCandidateArmor(
    const NetworkDetection& detection, const cv::Mat& bgr_img) const
{
  CandidateArmor armor;
  armor.color = detection.color;
  armor.number = detection.number;
  armor.confidence = detection.confidence;
  armor.box = detection.box;
  armor.points = detection.points;
  armor.center = detail::quad_center(armor.points);
  armor.center_norm = GetNormalizedCenter(bgr_img, armor.center);
  UpdateGeometryMetrics(armor);
  return armor;
}

template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorDetector<CameraInfoV>::RefineArmorCorners(
    CandidateArmor& armor, const cv::Mat& bgr_img) const
{
  const cv::Point2f top_left = armor.points[0];
  const cv::Point2f top_right = armor.points[1];
  const cv::Point2f bottom_right = armor.points[2];
  const cv::Point2f bottom_left = armor.points[3];

  const cv::Point2f left_to_bottom = bottom_left - top_left;
  const cv::Point2f right_to_bottom = bottom_right - top_right;
  const cv::Point2f top_left_1 = (top_left + bottom_left) * 0.5F - left_to_bottom;
  const cv::Point2f bottom_left_1 = (top_left + bottom_left) * 0.5F + left_to_bottom;
  const cv::Point2f bottom_right_1 =
      (top_right + bottom_right) * 0.5F + right_to_bottom;
  const cv::Point2f top_right_1 = (top_right + bottom_right) * 0.5F - right_to_bottom;

  const cv::Point2f top_left_to_top_right = top_right_1 - top_left_1;
  const cv::Point2f bottom_left_to_bottom_right = bottom_right_1 - bottom_left_1;
  const cv::Point2f top_left_2 =
      (top_left_1 + top_right) * 0.5F - 0.75F * top_left_to_top_right;
  const cv::Point2f top_right_2 =
      (top_left_1 + top_right) * 0.5F + 0.75F * top_left_to_top_right;
  const cv::Point2f bottom_left_2 =
      (bottom_left_1 + bottom_right) * 0.5F - 0.75F * bottom_left_to_bottom_right;
  const cv::Point2f bottom_right_2 =
      (bottom_left_1 + bottom_right) * 0.5F + 0.75F * bottom_left_to_bottom_right;

  std::vector<cv::Point> points = {top_left_2, top_right_2, bottom_right_2, bottom_left_2};
  const cv::Rect bounding_box = cv::minAreaRect(points).boundingRect();
  if (bounding_box.x < 0 || bounding_box.y < 0 ||
      (bounding_box.x + bounding_box.width) > bgr_img.cols ||
      (bounding_box.y + bounding_box.height) > bgr_img.rows)
  {
    return false;
  }

  const cv::Mat armor_roi = bgr_img(bounding_box);
  if (armor_roi.empty())
  {
    return false;
  }

  cv::Mat gray_img;
  cv::cvtColor(armor_roi, gray_img, cv::COLOR_BGR2GRAY);
  cv::Mat binary_img;
  cv::threshold(gray_img, binary_img, cfg_.traditional.threshold, 255,
                cv::THRESH_BINARY);

  auto lightbars = DetectLightbars(armor_roi, binary_img);
  if (lightbars.size() < 2U)
  {
    return false;
  }

  Lightbar* closest_left = nullptr;
  Lightbar* closest_right = nullptr;
  float left_distance = std::numeric_limits<float>::max();
  float right_distance = std::numeric_limits<float>::max();

  for (auto& lightbar : lightbars)
  {
    const cv::Point2f roi_offset(static_cast<float>(bounding_box.x),
                                 static_cast<float>(bounding_box.y));
    const float top_left_bottom_distance =
        cv::norm(top_left - (lightbar.top + roi_offset)) +
        cv::norm(bottom_left - (lightbar.bottom + roi_offset));
    if (top_left_bottom_distance < left_distance)
    {
      left_distance = top_left_bottom_distance;
      closest_left = &lightbar;
    }

    const float bottom_right_top_distance =
        cv::norm(bottom_right - (lightbar.bottom + roi_offset)) +
        cv::norm(top_right - (lightbar.top + roi_offset));
    if (bottom_right_top_distance < right_distance)
    {
      right_distance = bottom_right_top_distance;
      closest_right = &lightbar;
    }
  }

  if (closest_left == nullptr || closest_right == nullptr ||
      (left_distance + right_distance) >= 15.0F)
  {
    return false;
  }

  const cv::Point2f roi_offset(static_cast<float>(bounding_box.x),
                               static_cast<float>(bounding_box.y));
  armor.points[0] = closest_left->top + roi_offset;
  armor.points[1] = closest_right->top + roi_offset;
  armor.points[2] = closest_right->bottom + roi_offset;
  armor.points[3] = closest_left->bottom + roi_offset;
  armor.center = detail::quad_center(armor.points);
  armor.center_norm = GetNormalizedCenter(bgr_img, armor.center);
  armor.box = cv::boundingRect(
      std::vector<cv::Point2f>(armor.points.begin(), armor.points.end()));
  UpdateGeometryMetrics(armor);
  return true;
}

template <CameraTypes::CameraInfo CameraInfoV>
std::vector<typename ArmorDetector<CameraInfoV>::Lightbar>
ArmorDetector<CameraInfoV>::DetectLightbars(const cv::Mat& bgr_img,
                                            const cv::Mat& binary_img) const
{
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  std::vector<Lightbar> lightbars;
  lightbars.reserve(contours.size());
  std::size_t lightbar_id = 0U;

  for (const auto& contour : contours)
  {
    const auto rotated_rect = cv::minAreaRect(contour);
    Lightbar lightbar;
    lightbar.id = lightbar_id++;
    lightbar.rect = rotated_rect;
    lightbar.center = rotated_rect.center;

    std::vector<cv::Point2f> corners(4);
    rotated_rect.points(corners.data());
    std::sort(corners.begin(), corners.end(),
              [](const cv::Point2f& lhs, const cv::Point2f& rhs)
              {
                return lhs.y < rhs.y;
              });

    lightbar.top = (corners[0] + corners[1]) * 0.5F;
    lightbar.bottom = (corners[2] + corners[3]) * 0.5F;
    lightbar.top_to_bottom = lightbar.bottom - lightbar.top;
    lightbar.width = cv::norm(corners[0] - corners[1]);
    lightbar.length = cv::norm(lightbar.top_to_bottom);
    lightbar.ratio = lightbar.length / std::max(lightbar.width, 1e-6);
    lightbar.angle = std::atan2(lightbar.top_to_bottom.y, lightbar.top_to_bottom.x);
    lightbar.angle_error = std::abs(lightbar.angle - CV_PI / 2.0);

    if (!ValidateLightbar(lightbar))
    {
      continue;
    }

    lightbar.color = GetContourColor(bgr_img, contour);
    lightbars.emplace_back(lightbar);
  }

  std::sort(lightbars.begin(), lightbars.end(),
            [](const Lightbar& lhs, const Lightbar& rhs)
            {
              return lhs.center.x < rhs.center.x;
            });
  return lightbars;
}

template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorDetector<CameraInfoV>::ValidateLightbar(const Lightbar& lightbar) const
{
  const double max_angle_error = cfg_.traditional.max_angle_error_deg * detail::deg2rad;
  return lightbar.angle_error < max_angle_error &&
         lightbar.ratio > cfg_.traditional.min_lightbar_ratio &&
         lightbar.ratio < cfg_.traditional.max_lightbar_ratio &&
         lightbar.length > cfg_.traditional.min_lightbar_length;
}

template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorDetector<CameraInfoV>::ValidateArmorType(const CandidateArmor& armor) const
{
  if (armor.type == ArmorType::SMALL)
  {
    return !ArmorNumberIsLarge(armor.number);
  }

  return !ArmorNumberIsSmall(armor.number);
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorDetector<CameraInfoV>::UpdateGeometryMetrics(CandidateArmor& armor) const
{
  const cv::Point2f left_center = (armor.points[0] + armor.points[3]) * 0.5F;
  const cv::Point2f right_center = (armor.points[1] + armor.points[2]) * 0.5F;
  const cv::Point2f left_light = armor.points[3] - armor.points[0];
  const cv::Point2f right_light = armor.points[2] - armor.points[1];
  const cv::Point2f left_to_right = right_center - left_center;

  const double width = cv::norm(left_to_right);
  const double left_length = cv::norm(left_light);
  const double right_length = cv::norm(right_light);
  const double max_lightbar_length = std::max(left_length, right_length);

  armor.ratio = width / std::max(max_lightbar_length, 1e-6);
}

template <CameraTypes::CameraInfo CameraInfoV>
ArmorType ArmorDetector<CameraInfoV>::InferArmorType(const CandidateArmor& armor) const
{
  if (armor.ratio > 3.0)
  {
    return ArmorType::LARGE;
  }
  if (armor.ratio < 2.5)
  {
    return ArmorType::SMALL;
  }

  if (ArmorNumberIsLarge(armor.number))
  {
    return ArmorType::LARGE;
  }
  return ArmorType::SMALL;
}

template <CameraTypes::CameraInfo CameraInfoV>
cv::Point2f ArmorDetector<CameraInfoV>::GetNormalizedCenter(
    const cv::Mat& bgr_img, const cv::Point2f& center) const
{
  return {center.x / static_cast<float>(std::max(1, bgr_img.cols)),
          center.y / static_cast<float>(std::max(1, bgr_img.rows))};
}

template <CameraTypes::CameraInfo CameraInfoV>
ArmorColor ArmorDetector<CameraInfoV>::GetContourColor(
    const cv::Mat& bgr_img, const std::vector<cv::Point>& contour) const
{
  int red_sum = 0;
  int blue_sum = 0;
  for (const auto& point : contour)
  {
    const cv::Vec3b pixel = bgr_img.at<cv::Vec3b>(point);
    blue_sum += pixel[0];
    red_sum += pixel[2];
  }
  return (blue_sum > red_sum) ? ArmorColor::BLUE : ArmorColor::RED;
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorDetector<CameraInfoV>::FillResultMessage(
    const std::vector<CandidateArmor>& armors, const cv::Mat& bgr_img)
{
  armors_msg_.image_timestamp_us = latest_timestamp_us_;
  armors_msg_.results.clear();
  armors_msg_.results.reserve(armors.size());

  for (const auto& armor : armors)
  {
    ArmorDetectorResult result;
    result.color = armor.color;
    result.number = armor.number;
    result.type = armor.type;
    result.priority = GetArmorPriority(armor.number);
    result.confidence = armor.confidence;
    result.box = armor.box;
    result.points = armor.points;
    result.center = armor.center;
    result.center_norm = GetNormalizedCenter(bgr_img, armor.center);
    result.distance_to_image_center = pnp_solver_.CalculateDistanceToCenter(armor.center);

    cv::Mat rvec;
    cv::Mat tvec;
    if (pnp_solver_.SolvePnP(armor.points, armor.type, rvec, tvec))
    {
      result.pose = detail::make_pose(rvec, tvec);
    }

    armors_msg_.results.emplace_back(std::move(result));
  }
}
