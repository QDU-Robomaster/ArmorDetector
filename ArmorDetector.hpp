#pragma once

// clang-format off
/* === MODULE MANIFEST V2 ===
module_description: armor detector with openvino model, optional corner refine, and pnp
constructor_args:
  cfg:
    detect_color: 1
    traditional:
      threshold: 150.0
      max_angle_error_deg: 45.0
      min_lightbar_ratio: 1.5
      max_lightbar_ratio: 20.0
      min_lightbar_length: 8.0
    yolo:
      use_roi: false
      roi_x: 420
      roi_y: 50
      roi_width: 600
      roi_height: 600
      use_traditional_refine: true
      score_threshold: 0.7
      nms_threshold: 0.3
      min_confidence: 0.8
    debug:
      preview: false
      show_binary: false
      wait_key_ms: 1
      overlay_scale: 0.75
  sync: @camera_frame_sync
template_args:
  - Info:
      width: 1280
      height: 720
      step: 3840
      encoding: CameraTypes::Encoding::BGR8
      camera_matrix: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
      distortion_model: CameraTypes::DistortionModel::PLUMB_BOB
      distortion_coefficients: [0.0, 0.0, 0.0, 0.0, 0.0]
      rectification_matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
      projection_matrix: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
required_hardware: []
depends:
  - qdu-future/CameraFrameSync
=== END MANIFEST === */
// clang-format on

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

#include "CameraFrameSync.hpp"
#include "app_framework.hpp"
#include "armor.hpp"
#include "libxr.hpp"
#include "logger.hpp"
#include "pnp_solver.hpp"

template <CameraTypes::CameraInfo CameraInfoV>
class ArmorDetector : public LibXR::Application
{
 public:
  using Sync = CameraFrameSync<CameraInfoV>;
  using CameraInfo = typename Sync::CameraInfo;
  using SyncedFrame = typename Sync::SyncedFrame;

  static inline constexpr CameraInfo kCameraInfo = CameraInfoV;

  struct TraditionalParams
  {
    double threshold{150.0};
    double max_angle_error_deg{45.0};
    double min_lightbar_ratio{1.5};
    double max_lightbar_ratio{20.0};
    double min_lightbar_length{8.0};
  };

  struct YoloParams
  {
    bool use_roi{false};
    int roi_x{420};
    int roi_y{50};
    int roi_width{600};
    int roi_height{600};
    bool use_traditional_refine{true};
    double score_threshold{0.7};
    double nms_threshold{0.3};
    double min_confidence{0.8};
  };

  struct DebugParams
  {
    bool preview{false};
    bool show_binary{false};
    int wait_key_ms{1};
    double overlay_scale{0.75};
  };

  struct Config
  {
    // 0 = red, 1 = blue, 2 = any
    int detect_color{1};
    TraditionalParams traditional{};
    YoloParams yolo{};
    DebugParams debug{};
  };

  ArmorDetector(LibXR::HardwareContainer& hw, LibXR::ApplicationManager& app, Config cfg,
                Sync& sync);

  void SetConfig(const Config& cfg);
  void OnMonitor() override {}

 private:
  struct Lightbar
  {
    std::size_t id{0};
    ArmorColor color{ArmorColor::UNKNOWN};
    cv::RotatedRect rect{};
    cv::Point2f center{};
    cv::Point2f top{};
    cv::Point2f bottom{};
    cv::Point2f top_to_bottom{};
    double angle{0.0};
    double angle_error{0.0};
    double length{0.0};
    double width{0.0};
    double ratio{0.0};
  };

  struct CandidateArmor
  {
    // 经过网络解码、可选角点细化和类型判定后的内部装甲板表示。
    ArmorColor color{ArmorColor::UNKNOWN};
    ArmorType type{ArmorType::INVALID};
    ArmorNumber number{ArmorNumber::INVALID};
    float confidence{0.0F};
    cv::Rect box{};
    std::array<cv::Point2f, 4> points{};
    cv::Point2f center{};
    cv::Point2f center_norm{};
    double ratio{0.0};
  };

  struct NetworkDetection
  {
    // 网络输出的最小语义单元，还没有经过 refine 和类型修正。
    ArmorColor color{ArmorColor::UNKNOWN};
    ArmorNumber number{ArmorNumber::UNKNOWN};
    float confidence{0.0F};
    cv::Rect box{};
    std::array<cv::Point2f, 4> points{};
  };

  void ProcessImage(const cv::Mat& img_msg, uint64_t image_timestamp_us);
  void ProcessSyncedFrame(const SyncedFrame& frame);
  static void SyncFrameThreadFun(ArmorDetector<CameraInfoV>* self);

  std::vector<CandidateArmor> Detect(const cv::Mat& bgr_img, cv::Mat* binary_debug);
  std::vector<CandidateArmor> DecodeOutput(double scale, const cv::Mat& output,
                                           const cv::Mat& bgr_img);
  std::optional<NetworkDetection> DecodeDetection(
      double scale, const cv::Mat& output, int row) const;
  CandidateArmor BuildCandidateArmor(const NetworkDetection& detection,
                                     const cv::Mat& bgr_img) const;
  bool RefineArmorCorners(CandidateArmor& armor, const cv::Mat& bgr_img) const;
  std::vector<Lightbar> DetectLightbars(const cv::Mat& bgr_img,
                                        const cv::Mat& binary_img) const;
  bool ValidateLightbar(const Lightbar& lightbar) const;
  bool ValidateArmorType(const CandidateArmor& armor) const;
  void UpdateGeometryMetrics(CandidateArmor& armor) const;
  ArmorType InferArmorType(const CandidateArmor& armor) const;
  cv::Point2f GetNormalizedCenter(const cv::Mat& bgr_img,
                                  const cv::Point2f& center) const;
  ArmorColor GetContourColor(const cv::Mat& bgr_img,
                             const std::vector<cv::Point>& contour) const;
  void FillResultMessage(const std::vector<CandidateArmor>& armors,
                         const cv::Mat& bgr_img);
  void ShowDebugPreview(const cv::Mat& bgr_img, const cv::Mat* binary_debug);
  bool ShouldShowPreview();

 private:
  Config cfg_{};
  Sync& sync_;
  PnPSolver<CameraInfoV> pnp_solver_{};
  uint64_t latest_timestamp_us_{0};
  uint64_t frame_index_{0};
  LibXR::Thread sync_frame_thread_{};
  uint32_t refined_count_{0};
  uint32_t discarded_count_{0};
  bool model_ready_{false};
  bool preview_available_{true};
  bool preview_warned_{false};

  ov::Core ov_core_{};
  ov::CompiledModel compiled_model_{};
  ov::InferRequest infer_request_{};

  ArmorDetectionsMessage armors_msg_{};
  ArmorDetectorMetrics metrics_msg_{};

  LibXR::Topic::Domain armor_domain_ = LibXR::Topic::Domain("armor_detector");
  LibXR::Topic armors_topic_ =
      LibXR::Topic("armors_result", sizeof(ArmorDetectionsMessage), &armor_domain_);
  LibXR::Topic metrics_topic_ =
      LibXR::Topic("metrics", sizeof(ArmorDetectorMetrics), &armor_domain_);
};

namespace armor_detector_detail
{
// 只保留与模块主体无关的低层工具，避免 detector 主逻辑里充满魔法数字和绘图细节。
constexpr double deg2rad = CV_PI / 180.0;
constexpr int yolo_input_size = 640;
constexpr int info_panel_width = 360;
constexpr int max_debug_armors = 6;
constexpr double header_bar_alpha = 0.78;
constexpr float point_radius = 4.0F;
constexpr int preview_header_height = 54;
constexpr uint32_t sync_frame_wait_timeout_ms = 100;
constexpr uint32_t sync_frame_retry_sleep_ms = 200;
constexpr uint32_t metrics_log_period = 30;
constexpr size_t sync_frame_thread_stack_size = 1024U * 128U;

struct OutputLayout
{
  static constexpr int objectness_index = 8;
  static constexpr int color_begin = 9;
  static constexpr int color_end = 13;
  static constexpr int number_begin = 13;
  static constexpr int number_end = 22;
};

inline int CvTypeFromEncoding(CameraTypes::Encoding encoding)
{
  switch (encoding)
  {
    case CameraTypes::Encoding::RGB8:
    case CameraTypes::Encoding::BGR8:
      return CV_8UC3;
    case CameraTypes::Encoding::RGBA8:
    case CameraTypes::Encoding::BGRA8:
      return CV_8UC4;
    case CameraTypes::Encoding::MONO8:
      return CV_8UC1;
    default:
      return -1;
  }
}

inline cv::Mat ConvertToBgrWithEncoding(const cv::Mat& input,
                                        CameraTypes::Encoding encoding)
{
  switch (encoding)
  {
    case CameraTypes::Encoding::RGB8:
    {
      cv::Mat output;
      cv::cvtColor(input, output, cv::COLOR_RGB2BGR);
      return output;
    }
    case CameraTypes::Encoding::BGRA8:
    {
      cv::Mat output;
      cv::cvtColor(input, output, cv::COLOR_BGRA2BGR);
      return output;
    }
    case CameraTypes::Encoding::RGBA8:
    {
      cv::Mat output;
      cv::cvtColor(input, output, cv::COLOR_RGBA2BGR);
      return output;
    }
    default:
      return input;
  }
}

inline ArmorColor detect_color_from_config(int detect_color)
{
  if (detect_color == 0)
  {
    return ArmorColor::RED;
  }
  if (detect_color == 1)
  {
    return ArmorColor::BLUE;
  }
  return ArmorColor::UNKNOWN;
}

inline ArmorColor color_from_yolo_id(int color_id)
{
  if (color_id == 0)
  {
    return ArmorColor::BLUE;
  }
  if (color_id == 1)
  {
    return ArmorColor::RED;
  }
  if (color_id == 2)
  {
    return ArmorColor::EXTINGUISH;
  }
  return ArmorColor::UNKNOWN;
}

inline ArmorNumber number_from_yolo_id(int number_id)
{
  switch (number_id)
  {
    case 0:
      return ArmorNumber::GUARD;
    case 1:
      return ArmorNumber::ONE;
    case 2:
      return ArmorNumber::TWO;
    case 3:
      return ArmorNumber::THREE;
    case 4:
      return ArmorNumber::FOUR;
    case 5:
      return ArmorNumber::FIVE;
    case 6:
      return ArmorNumber::OUTPOST;
    case 7:
      return ArmorNumber::BASE;
    default:
      return ArmorNumber::UNKNOWN;
  }
}

inline std::array<cv::Point2f, 4> sort_keypoints(
    const std::array<cv::Point2f, 4>& keypoints)
{
  std::array<cv::Point2f, 4> sorted = keypoints;

  std::sort(sorted.begin(), sorted.end(),
            [](const cv::Point2f& lhs, const cv::Point2f& rhs)
            {
              return lhs.y < rhs.y;
            });

  std::array<cv::Point2f, 2> top_points = {sorted[0], sorted[1]};
  std::array<cv::Point2f, 2> bottom_points = {sorted[2], sorted[3]};
  std::sort(top_points.begin(), top_points.end(),
            [](const cv::Point2f& lhs, const cv::Point2f& rhs)
            {
              return lhs.x < rhs.x;
            });
  std::sort(bottom_points.begin(), bottom_points.end(),
            [](const cv::Point2f& lhs, const cv::Point2f& rhs)
            {
              return lhs.x < rhs.x;
            });

  return {top_points[0], top_points[1], bottom_points[1], bottom_points[0]};
}

inline std::string armor_number_to_string(ArmorNumber number)
{
  const std::size_t index = static_cast<std::size_t>(number);
  if (index >= ARMOR_NUMBER_NAMES.size())
  {
    return "invalid";
  }
  return std::string(ARMOR_NUMBER_NAMES[index]);
}

inline std::string armor_type_to_string(ArmorType type)
{
  const std::size_t index = static_cast<std::size_t>(type);
  if (index >= ARMOR_TYPE_NAMES.size())
  {
    return "invalid";
  }
  return std::string(ARMOR_TYPE_NAMES[index]);
}

inline std::string format_float(double value, int precision = 3)
{
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

inline cv::Scalar color_to_scalar(ArmorColor color)
{
  switch (color)
  {
    case ArmorColor::BLUE:
      return cv::Scalar(255, 180, 40);
    case ArmorColor::RED:
      return cv::Scalar(60, 90, 255);
    case ArmorColor::EXTINGUISH:
      return cv::Scalar(180, 180, 180);
    case ArmorColor::PURPLE:
      return cv::Scalar(220, 70, 220);
    case ArmorColor::UNKNOWN:
    default:
      return cv::Scalar(150, 220, 150);
  }
}

inline void draw_label_chip(cv::Mat& canvas, const std::string& text,
                            const cv::Point& origin, const cv::Scalar& color)
{
  constexpr int FONT = cv::FONT_HERSHEY_DUPLEX;
  constexpr double FONT_SCALE = 0.55;
  constexpr int THICKNESS = 1;
  constexpr int PADDING_X = 8;
  constexpr int PADDING_Y = 6;

  int baseline = 0;
  const cv::Size text_size =
      cv::getTextSize(text, FONT, FONT_SCALE, THICKNESS, &baseline);
  const cv::Rect bg_rect(origin.x, origin.y - text_size.height - PADDING_Y,
                         text_size.width + 2 * PADDING_X,
                         text_size.height + 2 * PADDING_Y);
  cv::rectangle(canvas, bg_rect, color, cv::FILLED, cv::LINE_AA);
  cv::putText(canvas, text,
              cv::Point(origin.x + PADDING_X,
                        origin.y - PADDING_Y + baseline / 2),
              FONT, FONT_SCALE, cv::Scalar(12, 16, 24), THICKNESS, cv::LINE_AA);
}

inline void draw_info_row(cv::Mat& canvas, int x, int y, const std::string& key,
                          const std::string& value,
                          const cv::Scalar& value_color)
{
  constexpr int FONT = cv::FONT_HERSHEY_DUPLEX;
  constexpr double FONT_SCALE = 0.55;
  constexpr int THICKNESS = 1;

  cv::putText(canvas, key, cv::Point(x, y), FONT, FONT_SCALE,
              cv::Scalar(170, 182, 196), THICKNESS, cv::LINE_AA);
  cv::putText(canvas, value, cv::Point(x + 145, y), FONT, FONT_SCALE, value_color,
              THICKNESS, cv::LINE_AA);
}

inline LibXR::Transform<double> make_pose(const cv::Mat& rvec, const cv::Mat& tvec)
{
  cv::Mat rotation_cv;
  cv::Rodrigues(rvec, rotation_cv);

  Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
  for (int row = 0; row < 3; ++row)
  {
    for (int col = 0; col < 3; ++col)
    {
      rotation_matrix(row, col) = rotation_cv.at<double>(row, col);
    }
  }

  return LibXR::Transform<double>(
      LibXR::Quaternion<double>(rotation_matrix),
      LibXR::Position<double>(tvec.at<double>(0), tvec.at<double>(1),
                              tvec.at<double>(2)));
}

inline cv::Point2f quad_center(const std::array<cv::Point2f, 4>& points)
{
  return (points[0] + points[1] + points[2] + points[3]) * 0.25F;
}

inline cv::Rect bounding_rect_from_points(
    const std::array<cv::Point2f, 4>& points)
{
  float min_x = points[0].x;
  float max_x = points[0].x;
  float min_y = points[0].y;
  float max_y = points[0].y;
  for (const auto& point : points)
  {
    min_x = std::min(min_x, point.x);
    max_x = std::max(max_x, point.x);
    min_y = std::min(min_y, point.y);
    max_y = std::max(max_y, point.y);
  }

  return {static_cast<int>(min_x), static_cast<int>(min_y),
          std::max(1, static_cast<int>(max_x - min_x)),
          std::max(1, static_cast<int>(max_y - min_y))};
}

inline const char* target_color_name(ArmorColor color)
{
  if (color == ArmorColor::BLUE)
  {
    return "blue";
  }
  if (color == ArmorColor::RED)
  {
    return "red";
  }
  return "any";
}

inline std::string armor_display_name(ArmorNumber number, ArmorType type)
{
  return armor_number_to_string(number) + " / " + armor_type_to_string(type);
}
}  // namespace armor_detector_detail

namespace detail = armor_detector_detail;

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

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorDetector<CameraInfoV>::ShowDebugPreview(const cv::Mat& bgr_img,
                                                  const cv::Mat* binary_debug)
{
  try
  {
    cv::Mat canvas(bgr_img.rows, bgr_img.cols + detail::info_panel_width, CV_8UC3,
                   cv::Scalar(18, 22, 28));
    bgr_img.copyTo(canvas(cv::Rect(0, 0, bgr_img.cols, bgr_img.rows)));

    cv::Mat header =
        canvas(cv::Rect(0, 0, bgr_img.cols, detail::preview_header_height));
    cv::Mat header_overlay = header.clone();
    cv::rectangle(header_overlay, cv::Rect(0, 0, header.cols, header.rows),
                  cv::Scalar(10, 16, 22), cv::FILLED);
    cv::addWeighted(header_overlay, detail::header_bar_alpha, header,
                    1.0 - detail::header_bar_alpha, 0.0, header);

    cv::putText(canvas, "ArmorDetector Preview", cv::Point(18, 34),
                cv::FONT_HERSHEY_DUPLEX, 0.85, cv::Scalar(240, 244, 250), 1,
                cv::LINE_AA);
    cv::putText(canvas,
                cfg_.yolo.use_roi ? "YOLOv5 + OpenVINO [ROI]" : "YOLOv5 + OpenVINO [FULL]",
                cv::Point(18, 52), cv::FONT_HERSHEY_DUPLEX, 0.48,
                cv::Scalar(151, 170, 192), 1, cv::LINE_AA);

    for (std::size_t index = 0; index < armors_msg_.results.size(); ++index)
    {
      const auto& armor = armors_msg_.results[index];
      const cv::Scalar armor_color = detail::color_to_scalar(armor.color);

      std::array<cv::Point, 4> polygon{};
      for (std::size_t point_index = 0; point_index < armor.points.size(); ++point_index)
      {
        polygon[point_index] = armor.points[point_index];
      }

      const cv::Point* polygon_points = polygon.data();
      const int polygon_size = static_cast<int>(polygon.size());
      cv::polylines(canvas, &polygon_points, &polygon_size, 1, true, armor_color, 2,
                    cv::LINE_AA);
      cv::rectangle(canvas, armor.box, armor_color, 1, cv::LINE_AA);

      for (const auto& point : armor.points)
      {
        cv::circle(canvas, point, static_cast<int>(detail::point_radius), armor_color,
                   cv::FILLED, cv::LINE_AA);
      }

      std::ostringstream label;
      label << detail::armor_number_to_string(armor.number) << " "
            << detail::armor_type_to_string(armor.type) << " "
            << std::fixed << std::setprecision(2) << armor.confidence;
      const cv::Point label_origin(
          std::max(armor.box.x, 6),
          std::max(armor.box.y - 6, 24));
      detail::draw_label_chip(canvas, label.str(), label_origin, armor_color);
    }

    const int panel_x = bgr_img.cols + 18;
    int panel_y = 42;
    cv::putText(canvas, "Frame Stats", cv::Point(panel_x, panel_y),
                cv::FONT_HERSHEY_DUPLEX, 0.72, cv::Scalar(243, 246, 250), 1,
                cv::LINE_AA);
    panel_y += 28;
    detail::draw_info_row(canvas, panel_x, panel_y, "frame",
                          std::to_string(metrics_msg_.frame_index),
                          cv::Scalar(240, 244, 250));
    panel_y += 24;
    detail::draw_info_row(canvas, panel_x, panel_y, "armors",
                          std::to_string(metrics_msg_.armor_count),
                          cv::Scalar(128, 226, 142));
    panel_y += 24;
    detail::draw_info_row(canvas, panel_x, panel_y, "refined",
                          std::to_string(metrics_msg_.refined_count),
                          cv::Scalar(91, 196, 255));
    panel_y += 24;
    detail::draw_info_row(canvas, panel_x, panel_y, "discarded",
                          std::to_string(metrics_msg_.discarded_count),
                          cv::Scalar(255, 166, 77));
    panel_y += 24;
    detail::draw_info_row(canvas, panel_x, panel_y, "detector_ms",
                          detail::format_float(metrics_msg_.detector_latency_ms, 2),
                          cv::Scalar(255, 214, 102));
    panel_y += 24;
    detail::draw_info_row(canvas, panel_x, panel_y, "publish_ms",
                          detail::format_float(metrics_msg_.publish_latency_ms, 2),
                          cv::Scalar(221, 235, 255));

    panel_y += 36;
    cv::putText(canvas, "Detector Config", cv::Point(panel_x, panel_y),
                cv::FONT_HERSHEY_DUPLEX, 0.72, cv::Scalar(243, 246, 250), 1,
                cv::LINE_AA);
    panel_y += 28;
    const ArmorColor configured_target_color =
        detail::detect_color_from_config(cfg_.detect_color);
    detail::draw_info_row(canvas, panel_x, panel_y, "target_color",
                          detail::target_color_name(configured_target_color),
                          cv::Scalar(240, 244, 250));
    panel_y += 24;
    detail::draw_info_row(canvas, panel_x, panel_y, "score_thres",
                          detail::format_float(cfg_.yolo.score_threshold, 2),
                          cv::Scalar(240, 244, 250));
    panel_y += 24;
    detail::draw_info_row(canvas, panel_x, panel_y, "min_conf",
                          detail::format_float(cfg_.yolo.min_confidence, 2),
                          cv::Scalar(240, 244, 250));
    panel_y += 24;
    detail::draw_info_row(canvas, panel_x, panel_y, "traditional_refine",
                          cfg_.yolo.use_traditional_refine ? "on" : "off",
                          cfg_.yolo.use_traditional_refine
                              ? cv::Scalar(128, 226, 142)
                              : cv::Scalar(255, 166, 77));

    panel_y += 36;
    cv::putText(canvas, "Detections", cv::Point(panel_x, panel_y),
                cv::FONT_HERSHEY_DUPLEX, 0.72, cv::Scalar(243, 246, 250), 1,
                cv::LINE_AA);
    panel_y += 26;

    if (armors_msg_.results.empty())
    {
      cv::putText(canvas, "No target in current frame", cv::Point(panel_x, panel_y),
                  cv::FONT_HERSHEY_DUPLEX, 0.54, cv::Scalar(151, 170, 192), 1,
                  cv::LINE_AA);
    }
    else
    {
      const int armor_count = std::min(static_cast<int>(armors_msg_.results.size()),
                                       detail::max_debug_armors);
      for (int index = 0; index < armor_count; ++index)
      {
        const auto& armor = armors_msg_.results[index];
        const cv::Scalar armor_color = detail::color_to_scalar(armor.color);
        const cv::Rect item_rect(panel_x - 10, panel_y - 18,
                                 detail::info_panel_width - 32, 52);
        cv::rectangle(canvas, item_rect, cv::Scalar(32, 39, 48), cv::FILLED,
                      cv::LINE_AA);
        cv::rectangle(canvas, item_rect, armor_color, 1, cv::LINE_AA);

        const std::string name =
            detail::armor_display_name(armor.number, armor.type);
        cv::putText(canvas, name, cv::Point(panel_x, panel_y), cv::FONT_HERSHEY_DUPLEX,
                    0.56, cv::Scalar(245, 247, 250), 1, cv::LINE_AA);

        std::ostringstream value;
        value << "conf=" << std::fixed << std::setprecision(2) << armor.confidence
              << "  px=" << static_cast<int>(armor.center.x) << ","
              << static_cast<int>(armor.center.y);
        cv::putText(canvas, value.str(), cv::Point(panel_x, panel_y + 20),
                    cv::FONT_HERSHEY_DUPLEX, 0.44, armor_color, 1, cv::LINE_AA);
        panel_y += 60;
      }
    }

    if (binary_debug != nullptr && !binary_debug->empty())
    {
      cv::Mat preview_binary;
      cv::cvtColor(*binary_debug, preview_binary, cv::COLOR_GRAY2BGR);

      const int thumb_width = detail::info_panel_width - 36;
      const int thumb_height = std::max(120, thumb_width * preview_binary.rows /
                                                 std::max(1, preview_binary.cols));
      cv::Mat thumb_resized;
      cv::resize(preview_binary, thumb_resized, cv::Size(thumb_width, thumb_height));

      int thumb_y = std::max(bgr_img.rows - thumb_resized.rows - 26, panel_y + 16);
      if ((thumb_y + thumb_resized.rows) > canvas.rows)
      {
        thumb_y = canvas.rows - thumb_resized.rows - 16;
      }
      cv::putText(canvas, "Binary Threshold", cv::Point(panel_x, thumb_y - 8),
                  cv::FONT_HERSHEY_DUPLEX, 0.56, cv::Scalar(243, 246, 250), 1,
                  cv::LINE_AA);
      thumb_resized.copyTo(
          canvas(cv::Rect(panel_x, thumb_y, thumb_resized.cols, thumb_resized.rows)));
      cv::rectangle(canvas, cv::Rect(panel_x, thumb_y, thumb_resized.cols, thumb_resized.rows),
                    cv::Scalar(91, 196, 255), 1, cv::LINE_AA);
    }

    cv::Mat display = canvas;
    if (std::abs(cfg_.debug.overlay_scale - 1.0) > 1e-6)
    {
      cv::resize(canvas, display, cv::Size(), cfg_.debug.overlay_scale,
                 cfg_.debug.overlay_scale);
    }

    cv::imshow("armor_detector_debug", display);
    cv::waitKey(std::max(cfg_.debug.wait_key_ms, 1));
  }
  catch (const cv::Exception& exception)
  {
    preview_available_ = false;
    if (!preview_warned_)
    {
      preview_warned_ = true;
      XR_LOG_WARN("ArmorDetector preview disabled: %s", exception.what());
    }
  }
}

template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorDetector<CameraInfoV>::ShouldShowPreview()
{
  if (!cfg_.debug.preview || !preview_available_)
  {
    return false;
  }

  const char* display = std::getenv("DISPLAY");
  const char* wayland_display = std::getenv("WAYLAND_DISPLAY");
  if (display == nullptr && wayland_display == nullptr)
  {
    preview_available_ = false;
    if (!preview_warned_)
    {
      preview_warned_ = true;
      XR_LOG_WARN("ArmorDetector preview disabled because DISPLAY is unavailable");
    }
    return false;
  }

  return true;
}
