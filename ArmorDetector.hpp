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
  sync: '@camera_frame_sync'
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
  uint32_t decoded_count_{0};
  uint32_t nms_count_{0};
  uint32_t semantic_kept_count_{0};
  uint32_t pnp_success_count_{0};
  uint32_t refined_count_{0};
  uint32_t discarded_count_{0};
  uint32_t semantic_discard_count_{0};
  uint32_t type_discard_count_{0};
  double max_objectness_{0.0};
  bool model_ready_{false};
  bool preview_available_{true};
  bool preview_warned_{false};
  bool audit_every_frame_{false};
  bool audit_zero_frames_{false};

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

#include "ArmorDetectorDetail.hpp"
#include "ArmorDetectorPipeline.hpp"
#include "ArmorDetectorPreview.hpp"
