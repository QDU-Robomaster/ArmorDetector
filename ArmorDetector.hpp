#pragma once

// clang-format off
/* === MODULE MANIFEST V2 ===
module_description: 基于 OpenVINO 的装甲板检测、可选角点细化和 PnP 位姿估计
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
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "CameraFrameSync.hpp"
#include "app_framework.hpp"
#include "armor.hpp"
#include "libxr.hpp"
#include "logger.hpp"
#include "pnp_solver.hpp"
#include "ArmorDetectorDetail.hpp"
#include "ArmorDetectorNetwork.hpp"

template <CameraTypes::CameraInfo CameraInfoV>
class ArmorDetector : public LibXR::Application
{
 public:
  using Sync = CameraFrameSync<CameraInfoV>;
  using CameraInfo = typename Sync::CameraInfo;
  using ImageFrame = typename Sync::ImageFrame;
  using ImuStamped = typename Sync::ImuStamped;
  using SyncedFrame = typename Sync::SyncedFrame;
  using DetectionPacket = ArmorDetectionsFramePacket<CameraInfoV>;
  using DetectionMessage = ArmorDetectionsFrameMessage<CameraInfoV>;

  static inline constexpr CameraInfo camera_info = CameraInfoV;

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

  struct Config
  {
    // 0 = 红色，1 = 蓝色，2 = 不限制颜色。
    int detect_color{1};
    TraditionalParams traditional{};
    YoloParams yolo{};
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
    double fill_ratio{0.0};
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
    bool raw_points_valid{false};
    bool refined{false};
    std::array<cv::Point2f, 4> raw_points{};
    cv::Point2f center{};
    cv::Point2f center_norm{};
    double ratio{0.0};
  };

  struct NetworkDetection
  {
    // 网络输出的最小语义单元，还没有经过角点细化和类型修正。
    ArmorColor color{ArmorColor::UNKNOWN};
    ArmorNumber number{ArmorNumber::UNKNOWN};
    float confidence{0.0F};
    cv::Rect box{};
    std::array<cv::Point2f, 4> points{};
  };

  struct FrameCounters
  {
    uint32_t decoded_count{0};
    uint32_t nms_count{0};
    uint32_t semantic_kept_count{0};
    uint32_t pnp_success_count{0};
    uint32_t refined_count{0};
    uint32_t refine_attempt_count{0};
    uint32_t refine_fail_bbox_oob_count{0};
    uint32_t refine_fail_roi_empty_count{0};
    uint32_t refine_fail_lightbar_zero_count{0};
    uint32_t refine_fail_lightbar_one_count{0};
    uint32_t refine_fail_pair_distance_count{0};
    uint32_t discarded_count{0};
    uint32_t semantic_discard_count{0};
    uint32_t type_discard_count{0};
    double max_objectness{0.0};
  };

  struct DiagnosticOptions
  {
    bool audit_every_frame{false};
    bool audit_zero_frames{false};
    bool disable_traditional_refine{false};
    bool center_letterbox{false};
    bool yolo_letterbox{false};
    bool dump_refine_fails{false};
    std::string dump_refine_fails_dir{};
    uint32_t dump_refine_fails_max{12};
    uint32_t dump_refine_fails_count{0};
  };

  void ProcessImage(const cv::Mat& img_msg, const SyncedFrame& synced_frame);
  void ProcessSyncedFrame(const SyncedFrame& frame);
  static void SyncFrameThreadFun(ArmorDetector<CameraInfoV>* self);

  std::vector<CandidateArmor> Detect(const cv::Mat& bgr_img);
  std::vector<CandidateArmor> DecodeOutput(double scale,
                                           const cv::Point2f& input_offset,
                                           const cv::Mat& output,
                                           const cv::Mat& bgr_img);
  std::optional<NetworkDetection> DecodeDetection(
      double scale, const cv::Point2f& input_offset, const cv::Mat& output,
      int row) const;
  CandidateArmor BuildCandidateArmor(const NetworkDetection& detection,
                                     const cv::Mat& bgr_img) const;
  cv::Mat BuildTraditionalBinary(const cv::Mat& bgr_img,
                                 ArmorColor target_color) const;
  bool RefineArmorCorners(CandidateArmor& armor, const cv::Mat& bgr_img);
  void MaybeDumpRefineFailure(const char* reason, const CandidateArmor& armor,
                              const cv::Rect& bounding_box,
                              const cv::Mat& armor_roi,
                              const cv::Mat& binary_img,
                              const std::vector<Lightbar>& lightbars);
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

 private:
  Config cfg_{};
  Sync& sync_;
  PnPSolver<CameraInfoV> pnp_solver_{};
  uint64_t latest_timestamp_us_{0};
  uint64_t frame_index_{0};
  LibXR::Thread sync_frame_thread_{};
  FrameCounters counters_{};
  DiagnosticOptions diagnostics_{};
  detail::OpenVinoYoloArmorNetwork network_{};

  ArmorDetectionsPacket armors_packet_{};
  DetectionPacket armors_frame_packet_{};
  ArmorDetectorMetrics metrics_msg_{};

  LibXR::Topic::Domain armor_domain_ = LibXR::Topic::Domain("armor_detector");
  LibXR::Topic armors_topic_ =
      LibXR::Topic("armors_result", sizeof(ArmorDetectionsMessage), &armor_domain_);
  LibXR::Topic armors_frame_topic_ =
      LibXR::Topic("armors_frame", sizeof(DetectionMessage), &armor_domain_);
  LibXR::Topic metrics_topic_ =
      LibXR::Topic("metrics", sizeof(ArmorDetectorMetrics), &armor_domain_);
};

#include "ArmorDetectorPipeline.hpp"
