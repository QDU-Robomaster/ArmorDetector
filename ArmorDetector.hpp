#pragma once

// clang-format off
/* === MODULE MANIFEST V2 ===
module_description: sp_vision yolov5 armor detector with openvino
constructor_args:
  cfg:
    detect_color: 1
    traditional:
      threshold: 150.0
      max_angle_error_deg: 45.0
      min_lightbar_ratio: 1.5
      max_lightbar_ratio: 20.0
      min_lightbar_length: 8.0
      min_armor_ratio: 1.0
      max_armor_ratio: 5.0
      max_side_ratio: 1.5
      max_rectangular_error_deg: 25.0
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
  camera_info:
    width: 1280
    height: 720
    step: 3840
    encoding: CameraBase::Encoding::BGR8
    camera_matrix: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    distortion_model: CameraBase::DistortionModel::PLUMB_BOB
    distortion_coefficients: [0.0, 0.0, 0.0, 0.0, 0.0]
    rectification_matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    projection_matrix: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
template_args: []
required_hardware: []
depends:
  - qdu-future/CameraBase
=== END MANIFEST === */
// clang-format on

#include <array>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include <openvino/openvino.hpp>

#include "CameraBase.hpp"
#include "app_framework.hpp"
#include "armor.hpp"
#include "libxr.hpp"
#include "pnp_solver.hpp"

class ArmorDetector : public LibXR::Application
{
 public:
  struct TraditionalParams
  {
    double threshold{150.0};
    double max_angle_error_deg{45.0};
    double min_lightbar_ratio{1.5};
    double max_lightbar_ratio{20.0};
    double min_lightbar_length{8.0};
    double min_armor_ratio{1.0};
    double max_armor_ratio{5.0};
    double max_side_ratio{1.5};
    double max_rectangular_error_deg{25.0};
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
                CameraBase::CameraInfo camera_info);

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
    ArmorColor color{ArmorColor::UNKNOWN};
    ArmorType type{ArmorType::INVALID};
    ArmorNumber number{ArmorNumber::INVALID};
    float confidence{0.0F};
    cv::Rect box{};
    std::array<cv::Point2f, 4> points{};
    cv::Point2f center{};
    cv::Point2f center_norm{};
    double ratio{0.0};
    double side_ratio{0.0};
    double rectangular_error{0.0};
    bool duplicated{false};
  };

  void ImageCallback(cv::Mat* img_msg);
  void HeaderCallback(CameraBase::ImageHeader* image_header);

  cv::Mat ConvertToBgr(const cv::Mat& input) const;
  std::vector<CandidateArmor> Detect(const cv::Mat& bgr_img, cv::Mat* binary_debug);
  std::vector<CandidateArmor> Parse(double scale, cv::Mat& output,
                                    const cv::Mat& bgr_img);
  bool RefineArmorCorners(CandidateArmor& armor, const cv::Mat& bgr_img) const;
  std::vector<Lightbar> DetectLightbars(const cv::Mat& bgr_img,
                                        const cv::Mat& binary_img) const;
  bool ValidateLightbar(const Lightbar& lightbar) const;
  bool ValidateArmorGeometry(const CandidateArmor& armor) const;
  bool ValidateArmorType(const CandidateArmor& armor) const;
  bool HasContainedLightbar(const Lightbar& left, const Lightbar& right,
                            const std::vector<Lightbar>& lightbars) const;
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
  CameraBase::CameraInfo camera_info_{};
  std::unique_ptr<PnPSolver> pnp_solver_{};
  uint64_t latest_timestamp_us_{0};
  uint64_t frame_index_{0};
  uint32_t refined_count_{0};
  uint32_t discarded_count_{0};
  bool model_ready_{false};
  bool preview_available_{true};
  bool preview_warned_{false};

  ov::Core ov_core_{};
  ov::CompiledModel compiled_model_{};

  ArmorDetectionsMessage armors_msg_{};
  ArmorDetectorMetrics metrics_msg_{};

  LibXR::Topic::Domain armor_domain_ = LibXR::Topic::Domain("armor_detector");
  LibXR::Topic armors_topic_ =
      LibXR::Topic("armors_result", sizeof(ArmorDetectionsMessage), &armor_domain_);
  LibXR::Topic metrics_topic_ =
      LibXR::Topic("metrics", sizeof(ArmorDetectorMetrics), &armor_domain_);
};
