#pragma once

// clang-format off
/* === MODULE MANIFEST V2 ===
module_description: 基于 OpenVINO 的装甲板检测和 PnP 位姿估计
constructor_args:
  cfg:
    detect_color: 1
    network:
      use_roi: false
      roi_x: 420
      roi_y: 50
      roi_width: 600
      roi_height: 600
      score_threshold: 0.1
      min_confidence: 0.1
      enable_quad_check: true
      min_quad_area_px: 16.0
      openvino_device: "AUTO_DETECT"
      openvino_performance_mode: "LATENCY"
  sync: '@camera_frame_sync'
template_args:
  - Info:
      width: 1280
      height: 720
      step: 3840
      encoding: CameraTypes::Encoding::BGR8
      camera_matrix: [800.0, 0.0, 640.0, 0.0, 800.0, 360.0, 0.0, 0.0, 1.0]
      distortion_model: CameraTypes::DistortionModel::PLUMB_BOB
      distortion_coefficients: [0.0, 0.0, 0.0, 0.0, 0.0]
      rectification_matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
      projection_matrix: [800.0, 0.0, 640.0, 0.0, 0.0, 800.0, 360.0, 0.0, 0.0, 0.0, 1.0, 0.0]
required_hardware: []
depends:
  - qdu-future/CameraFrameSync
=== END MANIFEST === */
// clang-format on

/**
 * @file ArmorDetector.hpp
 * @brief 装甲板 detector 模块主类声明和配置入口。
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
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

#ifndef ARMOR_DETECTOR_MODEL_PATH
#error "ARMOR_DETECTOR_MODEL_PATH must be defined by ArmorDetector CMakeLists.txt."
#endif

/**
 * @brief 装甲板检测应用模块。
 *
 * ArmorDetector 从 CameraFrameSync 读取已经对齐的图像/IMU 帧，执行 dense-grid
 * keypoint 检测、语义过滤、尺寸类型判断和 PnP 位姿求解，最后向 `armor_detector`
 * domain 发布检测结果、带原始帧引用的结果和运行指标。
 *
 * @tparam CameraInfoV 编译期相机参数，必须与实际图像尺寸、内参和编码一致。
 */
template <CameraTypes::CameraInfo CameraInfoV>
class ArmorDetector : public LibXR::Application
{
 public:
  /// 同步帧来源类型。
  using Sync = CameraFrameSync<CameraInfoV>;
  /// 相机参数类型。
  using CameraInfo = typename Sync::CameraInfo;
  /// 图像帧类型。
  using ImageFrame = typename Sync::ImageFrame;
  /// IMU 样本类型。
  using ImuStamped = typename Sync::ImuStamped;
  /// 图像/IMU 同步帧类型。
  using SyncedFrame = typename Sync::SyncedFrame;
  /// 带源帧引用的结果包。
  using DetectionPacket = ArmorDetectionsFramePacket<CameraInfoV>;
  /// 带源帧引用的 Topic payload。
  using DetectionMessage = ArmorDetectionsFrameMessage<CameraInfoV>;
  /**
   * @brief 当前模块实例绑定的编译期相机参数。
   */
  static inline constexpr CameraInfo camera_info = CameraInfoV;

  /**
   * @brief 网络 detector 和后处理参数。
   */
  struct NetworkParams
  {
    bool use_roi{false};                 ///< 是否只在 ROI 内运行 detector。
    int roi_x{420};                      ///< ROI 左上角 x。
    int roi_y{50};                       ///< ROI 左上角 y。
    int roi_width{600};                  ///< ROI 宽度；负值表示全图宽度。
    int roi_height{600};                 ///< ROI 高度；负值表示全图高度。
    double score_threshold{0.1};         ///< 网络目标置信度门限。
    double min_confidence{0.1};          ///< 语义过滤后的最终置信度门限。
    bool enable_quad_check{true};        ///< 是否检查网络四点凸性和面积。
    double min_quad_area_px{16.0};       ///< 网络四边形最小面积，单位 px^2。
    /// OpenVINO 编译设备；"AUTO_DETECT" 按 NPU、GPU、CPU 顺序自动选择。
    const char* openvino_device{"AUTO_DETECT"};
    /// OpenVINO 性能模式，例如 "LATENCY"、"THROUGHPUT"、"CUMULATIVE_THROUGHPUT"。
    const char* openvino_performance_mode{"LATENCY"};
  };

  /**
   * @brief ArmorDetector 模块配置。
   */
  struct Config
  {
    int detect_color{1};                 ///< 0=红色，1=蓝色，其他=不限制颜色。
    NetworkParams network{};             ///< 网络 detector 参数。
  };

  /**
   * @brief 构造 detector 并启动同步帧 worker。
   * @param hw 硬件容器，当前 detector 不直接取硬件对象。
   * @param app 应用管理器，用于注册本模块。
   * @param cfg detector 配置。
   * @param sync 图像/IMU 同步帧来源。
   */
  ArmorDetector(LibXR::HardwareContainer& hw, LibXR::ApplicationManager& app, Config cfg,
                Sync& sync);

  /**
   * @brief 更新 detector 配置并重新加载对应模型。
   * @param cfg 新配置。
   */
  void SetConfig(const Config& cfg);

  /**
   * @brief LibXR monitor hook，当前 detector 没有周期性 monitor 工作。
   */
  void OnMonitor() override {}

 private:
  /**
   * @brief detector 内部装甲板候选。
   *
   * 候选已经经过网络解码，可进一步做尺寸类型判定，最后再转为
   * ArmorDetectorResult。
   */
  struct CandidateArmor
  {
    ArmorColor color{ArmorColor::UNKNOWN};    ///< 网络判定颜色。
    ArmorType type{ArmorType::INVALID};       ///< 尺寸类型，后处理阶段填充。
    ArmorNumber number{ArmorNumber::INVALID}; ///< 网络判定编号。
    float confidence{0.0F};                    ///< 网络置信度。
    cv::Rect box{};                            ///< 当前候选包围盒。
    std::array<cv::Point2f, 4> points{};       ///< 当前角点，顺序为左上、右上、右下、左下。
    cv::Point2f center{};                      ///< 候选像素中心。
    cv::Point2f center_norm{};                 ///< 宽高归一化中心。
    double ratio{0.0};                         ///< 左右灯条中心距与灯条长度的比例。
  };

  /**
   * @brief 网络输出解码后的最小语义单元。
   *
   * 该结构不包含尺寸类型判定和 PnP 结果。
   */
  struct NetworkDetection
  {
    ArmorColor color{ArmorColor::UNKNOWN};    ///< 网络颜色类别。
    ArmorNumber number{ArmorNumber::UNKNOWN}; ///< 网络编号类别。
    float confidence{0.0F};                    ///< 网络置信度。
    cv::Rect box{};                            ///< 由角点生成的包围盒。
    std::array<cv::Point2f, 4> points{};       ///< 统一顺序后的四角点。
  };

  /**
   * @brief NMS / overlap suppression 前的候选引用。
   */
  struct DetectionSelection
  {
    std::size_t index{0};         ///< detections 中的候选下标。
    float confidence{0.0F};       ///< 用于排序和抑制的置信度。
    cv::Rect box{};               ///< 用于抑制的包围盒。
  };

  /**
   * @brief 单帧 detector 内部计数器。
   */
  struct FrameCounters
  {
    uint32_t decoded_count{0};                 ///< 网络 decoder 保留候选数量。
    uint32_t overlap_kept_count{0};            ///< 交叠抑制后候选数量。
    uint32_t semantic_kept_count{0};           ///< 语义过滤后候选数量。
    uint32_t pnp_success_count{0};             ///< PnP 成功数量。
    uint32_t discarded_count{0};               ///< 后处理丢弃候选总数。
    uint32_t semantic_discard_count{0};        ///< 语义过滤丢弃数量。
    uint32_t type_discard_count{0};            ///< 类型一致性过滤丢弃数量。
    double max_objectness{0.0};                ///< 本帧最大网络置信度。
  };

  /**
   * @brief 处理已经转换为 BGR Mat 的同步帧。
   * @param img_msg BGR 图像。
   * @param synced_frame 原始同步帧，用于保留 source_frame 指针。
   */
  void ProcessImage(const cv::Mat& img_msg, const SyncedFrame& synced_frame);

  /**
   * @brief 将 CameraFrameSync 输出帧转换成 OpenCV BGR 图像并进入处理链路。
   * @param frame 同步帧。
   */
  void ProcessSyncedFrame(const SyncedFrame& frame);

  /**
   * @brief 后台同步帧 worker 入口。
   * @param self detector 实例指针。
   */
  static void SyncFrameThreadFun(ArmorDetector<CameraInfoV>* self);

  /**
   * @brief 对单帧 BGR 图像执行网络检测和后处理。
   * @param bgr_img 输入 BGR 图像。
   * @return 本帧有效装甲板候选。
   */
  std::vector<CandidateArmor> Detect(const cv::Mat& bgr_img);

  /**
   * @brief 构建网络输入图像并记录输入坐标到源图像坐标的映射。
   * @param detector_img detector 处理区域图像。
   * @param mapping 输出坐标映射。
   * @return 网络输入尺寸的 BGR8 图像。
   */
  cv::Mat BuildNetworkInput(const cv::Mat& detector_img,
                            detail::NetworkInputMapping& mapping) const;

  /**
   * @brief 解码网络输出并执行交叠抑制、语义过滤和类型判定。
   * @param mapping 网络输入到源图像的坐标映射。
   * @param output 网络输出矩阵。
   * @param bgr_img detector 源图像。
   * @return 有效装甲板候选。
   */
  std::vector<CandidateArmor> DecodeOutput(
      const detail::NetworkInputMapping& mapping, const cv::Mat& output,
      const cv::Mat& bgr_img);

  /**
   * @brief 对已解码候选执行 dense-grid 源语义的交叠抑制。
   * @param detections 已通过 decoder 门限的候选。
   * @return 按置信度排序并去除任意 bbox 交叠后的候选下标。
   */
  std::vector<int> SelectDetectionsAfterOverlapSuppression(
      const std::vector<NetworkDetection>& detections) const;

  /**
   * @brief 解码 dense-grid keypoint 模型的一行输出。
   * @param mapping 网络输入到源图像的坐标映射。
   * @param output 网络输出矩阵。
   * @param row 待解码行号。
   * @return 通过置信度和四边形检查时返回检测单元。
   */
  std::optional<NetworkDetection> DecodeDirectKeypointDetection(
      const detail::NetworkInputMapping& mapping,
      const detail::DirectKeypointOutputView& output, int row) const;

  /**
   * @brief 将网络检测单元转换为内部候选并计算基础几何指标。
   * @param detection 网络检测单元。
   * @param bgr_img detector 源图像。
   * @return 内部装甲板候选。
   */
  CandidateArmor BuildCandidateArmor(const NetworkDetection& detection,
                                     const cv::Mat& bgr_img) const;

  /**
   * @brief 检查候选编号先验与尺寸类型是否冲突。
   * @param armor 待检查候选。
   * @return 无冲突时返回 true。
   */
  bool ValidateArmorType(const CandidateArmor& armor) const;

  /**
   * @brief 更新候选的宽高比例等几何派生量。
   * @param armor 待更新候选。
   */
  void UpdateGeometryMetrics(CandidateArmor& armor) const;

  /**
   * @brief 在编号先验不足时根据几何比例推断装甲板尺寸类型。
   * @param armor 待推断候选。
   * @return 推断出的尺寸类型。
   */
  ArmorType InferArmorType(const CandidateArmor& armor) const;

  /**
   * @brief 计算图像宽高归一化中心。
   * @param bgr_img 源图像。
   * @param center 像素中心点。
   * @return x/width、y/height 归一化中心。
   */
  cv::Point2f GetNormalizedCenter(const cv::Mat& bgr_img,
                                  const cv::Point2f& center) const;

  /**
   * @brief 将内部候选转换成 Topic 结果包并执行 PnP。
   * @param armors 内部候选列表。
   * @param bgr_img 源图像。
   */
  void FillResultMessage(const std::vector<CandidateArmor>& armors,
                         const cv::Mat& bgr_img);

 private:
  Config cfg_{};                         ///< 当前 detector 配置。
  Sync& sync_;                           ///< 同步帧来源引用。
  PnPSolver<CameraInfoV> pnp_solver_{};  ///< 装甲板 PnP 求解器。
  uint64_t latest_timestamp_us_{0};      ///< 最近处理图像的传感器时间戳，单位 us。
  uint64_t frame_index_{0};              ///< 已处理帧计数。
  LibXR::Thread sync_frame_thread_{};    ///< 后台同步帧消费线程。
  FrameCounters counters_{};             ///< 当前帧内部计数器。
  detail::OpenVinoArmorNetwork network_{}; ///< OpenVINO 网络封装。

  ArmorDetectionsPacket armors_packet_{}; ///< 复用的检测结果包。
  DetectionPacket armors_frame_packet_{}; ///< 复用的带源帧引用结果包。
  ArmorDetectorMetrics metrics_msg_{};    ///< 复用的 metrics 消息。

  /**
   * @brief detector Topic domain。
   */
  LibXR::Topic::Domain armor_domain_ = LibXR::Topic::Domain("armor_detector");

  /**
   * @brief 只包含检测结果指针的结果 Topic。
   */
  LibXR::Topic armors_topic_ =
      LibXR::Topic("armors_result", sizeof(ArmorDetectionsMessage), &armor_domain_);

  /**
   * @brief 包含检测结果和源同步帧指针的结果 Topic。
   */
  LibXR::Topic armors_frame_topic_ =
      LibXR::Topic("armors_frame", sizeof(DetectionMessage), &armor_domain_);

  /**
   * @brief detector 运行指标 Topic。
   */
  LibXR::Topic metrics_topic_ =
      LibXR::Topic("metrics", sizeof(ArmorDetectorMetrics), &armor_domain_);
};

#include "ArmorDetectorPipeline.hpp"
