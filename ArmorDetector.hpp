#pragma once

// clang-format off
/* === MODULE MANIFEST V2 ===
module_description: 基于 HailoRT 的装甲板检测和 PnP 位姿估计
constructor_args:
  cfg:
    detect_color: 1
    network:
      model: ArmorDetectorModel::INT16_HEAD_L
      min_confidence: 0.1
      enable_quad_check: true
      min_quad_area_px: 16.0
      logit_threshold: 0.619
      nms_threshold: 0.45
      bbox_expand: 0.1
      max_detections: 128
    referee_auto_detect_color: false
    referee_domain: "host"
    referee_topic: "sentry_ref"
    preview:
      enabled: false
      preview_window_name: "armor_detector_preview"
      preview_scale: 0.5
      preview_wait_key_ms: 1
      queue_capacity: 1
      output_mode: "window"
      web_bind_address: "0.0.0.0"
      web_port: 8080
      web_stream_name: "armor_detector"
      max_fps: 30.0
  sync: '@camera_frame_sync'
template_args:
  - Layout:
      width: 1280
      height: 720
      step: 3840
      encoding: CameraTypes::Encoding::BGR8
required_hardware: []
depends:
  - qdu-future/CameraFrameSync
  - qdu-future/VisionPreview
  - xrobot-org/DurationStatistics
=== END MANIFEST === */
// clang-format on

/**
 * @file ArmorDetector.hpp
 * @brief 装甲板 detector 模块主类声明和配置入口。
 */

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#include "ArmorDetectorDetail.hpp"
#include "ArmorDetectorNetwork.hpp"
#include "ArmorDetectorPipeline.hpp"
#include "ArmorDetectorPnPSolver.hpp"
#include "ArmorDetectorPublishGeometry.hpp"
#include "ArmorDetectorTypes.hpp"
#include "CameraFrameSync.hpp"
#include "DurationStatistics.hpp"
#include "VisionPreview.hpp"
#include "app_framework.hpp"
#include "infer/ArmorDetectorModelAdapter.hpp"
#include "libxr.hpp"
#include "logger.hpp"

#ifndef ARMOR_DETECTOR_INT8_HEAD_L_HEF_PATH
#define ARMOR_DETECTOR_INT8_HEAD_L_HEF_PATH ""
#endif
#ifndef ARMOR_DETECTOR_INT8_GRID_L_HEF_PATH
#define ARMOR_DETECTOR_INT8_GRID_L_HEF_PATH ""
#endif
#ifndef ARMOR_DETECTOR_INT16_HEAD_L_HEF_PATH
#define ARMOR_DETECTOR_INT16_HEAD_L_HEF_PATH ""
#endif
#ifndef ARMOR_DETECTOR_INT16_FAST_L_HEF_PATH
#define ARMOR_DETECTOR_INT16_FAST_L_HEF_PATH ""
#endif
#ifndef ARMOR_DETECTOR_INT8_HEAD_HEF_PATH
#define ARMOR_DETECTOR_INT8_HEAD_HEF_PATH ""
#endif
#ifndef ARMOR_DETECTOR_INT8_GRID_HEF_PATH
#define ARMOR_DETECTOR_INT8_GRID_HEF_PATH ""
#endif
#ifndef ARMOR_DETECTOR_INT16_HEAD_HEF_PATH
#define ARMOR_DETECTOR_INT16_HEAD_HEF_PATH ""
#endif
#ifndef ARMOR_DETECTOR_INT16_FAST_HEF_PATH
#define ARMOR_DETECTOR_INT16_FAST_HEF_PATH ""
#endif
/**
 * @brief 装甲板检测应用模块。
 *
 * ArmorDetector 从 CameraFrameSync 读取已经对齐的图像/IMU
 * 帧，执行网络角点检测、 语义过滤、尺寸类型判断和 PnP 位姿求解，最后向
 * `armor_detector` domain 发布带原始帧引用的检测结果。
 *
 * @tparam FrameLayoutV 编译期图像缓冲区布局和像素编码。
 */
template <CameraTypes::FrameLayout FrameLayoutV>
class ArmorDetector : public LibXR::Application
{
 public:
  /// 同步帧来源类型。
  using Sync = CameraFrameSync<FrameLayoutV>;
  /// 相机基础类型。
  using Base = typename Sync::Base;
  /// 图像帧类型。
  using ImageFrame = typename Sync::ImageFrame;
  /// IMU 样本类型。
  using ImuStamped = typename Sync::ImuStamped;
  /// 图像/IMU 同步帧类型。
  using SyncedFrame = typename Sync::SyncedFrame;
  /// 同步帧普通 Topic 的借用 payload。
  using SyncedFrameTopicPayload = typename Sync::SyncedFrameTopicPayload;
  /// 共享图像所有权句柄。
  using SharedFrame = typename Base::SharedFrame;
  /// detector 阶段结果。
  using DetectionPacket = DetectedFrame<FrameLayoutV>;
  /// detector 普通 Topic 的借用 payload。
  using DetectionMessage = DetectedFrameMessage<FrameLayoutV>;
  /**
   * @brief 当前模块实例绑定的编译期帧存储布局。
   */
  static inline constexpr auto frame_layout = Base::frame_layout;

  /**
   * @brief 网络 detector 和后处理参数。
   */
  struct NetworkParams
  {
    /// 固定模型枚举：INT8_HEAD_L / INT8_GRID_L / INT16_HEAD_L / INT16_FAST_L /
    /// INT8_HEAD / INT8_GRID / INT16_HEAD / INT16_FAST。
    ArmorDetectorModel model{ArmorDetectorModel::INT16_HEAD_L};
    double min_confidence{0.1};     ///< 语义过滤后的最终置信度门限。
    bool enable_quad_check{true};   ///< 是否检查网络四点凸性和面积。
    double min_quad_area_px{16.0};  ///< 网络四边形最小面积，单位 px^2。
    /// objectness 原始 logit 门限。
    double logit_threshold{0.619};
    /// OpenCV NMS IoU 门限。
    double nms_threshold{0.45};
    /// NMS bbox 扩张比例。
    double bbox_expand{0.1};
    /// NMS 后最多保留候选数量。
    int max_detections{128};
  };

  /**
   * @brief 兼容旧配置的 refine 参数占位。
   *
   * 当前 detector 已不再执行 number refine；保留该结构仅用于兼容旧 YAML 和
   * 已生成配置，运行时会忽略这些字段。
   */
  struct NumberRefineParams
  {
    bool enabled{false};
    double detector_min_confidence{0.0};
    double classifier_min_confidence{0.0};
    bool enforce_type_compatibility{false};
  };

  /**
   * @brief ArmorDetector 模块配置。
   */
  struct Config
  {
    int detect_color{1};                      ///< 0=红色，1=蓝色，其他=不限制颜色。
    NetworkParams network{};                  ///< 网络 detector 参数。
    bool referee_auto_detect_color{false};    ///< 是否根据裁判系统动态切换敌方颜色。
    const char* referee_domain{"host"};       ///< 裁判系统所在主题域。
    const char* referee_topic{"sentry_ref"};  ///< 裁判系统摘要包主题名。
    VisionPreview::RuntimeParam preview{};    ///< 可选实时预览配置。
    NumberRefineParams number_refine{};       ///< 兼容旧配置；当前不生效。
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

  /** Return true after every admitted frame has left the detector pipeline. */
  [[nodiscard]] bool PipelineDrained() const noexcept;

  /**
   * @brief 输出 detector 本地阶段耗时统计。
   */
  void OnMonitor() override;

 private:
  /**
   * @brief detector 内部装甲板候选。
   *
   * 候选已经经过网络解码，可进一步做尺寸类型判定，最后再转为
   * ArmorDetectorResult。
   */
  struct CandidateArmor
  {
    ArmorColor color{ArmorColor::UNKNOWN};     ///< 网络判定颜色。
    ArmorType type{ArmorType::INVALID};        ///< 尺寸类型，后处理阶段填充。
    ArmorNumber number{ArmorNumber::INVALID};  ///< 网络判定编号。
    float confidence{0.0F};                    ///< 网络置信度。
    cv::Rect box{};                            ///< 当前候选包围盒。
    std::array<cv::Point2f, 4> points{};  ///< 当前角点，顺序为左上、右上、右下、左下。
    cv::Point2f center{};                 ///< 候选像素中心。
    double ratio{0.0};                    ///< 左右灯条中心距与灯条长度的比例。
  };

  /**
   * @brief 网络输出解码后的最小语义单元。
   *
   * 该结构不包含尺寸类型判定和 PnP 结果。
   */
  struct NetworkDetection
  {
    ArmorColor color{ArmorColor::UNKNOWN};     ///< 网络颜色类别。
    ArmorNumber number{ArmorNumber::UNKNOWN};  ///< 网络编号类别。
    float confidence{0.0F};                    ///< 网络置信度。
    cv::Rect box{};                            ///< 由角点生成的包围盒。
    std::array<cv::Point2f, 4> points{};       ///< 统一顺序后的四角点。
  };

  /**
   * @brief 单帧 detector 内部计数器。
   */
  struct FrameCounters
  {
    uint32_t decoded_count{0};           ///< 网络 decoder 保留候选数量。
    uint32_t overlap_kept_count{0};      ///< 交叠抑制后候选数量。
    uint32_t semantic_kept_count{0};     ///< 语义过滤后候选数量。
    uint32_t pnp_success_count{0};       ///< PnP 成功数量。
    uint32_t discarded_count{0};         ///< 后处理丢弃候选总数。
    uint32_t semantic_discard_count{0};  ///< 语义过滤丢弃数量。
    uint32_t type_discard_count{0};      ///< 类型一致性过滤丢弃数量。
    double max_objectness{0.0};          ///< 本帧最大网络置信度。
  };

  /**
   * @brief 单帧 detector 内部运行指标。
   *
   * 只用于日志和本模块预览，不作为 topic 发布。
   */
  struct FrameMetrics
  {
    uint64_t frame_index{0};             ///< detector 处理帧序号。
    uint64_t frame_timestamp_us{0};      ///< MCU FRAME_TRIGGER 陀螺仪时间，单位 us。
    uint64_t camera_timestamp_us{0};     ///< 相机设备图像时间，单位 us。
    uint32_t decoded_count{0};           ///< decoder 保留候选数量。
    uint32_t overlap_kept_count{0};      ///< 交叠抑制后保留候选数量。
    uint32_t semantic_kept_count{0};     ///< 语义过滤后保留候选数量。
    uint32_t armor_count{0};             ///< 最终发布的装甲板数量。
    uint32_t pnp_success_count{0};       ///< 本帧 PnP 成功数量。
    uint32_t discarded_count{0};         ///< 后处理丢弃候选总数。
    uint32_t semantic_discard_count{0};  ///< 语义过滤丢弃数量。
    uint32_t type_discard_count{0};      ///< 类型一致性过滤丢弃数量。
    double max_objectness{0.0};          ///< 本帧网络最大目标置信度。
    double preprocess_latency_ms{0.0};   ///< resize+BGR2RGB 等前处理耗时。
    double infer_latency_ms{0.0};        ///< network_.Infer() 总耗时。
    double postprocess_latency_ms{0.0};  ///< DecodeOutput/NMS/语义过滤耗时。
    double hailo_infer_latency_ms{0.0};  ///< Hailo 设备推理耗时，单位 ms。
    double hailo_tail_latency_ms{0.0};   ///< Hailo 输出融合耗时，单位 ms。
    double detector_latency_ms{0.0};     ///< 网络检测和候选过滤耗时，单位 ms。
    double result_latency_ms{0.0};       ///< PnP 和结果填充耗时，单位 ms。
  };

  /**
   * @brief 处理已经转换为 BGR Mat 的同步帧。
   * @param img_msg BGR 图像。
   * @param synced_frame 原始同步帧，用于复制共享图像所有权和同步 IMU。
   */
  void ProcessImage(
      const cv::Mat& img_msg, SyncedFrame& synced_frame,
      std::vector<CandidateArmor>&& armors, uint64_t frame_timestamp_us,
      uint64_t camera_timestamp_us,
      const detail::ArmorDetectorNetwork::HailoRawTimingSnapshot& infer_timing,
      const detail::ArmorDetectorNetwork::HailoDecodeTimingSnapshot& decode_timing,
      double preprocess_latency_ms, double postprocess_latency_ms);

  static void InferenceThreadFun(ArmorDetector<FrameLayoutV>* self);

  static void OutputFusionThreadFun(ArmorDetector<FrameLayoutV>* self);

  static void OnSyncedFrameStatic(bool, ArmorDetector<FrameLayoutV>* self,
                                  SyncedFrameTopicPayload borrowed);

  bool AdmitSyncedFrame(const SyncedFrame& synced_frame);

  void RunInference(armor_detector_pipeline::WorkItem item);

  void HandleInferenceCompletion(
      armor_detector_pipeline::WorkItem item, bool ok,
      detail::ArmorDetectorNetwork::HailoRawTimingSnapshot timing);

  void DrainCompletedInferencesLocked();

  void RunOutputFusion(armor_detector_pipeline::WorkItem item);

  void RunPostprocess(armor_detector_pipeline::WorkItem item);

  [[nodiscard]] bool HasFreePostSlotLocked() const;

  std::optional<armor_detector_pipeline::WorkItem> AcquirePostSlotLocked();

  bool HandoffInferToPostLocked(armor_detector_pipeline::WorkItem infer_item,
                                armor_detector_pipeline::WorkItem post_item);

  void ReleaseInferSlot(armor_detector_pipeline::WorkItem item);

  void ReleaseInferSlotLocked(armor_detector_pipeline::WorkItem item);

  void ReleasePostSlot(armor_detector_pipeline::WorkItem item);

  void ReleasePostSlotLocked(armor_detector_pipeline::WorkItem item);

  /**
   * @brief 对单帧 BGR 图像执行网络检测和后处理。
   * @param bgr_img 输入 BGR 图像。
   * @return 本帧有效装甲板候选。
   */
  std::vector<CandidateArmor> Detect(const cv::Mat& bgr_img);

  std::vector<CandidateArmor> DecodePipelineOutput(
      const cv::Mat& raw_img, const detail::NetworkInputMapping& input_mapping,
      const detail::ArmorDetectorNetwork::RawOutputSlot& raw_output,
      cv::Mat& decoded_output,
      detail::ArmorDetectorNetwork::HailoDecodeTimingSnapshot& decode_timing,
      double& postprocess_latency_ms);

  /**
   * @brief 构建网络输入图像并记录网络张量到源图像坐标的映射。
   * @param bgr_img 原始 BGR 图像。
   * @param mapping 输出坐标映射。
   * @return 网络张量宽高对应的 RGB8 图像。
   */
  cv::Mat BuildNetworkInput(const cv::Mat& bgr_img,
                            detail::NetworkInputMapping& mapping) const;

  bool BuildNetworkInput(const cv::Mat& bgr_img, detail::NetworkInputMapping& mapping,
                         cv::Mat& resized_bgr, cv::Mat& rgb_input) const;

  /**
   * @brief 在环境变量启用时一次性导出 Infer() 后的矩阵输出。
   * @param output 当前 detector 矩阵输出。
   */
  void MaybeDumpModelOutput(const cv::Mat& output);

  /**
   * @brief 解码网络输出并执行 NMS、语义过滤和类型判定。
   * @param mapping 网络输入到源图像的坐标映射。
   * @param output 网络输出矩阵。
   * @return 有效装甲板候选。
   */
  std::vector<CandidateArmor> DecodeOutput(const cv::Mat& raw_img,
                                           const detail::NetworkInputMapping& mapping,
                                           const cv::Mat& output);

  /**
   * @brief 对已经解码的网络候选执行 NMS、语义过滤和类型判定。
   * @param raw_img 当前源图像。
   * @param detections 已通过网络基础门限和角点检查的候选。
   * @return 有效装甲板候选。
   */
  std::vector<CandidateArmor> FinalizeDetections(
      const cv::Mat& raw_img, std::vector<NetworkDetection>&& detections);
  void SuppressNearDuplicateDetections(const std::vector<NetworkDetection>& detections,
                                       std::vector<int>& indices) const;

  /**
   * @brief 对网络候选执行 OpenCV NMS。
   * @param detections 已通过 decoder 门限的候选。
   * @return NMS 后候选下标。
   */
  std::vector<int> SelectDetectionsAfterOpenCvNms(
      const std::vector<NetworkDetection>& detections) const;

  /**
   * @brief 解码当前 detector 模型的一行输出。
   * @param mapping 网络输入到源图像的坐标映射。
   * @param output 网络输出矩阵。
   * @param row 待解码行号。
   * @return 通过门限和四边形检查时返回检测单元。
   */
  std::optional<NetworkDetection> DecodeModelDetection(
      const detail::NetworkInputMapping& mapping, const detail::ModelOutputView& output,
      int row) const;

  /**
   * @brief 从当前 detector family 的一行字段直接解码候选。
   * @param mapping 网络输入到源图像的坐标映射。
   * @param read 字段读取器，接受 field 索引并返回 float。
   * @param field_count 当前行字段数。
   * @return 通过门限和四边形检查时返回检测单元。
   */
  template <typename FieldReader>
  std::optional<NetworkDetection> DecodeModelDetectionFromFields(
      const detail::NetworkInputMapping& mapping, FieldReader&& read, int field_count,
      int row) const;

  /**
   * @brief 将网络检测单元转换为内部候选并计算基础几何指标。
   * @param detection 网络检测单元。
   * @return 内部装甲板候选。
   */
  CandidateArmor BuildCandidateArmor(const NetworkDetection& detection) const;

  /**
   * @brief 根据编号先验或几何比例更新装甲板尺寸类型。
   * @param armor 待更新候选。
   */
  void ApplyNumberTypePrior(CandidateArmor& armor) const;

  /**
   * @brief 几何比例给出的明确大小提示。
   * @param armor 待判断候选。
   * @return ratio 明确时返回 LARGE/SMALL，处于灰区时返回 INVALID。
   */
  ArmorType GeometryTypeHint(const CandidateArmor& armor) const;

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
   * @brief 裁判系统回调入口。
   *
   * 只读取 RobotGameReferee 包前缀中的 robot_id
   * 字节，根据阵营写入动态目标颜色标志。
   * @param data 裁判系统摘要包原始数据。
   */
  void OnRefereeRobotGame(const LibXR::ConstRawData& data);

  /**
   * @brief 根据当前配置和动态裁判系统标志计算目标颜色。
   * @return 目标颜色；UNKNOWN 表示不按颜色过滤。
   */
  ArmorColor CurrentTargetColor() const;

  /**
   * @brief 由机器人 ID 推导敌方颜色标志。
   * @param robot_id 本机机器人 ID。
   * @return 0=红，1=蓝，-1=不确定。
   */
  static int TargetColorFromRobotId(uint8_t robot_id);

  /**
   * @brief 将内部候选转换成 Topic 结果包并执行 PnP。
   * @param armors 内部候选列表。
   * @param bgr_img 源图像。
   */
  void FillResultMessage(const std::vector<CandidateArmor>& armors,
                         const cv::Mat& bgr_img,
                         const CameraTypes::FrameGeometry& geometry,
                         ArmorDetectorResults& results);

  /**
   * @brief 在环境变量启用时导出最终 detector 结果 TSV。
   * @param armors 当前帧内部候选。
   */
  void MaybeDumpResultsTsv(const std::vector<CandidateArmor>& armors);

  /**
   * @brief 把当前帧交给预览线程绘制检测结果。
   * @param bgr_img 当前 BGR 图像；Submit 内部会立即深拷贝。
   */
  void SubmitPreview(const cv::Mat& bgr_img, const std::vector<CandidateArmor>& armors);

 private:
  static constexpr std::size_t infer_inflight_slot_count = 2U;
  static constexpr std::size_t infer_prepared_slot_count = 1U;
  static constexpr std::size_t infer_slot_count =
      infer_inflight_slot_count + infer_prepared_slot_count;
  static constexpr std::size_t post_slot_count = 2U;
  static constexpr uint32_t async_inflight_limit = infer_inflight_slot_count;

  struct PipelineFrameContext
  {
    uint64_t frame_timestamp_us{0};
    uint64_t camera_timestamp_us{0};
    bool admission_counted{false};
    detail::NetworkInputMapping input_mapping{};
    SyncedFrame synced_frame{};
    detail::ArmorDetectorNetwork::HailoRawTimingSnapshot infer_timing{};
    detail::ArmorDetectorNetwork::HailoDecodeTimingSnapshot decode_timing{};
    double preprocess_latency_ms{0.0};
    double postprocess_latency_ms{0.0};
    bool async_completed{false};
    bool async_ok{false};
  };

  struct HailoBufferPair
  {
    detail::ArmorDetectorNetwork::RawOutputSlot raw_output{};
    cv::Mat network_input{};
  };

  struct InferSlot
  {
    std::atomic<armor_detector_pipeline::InferSlotState> state{
        armor_detector_pipeline::InferSlotState::FREE};
    std::atomic<uint64_t> generation{0};
    cv::Mat preprocess_scratch{};
    std::size_t hailo_buffer_id{0U};
    PipelineFrameContext context{};
  };

  struct PostSlot
  {
    std::atomic<armor_detector_pipeline::PostSlotState> state{
        armor_detector_pipeline::PostSlotState::FREE};
    std::atomic<uint64_t> generation{0};
    std::size_t hailo_buffer_id{0U};
    cv::Mat decoded_output{};
    PipelineFrameContext context{};
  };

  Config cfg_{};                            ///< 当前 detector 配置。
  Sync& sync_;                              ///< 同步帧来源引用。
  VisionPreview preview_{};                 ///< 可选实时预览。
  ArmorDetectorPnPSolver pnp_solver_;       ///< 原生标定下的装甲板 PnP 求解器。
  uint64_t latest_frame_timestamp_us_{0};   ///< 最近处理帧的 MCU 触发时间，单位 us。
  uint64_t latest_camera_timestamp_us_{0};  ///< 最近图像的相机设备时间，单位 us。
  uint64_t frame_index_{0};                 ///< 已处理帧计数。
  std::thread inference_thread_{};
  std::thread output_fusion_thread_{};
  FrameCounters counters_{};                 ///< 当前帧内部计数器。
  detail::ArmorDetectorNetwork network_{};   ///< detector 推理后端。
  double last_preprocess_latency_ms_{0.0};   ///< Detect() 最近一帧前处理耗时。
  double last_infer_latency_ms_{0.0};        ///< Detect() 最近一帧 network_.Infer 耗时。
  double last_postprocess_latency_ms_{0.0};  ///< Detect() 最近一帧后处理耗时。

  DetectionPacket detected_frame_{};           ///< 单 postprocess worker 复用的阶段结果。
  FrameMetrics metrics_msg_{};                 ///< 复用的内部运行指标。
  std::atomic<int> referee_target_color_{-1};  ///< 动态裁判系统目标颜色，-1 表示未设置。
  std::array<InferSlot, infer_slot_count> infer_slots_{};
  std::array<PostSlot, post_slot_count> post_slots_{};
  // Hailo callbacks and bindings require these buffer object addresses to stay
  // fixed.
  std::array<HailoBufferPair, infer_slot_count + post_slot_count> hailo_buffer_pool_{};
  armor_detector_pipeline::FixedSpscQueue<armor_detector_pipeline::WorkItem,
                                          infer_slot_count>
      inference_queue_{};
  armor_detector_pipeline::FixedSpscQueue<armor_detector_pipeline::WorkItem,
                                          post_slot_count>
      output_queue_{};
  armor_detector_pipeline::OrderedAsyncCompletions<infer_slot_count> async_completions_{};
  mutable std::mutex pipeline_mutex_{};
  std::condition_variable pipeline_cv_{};
  std::atomic<uint64_t> pipeline_admitted_count_{0};
  std::atomic<uint64_t> pipeline_completed_count_{0};
  std::atomic<uint64_t> pipeline_prepare_drop_count_{0};
  std::atomic<uint64_t> pipeline_no_free_count_{0};
  std::atomic<uint64_t> pipeline_infer_fail_count_{0};
  std::atomic<uint64_t> pipeline_post_fail_count_{0};
  XRobot::DurationStatistics preprocess_duration_{};
  XRobot::DurationStatistics inference_worker_duration_{};
  XRobot::DurationStatistics postprocess_duration_{};
  XRobot::DurationStatistics result_duration_{};
  std::atomic<bool> inference_worker_active_{false};
  std::atomic<bool> output_worker_active_{false};
  uint32_t async_inflight_{0};
  bool async_inference_enabled_{true};
  std::atomic<bool> workers_started_{false};
  LibXR::Topic synced_frame_topic_ = LibXR::Topic();
  LibXR::Topic::Callback synced_frame_callback_{};

  /**
   * @brief detector Topic domain，只发布 armors_frame。
   */
  LibXR::Topic::Domain armor_domain_ = LibXR::Topic::Domain("armor_detector");

  /**
   * @brief 包含共享图像所有权和检测结果的进程内 Topic。
   */
  LibXR::Topic armors_frame_topic_ =
      LibXR::Topic::CreateTopic<DetectionMessage>("armors_frame", &armor_domain_);

  /**
   * @brief 裁判系统主题域。
   */
  LibXR::Topic::Domain referee_domain_ = LibXR::Topic::Domain("host");

  /**
   * @brief 裁判系统摘要包主题。
   */
  LibXR::Topic referee_topic_ = LibXR::Topic();

  /**
   * @brief 裁判系统回调句柄。
   */
  LibXR::Topic::Callback referee_callback_ = LibXR::Topic::Callback();
};

#include "ArmorDetectorGeometry.hpp"
#include "ArmorDetectorInference.hpp"
#include "ArmorDetectorPublish.hpp"
#include "ArmorDetectorRuntime.hpp"
