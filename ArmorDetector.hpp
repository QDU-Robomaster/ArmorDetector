#pragma once

// clang-format off
/* === MODULE MANIFEST V2 ===
module_description: 装甲板检测模块：HailoRT / OpenVINO 模型推理与 PnP 位姿估计 / Armor detection Module with HailoRT or OpenVINO model inference and PnP pose estimation
depends:
- id: QDU-Robomaster/CameraFrameSync
  ref: same-or-dev
- id: QDU-Robomaster/VisionPreview
  ref: same-or-dev
- id: xrobot-org/DurationStatistics
  ref: same-or-dev
- id: QDU-Robomaster/CameraBase
  ref: same-or-dev
=== END MANIFEST === */
// clang-format on

/**
 * @file ArmorDetector.hpp
 * @brief 装甲板检测模块的主类声明和配置入口。
 *        Main class declaration and configuration entry of the armor detection Module.
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
 * @brief 装甲板检测模块：读取 CameraFrameSync 的同步帧，经 HailoRT 或 OpenVINO 推理、
 *        过滤和 PnP 求解，向 `armor_detector` 域发布 `armors_frame`。
 *        Armor detection Module: reads the synchronized frames of CameraFrameSync, runs
 *        HailoRT or OpenVINO inference, filtering and PnP, and publishes `armors_frame`
 *        in the `armor_detector` domain.
 *
 * @tparam FrameLayoutV 编译期图像缓冲区布局和像素编码，与 CameraFrameSync 和相机相同。
 *                      Compile-time image buffer layout and pixel encoding, identical to
 *                      those of CameraFrameSync and the camera.
 */
template <CameraTypes::FrameLayout FrameLayoutV>
class ArmorDetector
{
 public:
  /// 同步帧来源类型
  /// Synchronized frame source type
  using Sync = CameraFrameSync<FrameLayoutV>;
  /// 相机基础类型
  /// Camera base type
  using Base = typename Sync::Base;
  /// 图像帧类型
  /// Image frame type
  using ImageFrame = typename Sync::ImageFrame;
  /// IMU 样本类型
  /// IMU sample type
  using ImuStamped = typename Sync::ImuStamped;
  /// 图像与 IMU 的同步帧类型
  /// Synchronized image and IMU frame type
  using SyncedFrame = typename Sync::SyncedFrame;
  /// 同步帧 Topic 的借用载荷类型
  /// Borrowed payload type of the synchronized-frame Topic
  using SyncedFrameTopicPayload = typename Sync::SyncedFrameTopicPayload;
  /// 共享图像所有权句柄类型
  /// Shared image ownership handle type
  using SharedFrame = typename Base::SharedFrame;
  /// 一帧的检测结果类型
  /// Detection result type of one frame
  using DetectionPacket = DetectedFrame<FrameLayoutV>;
  /// `armors_frame` Topic 的借用载荷类型
  /// Borrowed payload type of the `armors_frame` Topic
  using DetectionMessage = DetectedFrameMessage<FrameLayoutV>;
  /// 编译期帧存储布局
  /// Compile-time frame storage layout
  static inline constexpr auto frame_layout = Base::frame_layout;

  /**
   * @brief 网络推理与后处理参数。
   *        Network inference and postprocessing parameters.
   */
  struct NetworkParams
  {
    /// 模型枚举，见 ArmorDetectorModel
    /// Model enum, see ArmorDetectorModel
    ArmorDetectorModel model{ArmorDetectorModel::INT16_HEAD_L};
    double min_confidence{0.1};  ///< 最终置信度门限
    ///< Final confidence threshold
    bool enable_quad_check{true};  ///< 是否检查四边形的凸性和面积
    ///< Whether to check quadrilateral convexity and area
    double min_quad_area_px{16.0};  ///< 四边形最小面积 (px^2)
    ///< Minimum quadrilateral area (px^2)
    double logit_threshold{0.619};  ///< objectness 预过滤门限
    ///< Objectness pre-filter threshold
    double nms_threshold{0.45};  ///< NMS 的 IoU 门限
    ///< IoU threshold of the NMS
    double bbox_expand{0.1};  ///< NMS 前包围盒的扩张比例
    ///< Bounding-box expansion ratio before the NMS
    int max_detections{128};  ///< NMS 后保留的候选数量上限
    ///< Maximum number of candidates kept after the NMS
  };

  /**
   * @brief 预留的 number refine 参数，运行时不读取这些字段。
   *        Reserved number-refine parameters; the fields are not read at runtime.
   */
  struct NumberRefineParams
  {
    bool enabled{false};  ///< 预留
    ///< Reserved
    double detector_min_confidence{0.0};  ///< 预留
    ///< Reserved
    double classifier_min_confidence{0.0};  ///< 预留
    ///< Reserved
    bool enforce_type_compatibility{false};  ///< 预留
    ///< Reserved
  };

  /**
   * @brief ArmorDetector 配置。
   *        ArmorDetector configuration.
   */
  struct Config
  {
    int detect_color{1};  ///< 保留的目标颜色：0 红色，1 蓝色，其他值关闭颜色过滤
    ///< Target color to keep: 0 red, 1 blue, any other value disables color filtering
    NetworkParams network{};  ///< 网络推理与后处理参数
    ///< Network inference and postprocessing parameters
    bool referee_auto_detect_color{false};  ///< 是否按裁判系统 robot_id 切换敌方颜色
    ///< Whether to switch the enemy color by the referee robot_id
    const char* referee_domain{"host"};  ///< 裁判系统摘要包所在的 Topic 域
    ///< Topic domain of the referee summary packet
    const char* referee_topic{"robot_game_ref"};  ///< 裁判系统摘要包的 Topic 名
    ///< Topic name of the referee summary packet
    VisionPreview::RuntimeParam preview{};  ///< 实时预览配置，默认关闭
    ///< Live preview configuration, off by default
    NumberRefineParams number_refine{};  ///< 预留的 number refine 参数
    ///< Reserved number-refine parameters
  };

  /**
   * @brief 返回全部取默认值的配置。
   *        Return the configuration with all default values.
   *
   * @return 默认配置。
   *         Default configuration.
   */
  static Config DefaultConfig() { return {}; }

  /**
   * @brief 构造 ArmorDetector：加载模型，启动推理线程与输出线程，并订阅同步帧。
   *        Construct ArmorDetector: load the model, start the inference and output
   *        threads, and subscribe to the synchronized frames.
   *
   * 模型或后端初始化失败时记录错误日志，流水线保持未启动。
   * A model or backend initialization failure is logged and leaves the pipeline stopped.
   *
   * @param sync 提供同步帧、IMU 和原生标定的 CameraFrameSync 实例。
   *             CameraFrameSync instance providing the synchronized frames, the IMU
   *             data and the native calibration.
   * @param cfg 检测配置。
   *            Detection configuration.
   */
  ArmorDetector(Sync& sync, Config cfg = DefaultConfig());

  /**
   * @brief 更新配置并重新加载模型。
   *        Update the configuration and reload the model.
   *
   * 流水线启动后调用时记录错误日志并返回。
   * A call after the pipeline has started logs an error and returns.
   *
   * @param cfg 新配置。
   *            New configuration.
   */
  void SetConfig(const Config& cfg);

  /**
   * @brief 判断所有已接纳的帧是否都已离开流水线。
   *        Check whether every admitted frame has left the pipeline.
   *
   * @return 工作线程空闲、队列为空且所有槽空闲时为 true。
   *         True when the workers are idle, the queues are empty and all slots are free.
   */
  [[nodiscard]] bool PipelineDrained() const noexcept;

  /**
   * @brief 输出预处理、推理线程、后处理和结果填充四个阶段的耗时统计。
   *        Log the duration statistics of the preprocess, inference-worker,
   *        postprocess and result stages.
   */
  void OnMonitor();

 private:
  /**
   * @brief 内部装甲板候选，已经过网络解码，尺寸类型判定后转为 ArmorDetectorResult。
   *        Internal armor candidate, decoded from the network output and converted to
   *        ArmorDetectorResult after the size-type classification.
   */
  struct CandidateArmor
  {
    ArmorColor color{ArmorColor::UNKNOWN};  ///< 网络判定的颜色
    ///< Color given by the network
    ArmorType type{ArmorType::INVALID};  ///< 尺寸类型，后处理阶段填充
    ///< Size type, filled in the postprocessing
    ArmorNumber number{ArmorNumber::INVALID};  ///< 网络判定的编号
    ///< Number given by the network
    float confidence{0.0F};  ///< 网络置信度
    ///< Network confidence
    cv::Rect box{};  ///< 包围盒
    ///< Bounding box
    std::array<cv::Point2f, 4> points{};  ///< 角点，顺序为左上、右上、右下、左下
    ///< Corner points, ordered top-left, top-right, bottom-right, bottom-left
    cv::Point2f center{};  ///< 像素中心
    ///< Pixel center
    double ratio{0.0};  ///< 左右灯条中心距与灯条长度的比例
    ///< Ratio of the distance between the light-bar centers to the light-bar length
  };

  /**
   * @brief 网络输出解码后的最小语义单元，尺寸类型和 PnP 结果在后续阶段给出。
   *        Smallest semantic unit decoded from the network output; the size type and
   *        the PnP result follow in later stages.
   */
  struct NetworkDetection
  {
    ArmorColor color{ArmorColor::UNKNOWN};  ///< 网络颜色类别
    ///< Network color class
    ArmorNumber number{ArmorNumber::UNKNOWN};  ///< 网络编号类别
    ///< Network number class
    float confidence{0.0F};  ///< 网络置信度
    ///< Network confidence
    cv::Rect box{};  ///< 由角点生成的包围盒
    ///< Bounding box generated from the corner points
    std::array<cv::Point2f, 4> points{};  ///< 统一顺序后的四角点
    ///< Four corner points in the common order
  };

  /**
   * @brief 单帧内部计数器。
   *        Per-frame internal counters.
   */
  struct FrameCounters
  {
    uint32_t decoded_count{0};  ///< 解码保留的候选数量
    ///< Candidates kept by the decoder
    uint32_t overlap_kept_count{0};  ///< 交叠抑制后的候选数量
    ///< Candidates after the overlap suppression
    uint32_t semantic_kept_count{0};  ///< 语义过滤后的候选数量
    ///< Candidates after the semantic filtering
    uint32_t pnp_success_count{0};  ///< PnP 成功的数量
    ///< Number of successful PnP solutions
    uint32_t discarded_count{0};  ///< 后处理丢弃的候选总数
    ///< Total candidates discarded by the postprocessing
    uint32_t semantic_discard_count{0};  ///< 语义过滤丢弃的数量
    ///< Candidates discarded by the semantic filtering
    uint32_t type_discard_count{0};  ///< 类型一致性过滤丢弃的数量
    ///< Candidates discarded by the type consistency check
    double max_objectness{0.0};  ///< 本帧最大网络置信度
    ///< Maximum network confidence of the frame
  };

  /**
   * @brief 单帧内部运行指标，用于日志和预览。
   *        Per-frame internal metrics, used for the logs and the preview.
   */
  struct FrameMetrics
  {
    uint64_t frame_index{0};  ///< 处理的帧序号
    ///< Index of the processed frame
    uint64_t frame_timestamp_us{0};  ///< MCU FRAME_TRIGGER 陀螺仪时间 (us)
    ///< Gyroscope time of the MCU FRAME_TRIGGER (us)
    uint64_t camera_timestamp_us{0};  ///< 相机设备图像时间 (us)
    ///< Camera device image time (us)
    uint32_t decoded_count{0};  ///< 解码保留的候选数量
    ///< Candidates kept by the decoder
    uint32_t overlap_kept_count{0};  ///< 交叠抑制后保留的候选数量
    ///< Candidates kept after the overlap suppression
    uint32_t semantic_kept_count{0};  ///< 语义过滤后保留的候选数量
    ///< Candidates kept after the semantic filtering
    uint32_t armor_count{0};  ///< 最终发布的装甲板数量
    ///< Number of armors finally published
    uint32_t pnp_success_count{0};  ///< 本帧 PnP 成功的数量
    ///< Number of successful PnP solutions in the frame
    uint32_t discarded_count{0};  ///< 后处理丢弃的候选总数
    ///< Total candidates discarded by the postprocessing
    uint32_t semantic_discard_count{0};  ///< 语义过滤丢弃的数量
    ///< Candidates discarded by the semantic filtering
    uint32_t type_discard_count{0};  ///< 类型一致性过滤丢弃的数量
    ///< Candidates discarded by the type consistency check
    double max_objectness{0.0};  ///< 本帧最大目标置信度
    ///< Maximum objectness confidence of the frame
    double preprocess_latency_ms{0.0};  ///< resize 与 BGR2RGB 等前处理耗时 (ms)
    ///< Preprocessing time such as resize and BGR2RGB (ms)
    double infer_latency_ms{0.0};  ///< 推理调用耗时 (ms)
    ///< Inference call time (ms)
    double postprocess_latency_ms{0.0};  ///< 解码、NMS 与语义过滤耗时 (ms)
    ///< Decoding, NMS and semantic filtering time (ms)
    double hailo_infer_latency_ms{0.0};  ///< Hailo 设备推理耗时 (ms)
    ///< Hailo device inference time (ms)
    double hailo_tail_latency_ms{0.0};  ///< Hailo 输出融合耗时 (ms)
    ///< Hailo output fusion time (ms)
    double detector_latency_ms{0.0};  ///< 网络检测和候选过滤耗时 (ms)
    ///< Network detection and candidate filtering time (ms)
    double result_latency_ms{0.0};  ///< PnP 和结果填充耗时 (ms)
    ///< PnP and result filling time (ms)
  };

  /**
   * @brief 处理已转换为 BGR Mat 的同步帧并发布结果。
   *        Process a synchronized frame converted to a BGR Mat and publish the result.
   *
   * @param img_msg BGR 图像。
   *                BGR image.
   * @param synced_frame 同步帧，提供共享图像所有权和同步 IMU。
   *                     Synchronized frame providing the shared image ownership and the
   *                     synchronized IMU.
   * @param armors 本帧的装甲板候选。
   *               Armor candidates of the frame.
   * @param frame_timestamp_us MCU FRAME_TRIGGER 陀螺仪时间 (us)。
   *                           Gyroscope time of the MCU FRAME_TRIGGER (us).
   * @param camera_timestamp_us 相机设备图像时间 (us)。
   *                            Camera device image time (us).
   * @param infer_timing Hailo 推理耗时快照。
   *                     Hailo inference timing snapshot.
   * @param decode_timing Hailo 输出融合耗时快照。
   *                      Hailo output fusion timing snapshot.
   * @param preprocess_latency_ms 前处理耗时 (ms)。
   *                              Preprocessing time (ms).
   * @param postprocess_latency_ms 后处理耗时 (ms)。
   *                               Postprocessing time (ms).
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
   *        Run the network detection and the postprocessing on one BGR image.
   *
   * @param bgr_img 输入 BGR 图像。
   *                Input BGR image.
   * @return 本帧有效的装甲板候选。
   *         Valid armor candidates of the frame.
   */
  std::vector<CandidateArmor> Detect(const cv::Mat& bgr_img);

  std::vector<CandidateArmor> DecodePipelineOutput(
      const cv::Mat& raw_img, const detail::NetworkInputMapping& input_mapping,
      const detail::ArmorDetectorNetwork::RawOutputSlot& raw_output,
      cv::Mat& decoded_output,
      detail::ArmorDetectorNetwork::HailoDecodeTimingSnapshot& decode_timing,
      double& postprocess_latency_ms);

  /**
   * @brief 构建网络输入图像，并记录网络张量到源图像坐标的映射。
   *        Build the network input image and record the mapping from the network tensor
   *        to the source image coordinates.
   *
   * @param bgr_img 原始 BGR 图像。
   *                Original BGR image.
   * @param mapping 输出的坐标映射。
   *                Output coordinate mapping.
   * @return 与网络张量同宽高的 RGB8 图像。
   *         RGB8 image with the width and height of the network tensor.
   */
  cv::Mat BuildNetworkInput(const cv::Mat& bgr_img,
                            detail::NetworkInputMapping& mapping) const;

  bool BuildNetworkInput(const cv::Mat& bgr_img, detail::NetworkInputMapping& mapping,
                         cv::Mat& resized_bgr, cv::Mat& rgb_input) const;

  /**
   * @brief 环境变量启用时导出一次网络输出矩阵。
   *        Export the network output matrix once when enabled by the environment
   *        variable.
   *
   * @param output 网络输出矩阵。
   *               Network output matrix.
   */
  void MaybeDumpModelOutput(const cv::Mat& output);

  /**
   * @brief 解码网络输出并执行 NMS、语义过滤和类型判定。
   *        Decode the network output and run the NMS, the semantic filtering and the
   *        type classification.
   *
   * @param raw_img 源图像。
   *                Source image.
   * @param mapping 网络输入到源图像的坐标映射。
   *                Coordinate mapping from the network input to the source image.
   * @param output 网络输出矩阵。
   *               Network output matrix.
   * @return 有效的装甲板候选。
   *         Valid armor candidates.
   */
  std::vector<CandidateArmor> DecodeOutput(const cv::Mat& raw_img,
                                           const detail::NetworkInputMapping& mapping,
                                           const cv::Mat& output);

  /**
   * @brief 对已解码的网络候选执行 NMS、语义过滤和类型判定。
   *        Run the NMS, the semantic filtering and the type classification on the
   *        decoded network candidates.
   *
   * @param raw_img 当前源图像。
   *                Current source image.
   * @param detections 已通过网络基础门限和角点检查的候选。
   *                   Candidates that passed the basic network threshold and the corner
   *                   check.
   * @return 有效的装甲板候选。
   *         Valid armor candidates.
   */
  std::vector<CandidateArmor> FinalizeDetections(
      const cv::Mat& raw_img, std::vector<NetworkDetection>&& detections);
  void SuppressNearDuplicateDetections(const std::vector<NetworkDetection>& detections,
                                       std::vector<int>& indices) const;

  /**
   * @brief 对网络候选执行 OpenCV NMS。
   *        Run the OpenCV NMS on the network candidates.
   *
   * @param detections 已通过解码门限的候选。
   *                   Candidates that passed the decoding threshold.
   * @return NMS 后保留的候选下标。
   *         Indices of the candidates kept by the NMS.
   */
  std::vector<int> SelectDetectionsAfterOpenCvNms(
      const std::vector<NetworkDetection>& detections) const;

  /**
   * @brief 解码当前模型输出的一行。
   *        Decode one row of the output of the current model.
   *
   * @param mapping 网络输入到源图像的坐标映射。
   *                Coordinate mapping from the network input to the source image.
   * @param output 网络输出矩阵视图。
   *               Network output matrix view.
   * @param row 待解码的行号。
   *            Row to decode.
   * @return 通过门限和四边形检查时返回检测单元。
   *         The detection unit when the threshold and the quadrilateral check pass.
   */
  std::optional<NetworkDetection> DecodeModelDetection(
      const detail::NetworkInputMapping& mapping, const detail::ModelOutputView& output,
      int row) const;

  /**
   * @brief 由当前模型一行的字段直接解码候选。
   *        Decode a candidate directly from the fields of one row of the current model.
   *
   * @param mapping 网络输入到源图像的坐标映射。
   *                Coordinate mapping from the network input to the source image.
   * @param read 字段读取器，接受字段索引并返回 float。
   *             Field reader taking a field index and returning a float.
   * @param field_count 当前行的字段数。
   *                    Number of fields in the row.
   * @param row 当前行号。
   *            Row index.
   * @return 通过门限和四边形检查时返回检测单元。
   *         The detection unit when the threshold and the quadrilateral check pass.
   */
  template <typename FieldReader>
  std::optional<NetworkDetection> DecodeModelDetectionFromFields(
      const detail::NetworkInputMapping& mapping, FieldReader&& read, int field_count,
      int row) const;

  /**
   * @brief 将网络检测单元转换为内部候选并计算基础几何量。
   *        Convert a network detection unit to an internal candidate and compute the
   *        basic geometry.
   *
   * @param detection 网络检测单元。
   *                  Network detection unit.
   * @return 内部装甲板候选。
   *         Internal armor candidate.
   */
  CandidateArmor BuildCandidateArmor(const NetworkDetection& detection) const;

  /**
   * @brief 根据编号先验或几何比例更新装甲板尺寸类型。
   *        Update the armor size type from the number prior or the geometric ratio.
   *
   * @param armor 待更新的候选。
   *              Candidate to update.
   */
  void ApplyNumberTypePrior(CandidateArmor& armor) const;

  /**
   * @brief 由几何比例给出明确的大小提示。
   *        Give a definite size hint from the geometric ratio.
   *
   * @param armor 待判断的候选。
   *              Candidate to judge.
   * @return ratio 明确时返回 LARGE 或 SMALL，处于灰区时返回 INVALID。
   *         LARGE or SMALL when the ratio is definite, INVALID when it is in the gray
   *         zone.
   */
  ArmorType GeometryTypeHint(const CandidateArmor& armor) const;

  /**
   * @brief 检查候选的编号先验与尺寸类型是否冲突。
   *        Check whether the number prior and the size type of a candidate conflict.
   *
   * @param armor 待检查的候选。
   *              Candidate to check.
   * @return 无冲突时返回 true。
   *         True when there is no conflict.
   */
  bool ValidateArmorType(const CandidateArmor& armor) const;

  /**
   * @brief 更新候选的宽高比例等几何派生量。
   *        Update the derived geometry of a candidate such as the aspect ratio.
   *
   * @param armor 待更新的候选。
   *              Candidate to update.
   */
  void UpdateGeometryMetrics(CandidateArmor& armor) const;

  /**
   * @brief 编号先验不足时，根据几何比例推断装甲板尺寸类型。
   *        Infer the armor size type from the geometric ratio when the number prior is
   *        insufficient.
   *
   * @param armor 待推断的候选。
   *              Candidate to infer.
   * @return 推断出的尺寸类型。
   *         Inferred size type.
   */
  ArmorType InferArmorType(const CandidateArmor& armor) const;

  /**
   * @brief 计算按图像宽高归一化的中心。
   *        Compute the center normalized by the image width and height.
   *
   * @param bgr_img 源图像。
   *                Source image.
   * @param center 像素中心点。
   *               Pixel center point.
   * @return x/width 与 y/height 构成的归一化中心。
   *         Normalized center formed by x/width and y/height.
   */
  cv::Point2f GetNormalizedCenter(const cv::Mat& bgr_img,
                                  const cv::Point2f& center) const;

  /**
   * @brief 裁判系统摘要包回调：读取首字节 robot_id，按阵营更新动态目标颜色。
   *        Referee summary packet callback: read the leading robot_id byte and update
   *        the dynamic target color by the side.
   *
   * @param data 裁判系统摘要包原始数据。
   *             Raw data of the referee summary packet.
   */
  void OnRefereeRobotGame(const LibXR::ConstRawData& data);

  /**
   * @brief 根据配置和动态裁判系统标志计算目标颜色。
   *        Compute the target color from the configuration and the dynamic referee flag.
   *
   * @return 目标颜色；UNKNOWN 表示关闭颜色过滤。
   *         Target color; UNKNOWN disables color filtering.
   */
  ArmorColor CurrentTargetColor() const;

  /**
   * @brief 由机器人 ID 推导敌方颜色标志。
   *        Derive the enemy color flag from the robot ID.
   *
   * @param robot_id 本机机器人 ID。
   *                 Robot ID of this robot.
   * @return 0 红，1 蓝，-1 无法判定。
   *         0 red, 1 blue, -1 undetermined.
   */
  static int TargetColorFromRobotId(uint8_t robot_id);

  /**
   * @brief 将内部候选转换为 Topic 结果并执行 PnP。
   *        Convert the internal candidates to Topic results and solve PnP.
   *
   * @param armors 内部候选列表。
   *               Internal candidate list.
   * @param bgr_img 源图像。
   *                Source image.
   * @param geometry 当前帧到原生传感器坐标的几何映射。
   *                 Geometry mapping from the current frame to the native sensor
   *                 coordinates.
   * @param results 输出的检测结果。
   *                Output detection results.
   */
  void FillResultMessage(const std::vector<CandidateArmor>& armors,
                         const cv::Mat& bgr_img,
                         const CameraTypes::FrameGeometry& geometry,
                         ArmorDetectorResults& results);

  /**
   * @brief 环境变量启用时导出最终结果 TSV。
   *        Export the final results as TSV when enabled by the environment variable.
   *
   * @param armors 当前帧的内部候选。
   *               Internal candidates of the current frame.
   */
  void MaybeDumpResultsTsv(const std::vector<CandidateArmor>& armors);

  /**
   * @brief 把当前帧交给预览线程绘制检测结果。
   *        Hand the current frame to the preview thread to draw the detection results.
   *
   * @param bgr_img 当前 BGR 图像，提交时立即深拷贝。
   *                Current BGR image, deep-copied on submission.
   * @param armors 当前帧的内部候选。
   *               Internal candidates of the current frame.
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

  Config cfg_{};                           ///< 当前配置 Current configuration
  Sync& sync_;                             ///< 同步帧来源 Synchronized frame source
  VisionPreview preview_{};                ///< 实时预览 Live preview
  ArmorDetectorPnPSolver pnp_solver_;      ///< 原生标定下的 PnP 求解器 PnP solver
  uint64_t latest_frame_timestamp_us_{0};  ///< 最近帧的 MCU 触发时间 (us)
  ///< MCU trigger time of the latest frame (us)
  uint64_t latest_camera_timestamp_us_{0};  ///< 最近图像的相机设备时间 (us)
  ///< Camera device time of the latest image (us)
  uint64_t frame_index_{0};  ///< 已处理的帧数 Number of processed frames
  std::thread inference_thread_{};
  std::thread output_fusion_thread_{};
  FrameCounters counters_{};  ///< 当前帧的内部计数器 Internal counters of the frame
  detail::ArmorDetectorNetwork network_{};  ///< 推理后端 Inference backend
  double last_preprocess_latency_ms_{0.0};  ///< 最近一帧的前处理耗时 (ms)
  ///< Preprocessing time of the latest frame (ms)
  double last_infer_latency_ms_{0.0};  ///< 最近一帧的推理耗时 (ms)
  ///< Inference time of the latest frame (ms)
  double last_postprocess_latency_ms_{0.0};  ///< 最近一帧的后处理耗时 (ms)
  ///< Postprocessing time of the latest frame (ms)

  DetectionPacket
      detected_frame_{};        ///< 输出线程复用的结果 Result reused by the output thread
  FrameMetrics metrics_msg_{};  ///< 复用的内部运行指标 Reused internal metrics
  std::atomic<int> referee_target_color_{-1};  ///< 裁判系统目标颜色，-1 为未设置
  ///< Referee target color, -1 when unset
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
   * @brief 发布 armors_frame 的 Topic 域 `armor_detector`。
   *        Topic domain `armor_detector` that publishes armors_frame.
   */
  LibXR::Topic::Domain armor_domain_ = LibXR::Topic::Domain("armor_detector");

  /**
   * @brief 携带共享图像所有权和检测结果的进程内 Topic。
   *        In-process Topic carrying the shared image ownership and the detection
   *        results.
   */
  LibXR::Topic armors_frame_topic_ =
      LibXR::Topic::CreateTopic<DetectionMessage>("armors_frame", &armor_domain_);

  /**
   * @brief 裁判系统摘要包所在的 Topic 域。
   *        Topic domain of the referee summary packet.
   */
  LibXR::Topic::Domain referee_domain_ = LibXR::Topic::Domain("host");

  /**
   * @brief 裁判系统摘要包 Topic。
   *        Referee summary packet Topic.
   */
  LibXR::Topic referee_topic_ = LibXR::Topic();

  /**
   * @brief 裁判系统摘要包的回调句柄。
   *        Callback handle of the referee summary packet.
   */
  LibXR::Topic::Callback referee_callback_ = LibXR::Topic::Callback();
};

#include "ArmorDetectorGeometry.hpp"
#include "ArmorDetectorInference.hpp"
#include "ArmorDetectorPublish.hpp"
#include "ArmorDetectorRuntime.hpp"
