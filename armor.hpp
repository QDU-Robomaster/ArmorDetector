#pragma once

/**
 * @file armor.hpp
 * @brief ArmorDetector 对外发布的数据结构、枚举和 Topic payload 类型。
 */

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>

#include "CameraBase.hpp"
#include "transform.hpp"

/**
 * @brief 装甲板颜色分类。
 */
enum class ArmorColor : uint8_t
{
  RED = 0,        ///< 红色装甲板。
  BLUE = 1,       ///< 蓝色装甲板。
  EXTINGUISH = 2, ///< 熄灭或弱亮度装甲板。
  PURPLE = 3,     ///< 紫色或红蓝混合观测。
  UNKNOWN = 4,    ///< 颜色未知或不参与颜色过滤。
};

/**
 * @brief 装甲板几何尺寸类型。
 */
enum class ArmorType : uint8_t
{
  SMALL = 0,  ///< 小装甲板。
  LARGE = 1,  ///< 大装甲板。
  INVALID = 2, ///< 无效或不可判定类型。
};

/**
 * @brief 装甲板数字/目标类别。
 */
enum class ArmorNumber : uint8_t
{
  ONE = 0,              ///< 1 号目标。
  TWO = 1,              ///< 2 号目标。
  THREE = 2,            ///< 3 号目标。
  FOUR = 3,             ///< 4 号目标。
  FIVE = 4,             ///< 5 号目标。
  OUTPOST = 5,          ///< 前哨站目标。
  GUARD = 6,            ///< 哨兵目标。
  BASE = 7,             ///< 基地目标。
  NEGATIVE = 8,         ///< 负样本或未识别类别。
  UNKNOWN = NEGATIVE,   ///< 未知类别，当前与负样本等价。
  INVALID = NEGATIVE,   ///< 无效类别，当前与负样本等价。
};

/**
 * @brief 根据目标编号给上层策略提供的默认优先级。
 */
enum class ArmorPriority : uint8_t
{
  FIRST = 1,        ///< 最高优先级。
  SECOND = 2,       ///< 第二优先级。
  THIRD = 3,        ///< 第三优先级。
  FOURTH = 4,       ///< 第四优先级。
  FORTH = FOURTH,   ///< 历史拼写兼容别名。
  FIFTH = 5,        ///< 最低优先级或默认兜底。
};

/**
 * @brief ArmorColor 到日志/调试字符串的稳定映射。
 */
inline constexpr std::array<std::string_view, 5> ARMOR_COLOR_NAMES = {
    "red", "blue", "extinguish", "purple", "unknown"};

/**
 * @brief ArmorType 到日志/调试字符串的稳定映射。
 */
inline constexpr std::array<std::string_view, 3> ARMOR_TYPE_NAMES = {
    "small", "large", "invalid"};

/**
 * @brief ArmorNumber 到日志/调试字符串的稳定映射。
 */
inline constexpr std::array<std::string_view, 9> ARMOR_NUMBER_NAMES = {
    "one",      "two",  "three",    "four", "five",
    "outpost",  "guard", "base",    "negative"};

/**
 * @brief 判断编号先验是否对应大装甲板。
 * @param number 装甲板编号。
 * @return 编号只可能出现在大装甲板上时返回 true。
 */
inline constexpr bool ArmorNumberIsLarge(ArmorNumber number)
{
  return number == ArmorNumber::ONE || number == ArmorNumber::BASE;
}

/**
 * @brief 判断编号先验是否对应小装甲板。
 * @param number 装甲板编号。
 * @return 编号只可能出现在小装甲板上时返回 true。
 */
inline constexpr bool ArmorNumberIsSmall(ArmorNumber number)
{
  return number == ArmorNumber::TWO || number == ArmorNumber::GUARD ||
         number == ArmorNumber::OUTPOST;
}

/**
 * @brief 判断编号是否已经被模型判定为有效目标类别。
 * @param number 装甲板编号。
 * @return 编号不是 UNKNOWN/NEGATIVE/INVALID 时返回 true。
 */
inline constexpr bool ArmorNumberIsKnown(ArmorNumber number)
{
  return number != ArmorNumber::UNKNOWN;
}

/**
 * @brief 将装甲板编号映射到默认射击/跟踪优先级。
 * @param number 装甲板编号。
 * @return 该编号对应的默认优先级。
 */
inline ArmorPriority GetArmorPriority(ArmorNumber number)
{
  switch (number)
  {
    case ArmorNumber::THREE:
    case ArmorNumber::FOUR:
      return ArmorPriority::FIRST;
    case ArmorNumber::ONE:
      return ArmorPriority::SECOND;
    case ArmorNumber::FIVE:
    case ArmorNumber::GUARD:
      return ArmorPriority::THIRD;
    case ArmorNumber::TWO:
      return ArmorPriority::FOURTH;
    case ArmorNumber::OUTPOST:
    case ArmorNumber::BASE:
    case ArmorNumber::NEGATIVE:
    default:
      return ArmorPriority::FIFTH;
  }
}

/**
 * @brief 单个装甲板检测结果，包含 2D 几何、语义和可选 PnP 位姿。
 */
struct ArmorDetectorResult
{
  ArmorColor color{ArmorColor::UNKNOWN};       ///< 检测到的目标颜色。
  ArmorNumber number{ArmorNumber::INVALID};    ///< 检测到的目标编号。
  ArmorType type{ArmorType::INVALID};          ///< 推断出的装甲板尺寸类型。
  ArmorPriority priority{ArmorPriority::FIFTH}; ///< 基于编号的默认目标优先级。
  float confidence{0.0F};                       ///< 网络输出置信度。
  cv::Rect box{};                              ///< 像素坐标下的包围盒。
  std::array<cv::Point2f, 4> points{};         ///< 角点，顺序为左上、右上、右下、左下。
  bool raw_points_valid{false};                ///< raw_points 是否保存了有效网络角点。
  bool refined{false};                         ///< points 是否经过传统灯条角点细化。
  std::array<cv::Point2f, 4> raw_points{};     ///< 细化前的网络原始角点。
  cv::Point2f center{};                        ///< 像素坐标下的四边形中心。
  cv::Point2f center_norm{};                   ///< 图像宽高归一化后的中心。
  double distance_to_image_center{0.0};        ///< 中心到相机主点的像素距离。
  bool pnp_valid{false};                       ///< pose 是否由 PnP 成功求得。
  double pnp_reprojection_error_px{0.0};       ///< PnP 平均重投影误差，单位 px。
  LibXR::Transform<double> pose{};             ///< 相机坐标系下的装甲板位姿。
};

/**
 * @brief 单帧所有装甲板检测结果。
 */
using ArmorDetectorResults = std::vector<ArmorDetectorResult>;

/**
 * @brief detector 发布的基础结果包。
 */
struct ArmorDetectionsPacket
{
  uint64_t image_timestamp_us{0};     ///< 图像帧传感器时间戳，单位 us。
  ArmorDetectorResults results{};     ///< 本帧检测出的所有有效装甲板。
};

/**
 * @brief armors_result 的 Topic payload。
 *
 * Topic 只发布 detector 持有的结果指针；不能跨进程传输或延后解引用。
 */
using ArmorDetectionsMessage = ArmorDetectionsPacket*;

/**
 * @brief detector 当前处理的原始同步帧引用。
 *
 * 指针只在同进程 callback 链路里有效，不能跨帧缓存。
 *
 * @tparam CameraInfoV 编译期相机参数。
 */
template <CameraTypes::CameraInfo CameraInfoV>
struct ArmorDetectionsSourceFrame
{
  /// 相机基础类型。
  using Base = CameraBase<CameraInfoV>;
  /// 同步图像帧类型。
  using ImageFrame = typename Base::ImageFrame;
  /// 同步 IMU 样本类型。
  using ImuStamped = typename Base::ImuStamped;

  uint64_t image_timestamp_us{0};       ///< 图像帧传感器时间戳，单位 us。
  const ImageFrame* image_frame{nullptr}; ///< 原始图像帧，只在回调期间有效。
  const ImuStamped* imu{nullptr};       ///< 与图像对齐的 IMU 样本，只在回调期间有效。
};

/**
 * @brief 带原始同步帧引用的 detector 结果包。
 *
 * detector 在发布时把当前帧的结果和原始帧引用一起交给 tracker，
 * tracker 必须在回调里立刻消费，不能把这些指针跨帧保存。
 *
 * @tparam CameraInfoV 编译期相机参数。
 */
template <CameraTypes::CameraInfo CameraInfoV>
struct ArmorDetectionsFramePacket
{
  ArmorDetectionsSourceFrame<CameraInfoV> source_frame{}; ///< 当前检测结果对应的原始帧。
  ArmorDetectionsPacket* detections{nullptr};             ///< 当前检测结果包。
};

/**
 * @brief armors_frame 的 Topic payload。
 *
 * tracker 必须在回调内同步消费指针。
 *
 * @tparam CameraInfoV 编译期相机参数。
 */
template <CameraTypes::CameraInfo CameraInfoV>
using ArmorDetectionsFrameMessage = ArmorDetectionsFramePacket<CameraInfoV>*;

/**
 * @brief detector 单帧运行指标，用于日志、调参和回放对齐。
 */
struct ArmorDetectorMetrics
{
  uint64_t frame_index{0};              ///< detector 处理帧序号。
  uint64_t image_timestamp_us{0};       ///< 图像帧传感器时间戳，单位 us。
  uint32_t decoded_count{0};            ///< 置信度门限前后处理得到的候选数量。
  uint32_t nms_count{0};                ///< NMS 后保留候选数量。
  uint32_t semantic_kept_count{0};      ///< 颜色/编号/置信度语义过滤后保留数量。
  uint32_t armor_count{0};              ///< 最终发布的装甲板数量。
  uint32_t pnp_success_count{0};        ///< 本帧 PnP 成功数量。
  uint32_t refined_count{0};            ///< 本帧角点细化成功数量。
  uint32_t discarded_count{0};          ///< 后处理阶段丢弃的总候选数量。
  uint32_t semantic_discard_count{0};   ///< 被语义过滤丢弃的候选数量。
  uint32_t type_discard_count{0};       ///< 被尺寸类型一致性过滤丢弃的候选数量。
  double max_objectness{0.0};           ///< 本帧网络最大目标置信度。
  double detector_latency_ms{0.0};      ///< 推理与后处理耗时，单位 ms。
  double publish_latency_ms{0.0};       ///< 结果填充与 Topic 发布耗时，单位 ms。
};
