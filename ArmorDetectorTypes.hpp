#pragma once

/**
 * @file ArmorDetectorTypes.hpp
 * @brief ArmorDetector 对外发布的数据结构、枚举和 Topic 数据类型。
 *        Data structures, enums and Topic data types published by ArmorDetector.
 */

#include <array>
#include <cstdint>
#include <opencv2/core.hpp>
#include <string_view>
#include <vector>

#include "CameraBase.hpp"
#include "transform.hpp"

/**
 * @brief 装甲板颜色分类。
 *        Armor color classes.
 */
enum class ArmorColor : uint8_t
{
  RED = 0,         ///< 红色装甲板 Red armor
  BLUE = 1,        ///< 蓝色装甲板 Blue armor
  EXTINGUISH = 2,  ///< 熄灭或弱亮度装甲板 Extinguished or dim armor
  PURPLE = 3,      ///< 紫色或红蓝混合观测 Purple or mixed red-blue observation
  UNKNOWN = 4,     ///< 颜色未知或不参与颜色过滤 Unknown color or not color-filtered
};

/**
 * @brief 装甲板几何尺寸类型。
 *        Armor geometric size types.
 */
enum class ArmorType : uint8_t
{
  SMALL = 0,    ///< 小装甲板 Small armor
  LARGE = 1,    ///< 大装甲板 Large armor
  INVALID = 2,  ///< 无效或无法判定的类型 Invalid or undeterminable type
};

/**
 * @brief 装甲板编号与目标类别。
 *        Armor numbers and target classes.
 */
enum class ArmorNumber : uint8_t
{
  ONE = 0,             ///< 1 号目标 Target number 1
  TWO = 1,             ///< 2 号目标 Target number 2
  THREE = 2,           ///< 3 号目标 Target number 3
  FOUR = 3,            ///< 4 号目标 Target number 4
  FIVE = 4,            ///< 5 号目标 Target number 5
  OUTPOST = 5,         ///< 前哨站目标 Outpost target
  GUARD = 6,           ///< 哨兵目标 Sentry target
  BASE = 7,            ///< 基地目标 Base target
  NEGATIVE = 8,        ///< 负样本或未识别的类别 Negative sample or unrecognized class
  UNKNOWN = NEGATIVE,  ///< NEGATIVE 的别名，表示未知类别 Alias of NEGATIVE, unknown class
  INVALID = NEGATIVE,  ///< NEGATIVE 的别名，表示无效类别 Alias of NEGATIVE, invalid class
};

/**
 * @brief 按目标编号给出的默认优先级，供上层策略使用。
 *        Default priority by target number, for use by upper-level strategies.
 */
enum class ArmorPriority : uint8_t
{
  FIRST = 1,       ///< 最高优先级 Highest priority
  SECOND = 2,      ///< 第二优先级 Second priority
  THIRD = 3,       ///< 第三优先级 Third priority
  FOURTH = 4,      ///< 第四优先级 Fourth priority
  FORTH = FOURTH,  ///< FOURTH 的别名 Alias of FOURTH
  FIFTH = 5,       ///< 最低优先级，也是默认值 Lowest priority, also the default
};

/**
 * @brief ArmorColor 到日志与调试字符串的映射。
 *        Mapping from ArmorColor to log and debug strings.
 */
inline constexpr std::array<std::string_view, 5> ARMOR_COLOR_NAMES = {
    "red", "blue", "extinguish", "purple", "unknown"};

/**
 * @brief ArmorType 到日志与调试字符串的映射。
 *        Mapping from ArmorType to log and debug strings.
 */
inline constexpr std::array<std::string_view, 3> ARMOR_TYPE_NAMES = {"small", "large",
                                                                     "invalid"};

/**
 * @brief ArmorNumber 到日志与调试字符串的映射。
 *        Mapping from ArmorNumber to log and debug strings.
 */
inline constexpr std::array<std::string_view, 9> ARMOR_NUMBER_NAMES = {
    "one", "two", "three", "four", "five", "outpost", "guard", "base", "negative"};

/**
 * @brief 判断编号先验是否对应大装甲板。
 *        Check whether the number prior corresponds to a large armor.
 *
 * @param number 装甲板编号。
 *               Armor number.
 * @return 编号只出现在大装甲板上时返回 true。
 *         True when the number appears on large armors only.
 */
inline constexpr bool ArmorNumberIsLarge(ArmorNumber number)
{
  return number == ArmorNumber::ONE || number == ArmorNumber::BASE;
}

/**
 * @brief 判断编号先验是否对应小装甲板。
 *        Check whether the number prior corresponds to a small armor.
 *
 * @param number 装甲板编号。
 *               Armor number.
 * @return 编号只出现在小装甲板上时返回 true。
 *         True when the number appears on small armors only.
 */
inline constexpr bool ArmorNumberIsSmall(ArmorNumber number)
{
  return number == ArmorNumber::TWO || number == ArmorNumber::THREE ||
         number == ArmorNumber::FOUR || number == ArmorNumber::FIVE ||
         number == ArmorNumber::OUTPOST || number == ArmorNumber::GUARD;
}

/**
 * @brief 判断编号是否为模型给出的有效目标类别。
 *        Check whether the number is a valid target class given by the model.
 *
 * @param number 装甲板编号。
 *               Armor number.
 * @return 编号不是 UNKNOWN、NEGATIVE 或 INVALID 时返回 true。
 *         True when the number is not UNKNOWN, NEGATIVE or INVALID.
 */
inline constexpr bool ArmorNumberIsKnown(ArmorNumber number)
{
  return number != ArmorNumber::UNKNOWN;
}

/**
 * @brief 将装甲板编号映射到默认的射击与跟踪优先级。
 *        Map an armor number to the default shooting and tracking priority.
 *
 * @param number 装甲板编号。
 *               Armor number.
 * @return 该编号对应的默认优先级。
 *         Default priority of the number.
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
 * @brief 单个装甲板检测结果，包含二维几何、语义和可选的 PnP 位姿。
 *        Detection result of one armor, with 2D geometry, semantics and an optional PnP
 *        pose.
 */
struct ArmorDetectorResult
{
  ArmorColor color{ArmorColor::UNKNOWN};  ///< 检测到的目标颜色
  ///< Detected target color
  ArmorNumber number{ArmorNumber::INVALID};  ///< 检测到的目标编号
  ///< Detected target number
  ArmorType type{ArmorType::INVALID};  ///< 推断出的装甲板尺寸类型
  ///< Inferred armor size type
  ArmorPriority priority{ArmorPriority::FIFTH};  ///< 按编号给出的默认优先级
  ///< Default priority by number
  float confidence{0.0F};  ///< 网络输出置信度
  ///< Network output confidence
  cv::Rect box{};  ///< 原生传感器像素坐标下的包围盒
  ///< Bounding box in native sensor pixel coordinates
  std::array<cv::Point2f, 4> points{};  ///< 原生角点，顺序为左上、右上、右下、左下
  ///< Native corner points, ordered top-left, top-right, bottom-right, bottom-left
  cv::Point2f center{};  ///< 原生传感器像素坐标下的中心
  ///< Center in native sensor pixel coordinates
  cv::Point2f center_norm{};  ///< 按当前帧宽高归一化的中心
  ///< Center normalized by the current-frame width and height
  double distance_to_image_center{0.0};  ///< 中心到相机主点的距离 (px)
  ///< Distance from the center to the camera principal point (px)
  bool pnp_valid{false};  ///< pose 是否由 PnP 成功求得
  ///< Whether pose was solved successfully by PnP
  double pnp_reprojection_error_px{0.0};  ///< PnP 平均重投影误差 (px)
  ///< Mean PnP reprojection error (px)
  LibXR::Transform<double> pose{};  ///< OpenCV 相机坐标系下的装甲板位姿
  ///< Armor pose in the OpenCV camera frame
};

/**
 * @brief 单帧所有装甲板的检测结果。
 *        Detection results of all armors in one frame.
 */
using ArmorDetectorResults = std::vector<ArmorDetectorResult>;

/**
 * @brief 检测完成一帧后发布的进程内结果。
 *        In-process result published after a frame has been detected.
 *
 * `image` 持有 CameraBase 对象池槽位。Topic 的载荷 `const DetectedFrame*` 在同步回调
 * 期间有效；异步处理的订阅者在回调返回前复制本对象，其中的 `SharedFrame` 延长图像寿命。
 * 帧几何从 `image` 指向的帧读取。
 * `image` holds a CameraBase object-pool slot. The Topic payload `const DetectedFrame*`
 * is valid during the synchronous callback; a subscriber that processes asynchronously
 * copies this object before the callback returns, and the `SharedFrame` inside extends
 * the image lifetime. The frame geometry is read from the frame `image` points to.
 *
 * @tparam FrameLayoutV 编译期帧布局。
 *                      Compile-time frame layout.
 */
template <CameraTypes::FrameLayout FrameLayoutV>
struct DetectedFrame
{
  using Base = CameraBase<FrameLayoutV>;
  using ImageFrame = typename Base::ImageFrame;
  using SharedFrame = typename Base::SharedFrame;
  using ImuStamped = typename Base::ImuStamped;

  uint64_t sequence{};  ///< CameraFrameSync 分配的帧序号
  ///< Frame sequence number assigned by CameraFrameSync
  SharedFrame image{};  ///< 结果对应的共享图像所有权
  ///< Shared image ownership of the result
  ImuStamped imu{};  ///< 与图像对齐的 IMU 样本
  ///< IMU sample aligned with the image
  ArmorDetectorResults detections{};  ///< 本帧检测出的所有有效装甲板
  ///< All valid armors detected in the frame

  /**
   * @brief 获取共享图像帧。
   *        Get the shared image frame.
   *
   * @return 图像帧指针；没有图像时为空。
   *         Image frame pointer; null when there is no image.
   */
  [[nodiscard]] const ImageFrame* GetImageFrame() const noexcept
  {
    return image.Get();
  }

  /**
   * @brief 判断是否持有有效图像。
   *        Check whether a valid image is held.
   *
   * @return 持有有效图像时为 true。
   *         True when a valid image is held.
   */
  [[nodiscard]] bool Valid() const noexcept { return image.Valid(); }
};

/**
 * @brief armors_frame Topic 在同步回调期间借用的载荷。
 *        Payload borrowed by the armors_frame Topic during the synchronous callback.
 */
template <CameraTypes::FrameLayout FrameLayoutV>
using DetectedFrameMessage = const DetectedFrame<FrameLayoutV>*;
