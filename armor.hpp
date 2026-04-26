#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>

#include "CameraBase.hpp"
#include "transform.hpp"

enum class ArmorColor : uint8_t
{
  RED = 0,
  BLUE = 1,
  EXTINGUISH = 2,
  PURPLE = 3,
  UNKNOWN = 4,
};

enum class ArmorType : uint8_t
{
  SMALL = 0,
  LARGE = 1,
  INVALID = 2,
};

enum class ArmorNumber : uint8_t
{
  ONE = 0,
  TWO = 1,
  THREE = 2,
  FOUR = 3,
  FIVE = 4,
  OUTPOST = 5,
  GUARD = 6,
  BASE = 7,
  NEGATIVE = 8,
  UNKNOWN = NEGATIVE,
  INVALID = NEGATIVE,
};

enum class ArmorPriority : uint8_t
{
  FIRST = 1,
  SECOND = 2,
  THIRD = 3,
  FOURTH = 4,
  FORTH = FOURTH,
  FIFTH = 5,
};

inline constexpr std::array<std::string_view, 5> ARMOR_COLOR_NAMES = {
    "red", "blue", "extinguish", "purple", "unknown"};

inline constexpr std::array<std::string_view, 3> ARMOR_TYPE_NAMES = {
    "small", "large", "invalid"};

inline constexpr std::array<std::string_view, 9> ARMOR_NUMBER_NAMES = {
    "one",      "two",  "three",    "four", "five",
    "outpost",  "guard", "base",    "negative"};

inline constexpr bool ArmorNumberIsLarge(ArmorNumber number)
{
  return number == ArmorNumber::ONE || number == ArmorNumber::BASE;
}

inline constexpr bool ArmorNumberIsSmall(ArmorNumber number)
{
  return number == ArmorNumber::TWO || number == ArmorNumber::GUARD ||
         number == ArmorNumber::OUTPOST;
}

inline constexpr bool ArmorNumberIsKnown(ArmorNumber number)
{
  return number != ArmorNumber::UNKNOWN;
}

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

struct ArmorDetectorResult
{
  ArmorColor color{ArmorColor::UNKNOWN};
  ArmorNumber number{ArmorNumber::INVALID};
  ArmorType type{ArmorType::INVALID};
  ArmorPriority priority{ArmorPriority::FIFTH};
  float confidence{0.0F};
  cv::Rect box{};
  std::array<cv::Point2f, 4> points{};
  cv::Point2f center{};
  cv::Point2f center_norm{};
  double distance_to_image_center{0.0};
  LibXR::Transform<double> pose{};
};

using ArmorDetectorResults = std::vector<ArmorDetectorResult>;

// 这一层专门描述 detector 当前处理的原始同步帧引用。
// 指针只在同进程 callback 链路里有效，不能跨帧缓存。
template <CameraTypes::CameraInfo CameraInfoV>
struct ArmorDetectionsSourceFrame
{
  using Base = CameraBase<CameraInfoV>;
  using ImageFrame = typename Base::ImageFrame;
  using ImuStamped = typename Base::ImuStamped;

  uint64_t image_timestamp_us{0};
  const ImageFrame* image_frame{nullptr};
  const ImuStamped* imu{nullptr};
};

// 这类消息只用于同进程 callback 链路：
// detector 在发布时把当前帧的结果和原始帧引用一起交给 tracker，
// tracker 必须在回调里立刻消费，不能把这些指针跨帧保存。
template <CameraTypes::CameraInfo CameraInfoV>
struct ArmorDetectionsFrameMessage
{
  ArmorDetectionsSourceFrame<CameraInfoV> source_frame{};
  ArmorDetectorResults results{};
};

struct ArmorDetectionsMessage
{
  uint64_t image_timestamp_us{0};
  ArmorDetectorResults results{};
};

struct ArmorDetectorMetrics
{
  uint64_t frame_index{0};
  uint64_t image_timestamp_us{0};
  uint32_t decoded_count{0};
  uint32_t nms_count{0};
  uint32_t semantic_kept_count{0};
  uint32_t armor_count{0};
  uint32_t pnp_success_count{0};
  uint32_t refined_count{0};
  uint32_t discarded_count{0};
  uint32_t semantic_discard_count{0};
  uint32_t type_discard_count{0};
  double max_objectness{0.0};
  double detector_latency_ms{0.0};
  double publish_latency_ms{0.0};
};
