#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>

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
  INVALID = NEGATIVE,
};

enum class ArmorPriority : uint8_t
{
  FIRST = 1,
  SECOND = 2,
  THIRD = 3,
  FORTH = 4,
  FIFTH = 5,
};

inline constexpr std::array<std::string_view, 5> ARMOR_COLOR_NAMES = {
    "red", "blue", "extinguish", "purple", "unknown"};

inline constexpr std::array<std::string_view, 3> ARMOR_TYPE_NAMES = {
    "small", "large", "invalid"};

inline constexpr std::array<std::string_view, 9> ARMOR_NUMBER_NAMES = {
    "one",      "two",  "three",    "four", "five",
    "outpost",  "guard", "base",    "negative"};

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
      return ArmorPriority::FORTH;
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

struct ArmorDetectionsMessage
{
  uint64_t image_timestamp_us{0};
  ArmorDetectorResults results{};
};

struct ArmorDetectorMetrics
{
  uint64_t frame_index{0};
  uint64_t image_timestamp_us{0};
  uint32_t armor_count{0};
  uint32_t refined_count{0};
  uint32_t discarded_count{0};
  double detector_latency_ms{0.0};
  double publish_latency_ms{0.0};
};
