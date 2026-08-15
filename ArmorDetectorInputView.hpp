#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <opencv2/core.hpp>

namespace armor_detector_detail
{

struct RawRgbInputViewSpec
{
  uint8_t* data{nullptr};
  std::size_t bytes{0U};
  int width{0};
  int height{0};
};

inline bool IsValidRawRgbInputViewSpec(const RawRgbInputViewSpec& spec)
{
  if (spec.data == nullptr || spec.width <= 0 || spec.height <= 0)
  {
    return false;
  }

  constexpr std::size_t channels = 3U;
  const auto width = static_cast<std::size_t>(spec.width);
  const auto height = static_cast<std::size_t>(spec.height);
  if (width > std::numeric_limits<std::size_t>::max() / channels)
  {
    return false;
  }
  const std::size_t row_bytes = width * channels;
  return height <= std::numeric_limits<std::size_t>::max() / row_bytes &&
         spec.bytes == row_bytes * height;
}

inline bool MatchesRawRgbInputView(const RawRgbInputViewSpec& spec, const cv::Mat& input)
{
  if (!IsValidRawRgbInputViewSpec(spec) || input.empty() || input.type() != CV_8UC3 ||
      input.rows != spec.height || input.cols != spec.width || !input.isContinuous() ||
      input.data != spec.data)
  {
    return false;
  }

  const std::size_t row_bytes = static_cast<std::size_t>(spec.width) * 3U;
  return input.step[0] == row_bytes && input.total() * input.elemSize() == spec.bytes;
}

inline bool BindRawRgbInputView(const RawRgbInputViewSpec& spec, cv::Mat& input)
{
  input.release();
  if (!IsValidRawRgbInputViewSpec(spec))
  {
    return false;
  }

  input = cv::Mat(spec.height, spec.width, CV_8UC3, spec.data);
  return MatchesRawRgbInputView(spec, input);
}

}  // namespace armor_detector_detail
