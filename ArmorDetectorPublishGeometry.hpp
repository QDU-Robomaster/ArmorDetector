#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <opencv2/core.hpp>

#include "CameraBase.hpp"

namespace armor_detector_detail
{

/**
 * Maps detected light-bar end keypoints onto the PnP model end points.
 *
 * Corners are top-left, top-right, bottom-right, bottom-left; the light bars are TL-BL and
 * TR-BR. The network keypoints of a bar are modeled as the projection of two points
 * `keypoint_length_mm` apart on the bar plus a constant outward overshoot of `end_offset_px`
 * per end (glow, blur, keypoint bias; sensor px, independent of range). Each bar keeps its
 * midpoint and direction; its half length h becomes model/keypoint * (h - offset), so the
 * corners again correspond to the `model_length_mm` model. Light-bar midpoints, and therefore
 * bearing and bar separation, are unchanged. keypoint == model and offset 0 is the identity.
 */
[[nodiscard]] inline std::array<cv::Point2f, 4> CorrectLightbarKeypoints(
    const std::array<cv::Point2f, 4>& points, double model_length_mm,
    double keypoint_length_mm, double end_offset_px)
{
  if (!(keypoint_length_mm > 0.0) ||
      (keypoint_length_mm == model_length_mm && end_offset_px == 0.0))
  {
    return points;
  }
  const double scale = model_length_mm / keypoint_length_mm;
  std::array<cv::Point2f, 4> out = points;
  for (const auto& bar : {std::array<std::size_t, 2>{0U, 3U}, std::array<std::size_t, 2>{1U, 2U}})
  {
    const cv::Point2d top(points[bar[0]]);
    const cv::Point2d bottom(points[bar[1]]);
    const cv::Point2d mid = 0.5 * (top + bottom);
    const cv::Point2d half_vec = 0.5 * (bottom - top);
    const double half = std::hypot(half_vec.x, half_vec.y);
    if (!(half > 0.0))
    {
      continue;
    }
    // Never shrink a bar below half its detected length (tiny or degenerate detections).
    const double corrected = scale * std::max(half - end_offset_px, 0.5 * half);
    const cv::Point2d v = half_vec * (corrected / half);
    out[bar[0]] = cv::Point2f(static_cast<float>(mid.x - v.x), static_cast<float>(mid.y - v.y));
    out[bar[1]] = cv::Point2f(static_cast<float>(mid.x + v.x), static_cast<float>(mid.y + v.y));
  }
  return out;
}

/** Geometry exposed by ArmorDetector in native sensor coordinates. */
struct PublishGeometry
{
  std::array<cv::Point2f, 4> points{};
  cv::Point2f center{};
  cv::Rect box{};
};

/**
 * Maps frame-local detection geometry to native sensor coordinates.
 *
 * Input and output corners use top-left, top-right, bottom-right, bottom-left
 * order. Reversed axes therefore remap both coordinates and corner indices.
 */
[[nodiscard]] inline PublishGeometry MapPublishGeometry(
    const std::array<cv::Point2f, 4>& frame_points, const cv::Point2f& frame_center,
    const cv::Rect& frame_box, const CameraTypes::FrameGeometry& geometry)
{
  const auto to_native_point = [&geometry](const cv::Point2f& point)
  {
    return cv::Point2f(
        static_cast<float>(CameraTypes::FrameToNativeX(geometry, point.x)),
        static_cast<float>(CameraTypes::FrameToNativeY(geometry, point.y)));
  };

  std::array<cv::Point2f, 4> mapped_points{};
  for (std::size_t index = 0; index < frame_points.size(); ++index)
  {
    mapped_points[index] = to_native_point(frame_points[index]);
  }

  const bool reverse_x =
      CameraTypes::HasGeometryFlag(geometry, CameraTypes::FRAME_GEOMETRY_REVERSE_X);
  const bool reverse_y =
      CameraTypes::HasGeometryFlag(geometry, CameraTypes::FRAME_GEOMETRY_REVERSE_Y);
  const std::array<std::size_t, 4> native_order =
      reverse_x && reverse_y ? std::array<std::size_t, 4>{2U, 3U, 0U, 1U}
      : reverse_x            ? std::array<std::size_t, 4>{1U, 0U, 3U, 2U}
      : reverse_y            ? std::array<std::size_t, 4>{3U, 2U, 1U, 0U}
                             : std::array<std::size_t, 4>{0U, 1U, 2U, 3U};

  PublishGeometry result{};
  for (std::size_t index = 0; index < result.points.size(); ++index)
  {
    result.points[index] = mapped_points[native_order[index]];
  }
  result.center = to_native_point(frame_center);

  const double box_x =
      static_cast<double>(geometry.roi_offset_x_native) + geometry.sample_phase_x_native +
      static_cast<double>(geometry.decimation_x) *
          (reverse_x ? static_cast<double>(geometry.width) - frame_box.x - frame_box.width
                     : frame_box.x);
  const double box_y =
      static_cast<double>(geometry.roi_offset_y_native) + geometry.sample_phase_y_native +
      static_cast<double>(geometry.decimation_y) *
          (reverse_y
               ? static_cast<double>(geometry.height) - frame_box.y - frame_box.height
               : frame_box.y);
  result.box =
      cv::Rect(cvRound(box_x), cvRound(box_y),
               cvRound(static_cast<double>(geometry.decimation_x) * frame_box.width),
               cvRound(static_cast<double>(geometry.decimation_y) * frame_box.height));
  return result;
}

}  // namespace armor_detector_detail
