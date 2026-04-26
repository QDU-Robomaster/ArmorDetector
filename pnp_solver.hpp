#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "CameraBase.hpp"
#include "armor.hpp"

template <CameraTypes::CameraInfo CameraInfoV>
class PnPSolver
{
 public:
  using CameraInfo = CameraTypes::CameraInfo;

  static inline constexpr CameraInfo camera_info = CameraInfoV;
  static inline constexpr auto pnp_dist_coeffs_info =
      CameraTypes::BuildPnPDistCoeffs(camera_info);

  PnPSolver();

  [[nodiscard]] bool SolvePnP(const std::array<cv::Point2f, 4>& image_armor_points,
                              ArmorType armor_type, cv::Mat& rvec,
                              cv::Mat& tvec) const;

  [[nodiscard]] double CalculateDistanceToCenter(
      const cv::Point2f& image_point) const;

 private:
  [[nodiscard]] static double ComputeReprojectionError(
      const std::vector<cv::Point3f>& object_points,
      const std::vector<cv::Point2f>& image_points, const cv::Mat& camera_matrix,
      const cv::Mat& dist_coeffs, const cv::Mat& rvec, const cv::Mat& tvec);
  static cv::Mat BuildCameraMatrix();
  static cv::Mat BuildDistCoeffs();
  static std::vector<cv::Point3f> BuildArmorPoints(double width_mm,
                                                   double height_mm);

 private:
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  std::vector<cv::Point3f> small_armor_points_;
  std::vector<cv::Point3f> large_armor_points_;

  inline static constexpr double small_armor_width_mm = 135.0;
  inline static constexpr double small_armor_height_mm = 56.0;
  inline static constexpr double large_armor_width_mm = 225.0;
  inline static constexpr double large_armor_height_mm = 56.0;
};

template <CameraTypes::CameraInfo CameraInfoV>
PnPSolver<CameraInfoV>::PnPSolver()
    : camera_matrix_(BuildCameraMatrix()),
      dist_coeffs_(BuildDistCoeffs()),
      small_armor_points_(
          BuildArmorPoints(small_armor_width_mm, small_armor_height_mm)),
      large_armor_points_(
          BuildArmorPoints(large_armor_width_mm, large_armor_height_mm))
{
}

template <CameraTypes::CameraInfo CameraInfoV>
cv::Mat PnPSolver<CameraInfoV>::BuildCameraMatrix()
{
  return cv::Mat(3, 3, CV_64F,
                 const_cast<double*>(camera_info.camera_matrix.data()))
      .clone();
}

template <CameraTypes::CameraInfo CameraInfoV>
cv::Mat PnPSolver<CameraInfoV>::BuildDistCoeffs()
{
  if constexpr (pnp_dist_coeffs_info.uses_rational_polynomial_extension)
  {
    XR_LOG_WARN(
        "PnPSolver: using 8-term rational; extend to 14 if backend supports.");
  }

  if constexpr (pnp_dist_coeffs_info.requires_undistort_first)
  {
    XR_LOG_WARN(
        "PnPSolver: distortion model not natively supported (%d). "
        "TODO: undistort to pinhole first, then call PnP with NONE.",
        int(camera_info.distortion_model));
    return {};
  }

  if constexpr (pnp_dist_coeffs_info.size == 0)
  {
    return {};
  }

  return cv::Mat(1, static_cast<int>(pnp_dist_coeffs_info.size), CV_64F,
                 const_cast<double*>(pnp_dist_coeffs_info.values.data()))
      .clone();
}

template <CameraTypes::CameraInfo CameraInfoV>
std::vector<cv::Point3f> PnPSolver<CameraInfoV>::BuildArmorPoints(double width_mm,
                                                                  double height_mm)
{
  const double half_width_m = width_mm * 0.5 / 1000.0;
  const double half_height_m = height_mm * 0.5 / 1000.0;

  return {
      {0.0f, static_cast<float>(half_width_m), static_cast<float>(-half_height_m)},
      {0.0f, static_cast<float>(half_width_m), static_cast<float>(half_height_m)},
      {0.0f, static_cast<float>(-half_width_m), static_cast<float>(half_height_m)},
      {0.0f, static_cast<float>(-half_width_m), static_cast<float>(-half_height_m)}};
}

template <CameraTypes::CameraInfo CameraInfoV>
bool PnPSolver<CameraInfoV>::SolvePnP(
    const std::array<cv::Point2f, 4>& image_armor_points, ArmorType armor_type,
    cv::Mat& rvec, cv::Mat& tvec) const
{
  const std::array<cv::Point2f, 4> base_points = {
      image_armor_points[3], image_armor_points[0], image_armor_points[1],
      image_armor_points[2]};

  const auto& object_points =
      (armor_type == ArmorType::SMALL) ? small_armor_points_ : large_armor_points_;
  constexpr std::array<int, 2> methods = {
      cv::SOLVEPNP_ITERATIVE,
      cv::SOLVEPNP_EPNP,
  };
  bool found = false;
  double best_error = std::numeric_limits<double>::infinity();
  const std::vector<cv::Point2f> image_points = {
      base_points[0], base_points[1], base_points[2], base_points[3]};

  {
    std::vector<cv::Mat> candidate_rvecs;
    std::vector<cv::Mat> candidate_tvecs;
    if (cv::solvePnPGeneric(object_points, image_points, camera_matrix_, dist_coeffs_,
                            candidate_rvecs, candidate_tvecs, false,
                            cv::SOLVEPNP_IPPE) > 0)
    {
      for (std::size_t candidate_index = 0;
           candidate_index < candidate_rvecs.size() &&
           candidate_index < candidate_tvecs.size();
           ++candidate_index)
      {
        const auto& candidate_rvec = candidate_rvecs[candidate_index];
        const auto& candidate_tvec = candidate_tvecs[candidate_index];
        if (candidate_rvec.empty() || candidate_tvec.empty())
        {
          continue;
        }

        const double z = candidate_tvec.at<double>(2);
        if (!std::isfinite(z) || z <= 1e-6)
        {
          continue;
        }

        const double reprojection_error =
            ComputeReprojectionError(object_points, image_points, camera_matrix_,
                                     dist_coeffs_, candidate_rvec, candidate_tvec);
        if (!std::isfinite(reprojection_error))
        {
          continue;
        }
        if (!found || reprojection_error < best_error)
        {
          found = true;
          best_error = reprojection_error;
          rvec = candidate_rvec.clone();
          tvec = candidate_tvec.clone();
        }
      }
    }
  }

  for (const int method : methods)
  {
    cv::Mat candidate_rvec;
    cv::Mat candidate_tvec;
    if (!cv::solvePnP(object_points, image_points, camera_matrix_, dist_coeffs_,
                      candidate_rvec, candidate_tvec, false, method))
    {
      continue;
    }

    if (candidate_rvec.empty() || candidate_tvec.empty())
    {
      continue;
    }

    const double z = candidate_tvec.at<double>(2);
    if (!std::isfinite(z) || z <= 1e-6)
    {
      continue;
    }

    const double reprojection_error =
        ComputeReprojectionError(object_points, image_points, camera_matrix_,
                                 dist_coeffs_, candidate_rvec, candidate_tvec);
    if (!std::isfinite(reprojection_error))
    {
      continue;
    }
    if (!found || reprojection_error < best_error)
    {
      found = true;
      best_error = reprojection_error;
      rvec = candidate_rvec;
      tvec = candidate_tvec;
    }
  }

#if (CV_VERSION_MAJOR > 4) || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 1)
  if (found)
  {
    cv::Mat refined_rvec = rvec.clone();
    cv::Mat refined_tvec = tvec.clone();
    try
    {
      cv::solvePnPRefineLM(object_points, image_points, camera_matrix_, dist_coeffs_,
                           refined_rvec, refined_tvec);
      const double refined_z = refined_tvec.at<double>(2);
      const double refined_error =
          ComputeReprojectionError(object_points, image_points, camera_matrix_,
                                   dist_coeffs_, refined_rvec, refined_tvec);
      if (std::isfinite(refined_z) && refined_z > 1e-6 &&
          std::isfinite(refined_error) && refined_error <= best_error)
      {
        best_error = refined_error;
        rvec = refined_rvec;
        tvec = refined_tvec;
      }
    }
    catch (const cv::Exception&)
    {
    }
  }
#endif

  return found;
}

template <CameraTypes::CameraInfo CameraInfoV>
double PnPSolver<CameraInfoV>::CalculateDistanceToCenter(
    const cv::Point2f& image_point) const
{
  const float cx = static_cast<float>(camera_matrix_.at<double>(0, 2));
  const float cy = static_cast<float>(camera_matrix_.at<double>(1, 2));
  return cv::norm(image_point - cv::Point2f(cx, cy));
}

template <CameraTypes::CameraInfo CameraInfoV>
double PnPSolver<CameraInfoV>::ComputeReprojectionError(
    const std::vector<cv::Point3f>& object_points,
    const std::vector<cv::Point2f>& image_points, const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs, const cv::Mat& rvec, const cv::Mat& tvec)
{
  std::vector<cv::Point2f> projected_points;
  cv::projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs,
                    projected_points);
  if (projected_points.size() != image_points.size())
  {
    return std::numeric_limits<double>::infinity();
  }

  double error_sum = 0.0;
  for (std::size_t index = 0; index < image_points.size(); ++index)
  {
    error_sum += cv::norm(projected_points[index] - image_points[index]);
  }
  return error_sum / static_cast<double>(image_points.size());
}
