#pragma once

#include <algorithm>
#include <array>
#include <cmath>
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

  PnPSolver();

  [[nodiscard]] bool SolvePnP(const std::array<cv::Point2f, 4>& image_armor_points,
                              ArmorType armor_type, cv::Mat& rvec,
                              cv::Mat& tvec) const;

  [[nodiscard]] double CalculateDistanceToCenter(
      const cv::Point2f& image_point) const;

 private:
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
  inline static constexpr double small_armor_height_mm = 55.0;
  inline static constexpr double large_armor_width_mm = 225.0;
  inline static constexpr double large_armor_height_mm = 55.0;
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
  const auto dist_coeffs = CameraTypes::CameraInfo::ToPnPDistCoeffs(
      camera_info.distortion_model, camera_info.distortion_coefficients);
  if (dist_coeffs.empty())
  {
    return {};
  }

  return cv::Mat(1, static_cast<int>(dist_coeffs.size()), CV_64F,
                 const_cast<double*>(dist_coeffs.data()))
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
  std::array<int, 4> order = {0, 1, 2, 3};
  constexpr std::array<int, 3> methods = {
      cv::SOLVEPNP_IPPE,
      cv::SOLVEPNP_ITERATIVE,
      cv::SOLVEPNP_EPNP,
  };

  do
  {
    const std::vector<cv::Point2f> image_points = {
        base_points[order[0]], base_points[order[1]], base_points[order[2]],
        base_points[order[3]]};

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

      rvec = candidate_rvec;
      tvec = candidate_tvec;
      return true;
    }
  }
  while (std::next_permutation(order.begin(), order.end()));

  return false;
}

template <CameraTypes::CameraInfo CameraInfoV>
double PnPSolver<CameraInfoV>::CalculateDistanceToCenter(
    const cv::Point2f& image_point) const
{
  const float cx = static_cast<float>(camera_matrix_.at<double>(0, 2));
  const float cy = static_cast<float>(camera_matrix_.at<double>(1, 2));
  return cv::norm(image_point - cv::Point2f(cx, cy));
}
