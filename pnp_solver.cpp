#include "pnp_solver.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include <opencv2/calib3d.hpp>

PnPSolver::PnPSolver(const CameraBase::CameraInfo& camera_info)
    : camera_matrix_(cv::Mat(3, 3, CV_64F,
                             const_cast<double*>(camera_info.camera_matrix.data()))
                         .clone())
{
  const auto dist_coeffs = CameraBase::CameraInfo::ToPnPDistCoeffs(
      camera_info.distortion_model, camera_info.distortion_coefficients);
  if (dist_coeffs.empty())
  {
    dist_coeffs_ = cv::Mat();
  }
  else
  {
    dist_coeffs_ = cv::Mat(1, static_cast<int>(dist_coeffs.size()), CV_64F,
                           const_cast<double*>(dist_coeffs.data()))
                       .clone();
  }

  // 将毫米转换为米，并构造模型点
  constexpr double SMALL_HALF_T = SMALL_ARMOR_WIDTH * 0.5 / 1000.0;
  constexpr double SMALL_HALF_Z = SMALL_ARMOR_HEIGHT * 0.5 / 1000.0;
  constexpr double LARGE_HALF_Y = LARGE_ARMOR_WIDTH * 0.5 / 1000.0;
  constexpr double LARGE_HALF_Z = LARGE_ARMOR_HEIGHT * 0.5 / 1000.0;

  // 点顺序：左下、左上、右上、右下（顺时针）
  // 模型坐标：x 前，y 左，z 上
  small_armor_points_ = {
      {0.0f, static_cast<float>(SMALL_HALF_T), static_cast<float>(-SMALL_HALF_Z)},
      {0.0f, static_cast<float>(SMALL_HALF_T), static_cast<float>(SMALL_HALF_Z)},
      {0.0f, static_cast<float>(-SMALL_HALF_T), static_cast<float>(SMALL_HALF_Z)},
      {0.0f, static_cast<float>(-SMALL_HALF_T), static_cast<float>(-SMALL_HALF_Z)}};

  large_armor_points_ = {
      {0.0f, static_cast<float>(LARGE_HALF_Y), static_cast<float>(-LARGE_HALF_Z)},
      {0.0f, static_cast<float>(LARGE_HALF_Y), static_cast<float>(LARGE_HALF_Z)},
      {0.0f, static_cast<float>(-LARGE_HALF_Y), static_cast<float>(LARGE_HALF_Z)},
      {0.0f, static_cast<float>(-LARGE_HALF_Y), static_cast<float>(-LARGE_HALF_Z)}};
}

bool PnPSolver::SolvePnP(const std::array<cv::Point2f, 4>& image_armor_points,
                         ArmorType armor_type, cv::Mat& rvec, cv::Mat& tvec) const
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

double PnPSolver::CalculateDistanceToCenter(const cv::Point2f& image_point)
{
  float cx = static_cast<float>(camera_matrix_.at<double>(0, 2));
  float cy = static_cast<float>(camera_matrix_.at<double>(1, 2));
  return cv::norm(image_point - cv::Point2f(cx, cy));
}
