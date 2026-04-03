#include "pnp_solver.hpp"

#include <algorithm>
#include <array>

#include <opencv2/calib3d.hpp>

PnPSolver::PnPSolver(std::array<double, 9>& camera_matrix,
                     std::array<double, 5>& dist_coeffs)
    : camera_matrix_(cv::Mat(3, 3, CV_64F, camera_matrix.data()).clone()),
      dist_coeffs_(cv::Mat(1, 5, CV_64F, dist_coeffs.data()).clone())
{
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

bool PnPSolver::SolvePnP(const Armor& armor, cv::Mat& rvec, cv::Mat& tvec)
{
  const std::array<cv::Point2f, 4> base_points = {
      armor.left_light.bottom, armor.left_light.top, armor.right_light.top,
      armor.right_light.bottom};
  const auto& object_points =
      (armor.type == ArmorType::SMALL) ? small_armor_points_ : large_armor_points_;

  std::array<int, 4> order = {0, 1, 2, 3};
  const std::array<int, 3> methods = {cv::SOLVEPNP_IPPE, cv::SOLVEPNP_ITERATIVE,
                                      cv::SOLVEPNP_EPNP};

  do
  {
    std::vector<cv::Point2f> image_armor_points = {
        base_points[order[0]], base_points[order[1]], base_points[order[2]],
        base_points[order[3]]};

    for (int method : methods)
    {
      if (!cv::solvePnP(object_points, image_armor_points, camera_matrix_, dist_coeffs_,
                        rvec, tvec, false, method))
      {
        continue;
      }

      if (rvec.empty() || tvec.empty())
      {
        continue;
      }

      const double z = tvec.at<double>(2);
      if (std::isfinite(z) && z > 1e-6)
      {
        return true;
      }
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
