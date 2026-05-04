#pragma once

/**
 * @file pnp_solver.hpp
 * @brief 基于编译期相机标定参数的装甲板 PnP 求解器。
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "CameraBase.hpp"
#include "armor.hpp"

/**
 * @brief 将装甲板四个图像角点求解为相机坐标系位姿。
 *
 * 求解器从 CameraInfoV 构造相机内参和畸变系数，并为大小装甲板缓存
 * 物理角点。输入角点约定为左上、右上、右下、左下。
 *
 * @tparam CameraInfoV 编译期相机参数。
 */
template <CameraTypes::CameraInfo CameraInfoV>
class PnPSolver
{
 public:
  using CameraInfo = CameraTypes::CameraInfo; ///< 相机参数类型别名。

  /**
   * @brief 当前求解器绑定的相机标定参数。
   */
  static inline constexpr CameraInfo camera_info = CameraInfoV;

  /**
   * @brief 从 CameraInfoV 中提取出的 PnP 可用畸变系数信息。
   */
  static inline constexpr auto pnp_dist_coeffs_info =
      CameraTypes::BuildPnPDistCoeffs(camera_info);

  /**
   * @brief 构造 PnP 求解器并缓存相机矩阵、畸变系数和装甲板物理角点。
   */
  PnPSolver();

  /**
   * @brief 求解单个装甲板在相机坐标系下的位姿。
   * @param image_armor_points 图像角点，顺序为左上、右上、右下、左下。
   * @param armor_type 装甲板尺寸类型，用于选择物理宽高。
   * @param rvec 输出旋转向量。
   * @param tvec 输出平移向量，单位 m。
   * @param reprojection_error_px 可选输出平均重投影误差，单位 px。
   * @return 至少一个正深度候选成功时返回 true。
   */
  [[nodiscard]] bool SolvePnP(const std::array<cv::Point2f, 4>& image_armor_points,
                              ArmorType armor_type, cv::Mat& rvec,
                              cv::Mat& tvec,
                              double* reprojection_error_px = nullptr) const;

  /**
   * @brief 计算图像点到相机主点的像素距离。
   * @param image_point 图像像素点。
   * @return 到主点的欧氏距离，单位 px。
   */
  [[nodiscard]] double CalculateDistanceToCenter(
      const cv::Point2f& image_point) const;

 private:
  /**
   * @brief 计算一组 3D/2D 对应点的平均重投影误差。
   * @param object_points 物理坐标点，单位 m。
   * @param image_points 图像像素点。
   * @param camera_matrix 相机内参矩阵。
   * @param dist_coeffs 畸变系数矩阵。
   * @param rvec 旋转向量。
   * @param tvec 平移向量。
   * @return 平均像素误差；投影失败时返回 infinity。
   */
  [[nodiscard]] static double ComputeReprojectionError(
      const std::vector<cv::Point3f>& object_points,
      const std::vector<cv::Point2f>& image_points, const cv::Mat& camera_matrix,
      const cv::Mat& dist_coeffs, const cv::Mat& rvec, const cv::Mat& tvec);

  /**
   * @brief 从 CameraInfoV 构造 OpenCV camera_matrix。
   * @return 3x3 CV_64F 相机内参矩阵。
   */
  static cv::Mat BuildCameraMatrix();

  /**
   * @brief 从 CameraInfoV 构造 OpenCV dist_coeffs。
   * @return OpenCV 可直接传入 solvePnP 的畸变系数矩阵。
   */
  static cv::Mat BuildDistCoeffs();

  /**
   * @brief 按装甲板物理尺寸构造四个物理角点。
   * @param width_mm 装甲板宽度，单位 mm。
   * @param height_mm 装甲板高度，单位 mm。
   * @return 相机/PnP 坐标约定下的四个物理角点，单位 m。
   */
  static std::vector<cv::Point3f> BuildArmorPoints(double width_mm,
                                                   double height_mm);

 private:
  cv::Mat camera_matrix_; ///< OpenCV 相机内参矩阵。
  cv::Mat dist_coeffs_;   ///< OpenCV 畸变系数矩阵。
  std::vector<cv::Point3f> small_armor_points_; ///< 小装甲板物理角点。
  std::vector<cv::Point3f> large_armor_points_; ///< 大装甲板物理角点。

  /**
   * @brief 小装甲板宽度，单位 mm。
   */
  inline static constexpr double small_armor_width_mm = 135.0;

  /**
   * @brief 小装甲板高度，单位 mm。
   */
  inline static constexpr double small_armor_height_mm = 56.0;

  /**
   * @brief 大装甲板宽度，单位 mm。
   */
  inline static constexpr double large_armor_width_mm = 225.0;

  /**
   * @brief 大装甲板高度，单位 mm。
   */
  inline static constexpr double large_armor_height_mm = 56.0;
};

/**
 * @brief 构造并缓存 PnP 求解所需的固定矩阵和物理模型。
 */
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

/**
 * @brief 从编译期 CameraInfoV 复制相机矩阵到 OpenCV Mat。
 */
template <CameraTypes::CameraInfo CameraInfoV>
cv::Mat PnPSolver<CameraInfoV>::BuildCameraMatrix()
{
  return cv::Mat(3, 3, CV_64F,
                 const_cast<double*>(camera_info.camera_matrix.data()))
      .clone();
}

/**
 * @brief 根据畸变模型生成 OpenCV PnP 可用的畸变系数。
 */
template <CameraTypes::CameraInfo CameraInfoV>
cv::Mat PnPSolver<CameraInfoV>::BuildDistCoeffs()
{
  if constexpr (pnp_dist_coeffs_info.uses_rational_polynomial_extension)
  {
    XR_LOG_WARN(
        "PnPSolver: 当前使用 8 项 rational 畸变系数；后端支持后再扩展到 14 项。");
  }

  if constexpr (pnp_dist_coeffs_info.requires_undistort_first)
  {
    XR_LOG_WARN(
        "PnPSolver: 当前畸变模型不直接支持 (%d)，应先去畸变到 pinhole "
        "再按 NONE 模型求解 PnP。",
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

/**
 * @brief 生成中心在原点的装甲板四角物理坐标。
 */
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

/**
 * @brief 使用 IPPE 求解装甲板平面位姿。
 */
template <CameraTypes::CameraInfo CameraInfoV>
bool PnPSolver<CameraInfoV>::SolvePnP(
    const std::array<cv::Point2f, 4>& image_armor_points, ArmorType armor_type,
    cv::Mat& rvec, cv::Mat& tvec, double* reprojection_error_px) const
{
  if (reprojection_error_px != nullptr)
  {
    *reprojection_error_px = std::numeric_limits<double>::infinity();
  }

  const std::array<cv::Point2f, 4> base_points = {
      image_armor_points[3], image_armor_points[0], image_armor_points[1],
      image_armor_points[2]};

  const auto& object_points =
      (armor_type == ArmorType::SMALL) ? small_armor_points_ : large_armor_points_;
  const std::vector<cv::Point2f> image_points = {
      base_points[0], base_points[1], base_points[2], base_points[3]};

  cv::Mat candidate_rvec;
  cv::Mat candidate_tvec;
  if (!cv::solvePnP(object_points, image_points, camera_matrix_, dist_coeffs_,
                    candidate_rvec, candidate_tvec, false, cv::SOLVEPNP_IPPE))
  {
    return false;
  }

  if (candidate_rvec.empty() || candidate_tvec.empty())
  {
    return false;
  }

  const double z = candidate_tvec.at<double>(2);
  if (!std::isfinite(z) || z <= 1e-6)
  {
    return false;
  }

  const double reprojection_error =
      ComputeReprojectionError(object_points, image_points, camera_matrix_,
                               dist_coeffs_, candidate_rvec, candidate_tvec);
  if (!std::isfinite(reprojection_error))
  {
    return false;
  }

  rvec = candidate_rvec;
  tvec = candidate_tvec;
  if (reprojection_error_px != nullptr)
  {
    *reprojection_error_px = reprojection_error;
  }
  return true;
}

/**
 * @brief 计算图像点到相机主点的像素距离。
 */
template <CameraTypes::CameraInfo CameraInfoV>
double PnPSolver<CameraInfoV>::CalculateDistanceToCenter(
    const cv::Point2f& image_point) const
{
  const float cx = static_cast<float>(camera_matrix_.at<double>(0, 2));
  const float cy = static_cast<float>(camera_matrix_.at<double>(1, 2));
  return cv::norm(image_point - cv::Point2f(cx, cy));
}

/**
 * @brief 对候选位姿执行 projectPoints 并返回平均像素误差。
 */
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
