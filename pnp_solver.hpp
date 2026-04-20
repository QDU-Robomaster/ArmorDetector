#pragma once

#include <array>
#include <opencv2/core.hpp>
#include <vector>

#include "CameraBase.hpp"
#include "armor.hpp"

/**
 * @class PnPSolver
 * @brief 使用 SolvePnP 估计装甲板在相机坐标系下的位姿。
 */
class PnPSolver
{
 public:
  /**
   * @brief 构造函数
   * @param camera_info 相机内参和畸变参数。
   */
  explicit PnPSolver(const CameraTypes::CameraInfo& camera_info);

  /**
   * @brief 估计装甲板的三维位姿。
   * @param image_armor_points 输入图像四角点，顺序为左上、右上、右下、左下。
   * @param armor_type 装甲板类型。
   * @param rvec 输出旋转向量 (Rodrigues)，从模型坐标到相机坐标。
   * @param tvec 输出平移向量 (米)，从模型原点到相机原点。
   * @return true 计算成功。
   * @return false 计算失败（如点退化）。
   */
  [[nodiscard]] bool SolvePnP(const std::array<cv::Point2f, 4>& image_armor_points,
                              ArmorType armor_type, cv::Mat& rvec,
                              cv::Mat& tvec) const;

  /**
   * @brief 计算图像点到图像中心（主点）的像素距离。
   * @param image_point 输入图像点（像素坐标）。
   * @return 到主点 (cx, cy) 的像素距离。
   */
  [[nodiscard]] double CalculateDistanceToCenter(const cv::Point2f& image_point);

 private:
  cv::Mat camera_matrix_;  ///< 相机内参矩阵 (3x3, CV_64F)
  cv::Mat dist_coeffs_;    ///< 相机畸变参数 (1x5, CV_64F)

  // 装甲板尺寸（毫米）
  inline static constexpr float SMALL_ARMOR_WIDTH = 135.f;
  inline static constexpr float SMALL_ARMOR_HEIGHT = 55.f;
  inline static constexpr float LARGE_ARMOR_WIDTH = 225.f;
  inline static constexpr float LARGE_ARMOR_HEIGHT = 55.f;

  // 装甲板四个角点（单位：米），按顺时针顺序排列
  std::vector<cv::Point3f> small_armor_points_;
  std::vector<cv::Point3f> large_armor_points_;
};
