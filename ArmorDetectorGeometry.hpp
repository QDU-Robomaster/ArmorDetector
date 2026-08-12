#pragma once

/**
 * @file ArmorDetectorGeometry.hpp
 * @brief ArmorDetector 候选几何派生量和尺寸类型判定实现。
 */

/**
 * @brief 检查候选的编号先验和尺寸类型是否冲突。
 *
 * @tparam FrameLayoutV 编译期帧布局。
 * @param armor 待检查候选。
 * @return 没有大/小装甲编号冲突时返回 true。
 */
template <CameraTypes::FrameLayout FrameLayoutV>
bool ArmorDetector<FrameLayoutV>::ValidateArmorType(const CandidateArmor& armor) const
{
  return armor.type != ArmorType::INVALID;
}

/**
 * @brief 根据编号先验或几何比例更新装甲板尺寸类型。
 *
 * @tparam FrameLayoutV 编译期帧布局。
 * @param armor 待更新候选。
 */
template <CameraTypes::FrameLayout FrameLayoutV>
void ArmorDetector<FrameLayoutV>::ApplyNumberTypePrior(CandidateArmor& armor) const
{
  const ArmorType model_type = detail::type_from_model_number(armor.number);
  armor.type = model_type != ArmorType::INVALID ? model_type : InferArmorType(armor);
}

/**
 * @brief 更新候选装甲板的几何比例。
 *
 * ratio 定义为左右竖边中心距除以左右竖边最大长度，用于无法从编号确定大小时的
 * 尺寸类型推断。
 *
 * @tparam FrameLayoutV 编译期帧布局。
 * @param armor 待更新候选。
 */
template <CameraTypes::FrameLayout FrameLayoutV>
void ArmorDetector<FrameLayoutV>::UpdateGeometryMetrics(CandidateArmor& armor) const
{
  const cv::Point2f left_center = (armor.points[0] + armor.points[3]) * 0.5F;
  const cv::Point2f right_center = (armor.points[1] + armor.points[2]) * 0.5F;
  const cv::Point2f left_edge = armor.points[3] - armor.points[0];
  const cv::Point2f right_edge = armor.points[2] - armor.points[1];
  const cv::Point2f left_to_right = right_center - left_center;

  const double width = cv::norm(left_to_right);
  const double left_length = cv::norm(left_edge);
  const double right_length = cv::norm(right_edge);
  const double max_edge_length = std::max(left_length, right_length);

  armor.ratio = width / std::max(max_edge_length, 1e-6);
}

/**
 * @brief 根据几何比例给出明确大小提示。
 *
 * @tparam FrameLayoutV 编译期帧布局。
 * @param armor 待判断候选。
 * @return ratio 明确时返回 LARGE/SMALL，灰区返回 INVALID。
 */
template <CameraTypes::FrameLayout FrameLayoutV>
ArmorType ArmorDetector<FrameLayoutV>::GeometryTypeHint(const CandidateArmor& armor) const
{
  if (armor.ratio > 3.0)
  {
    return ArmorType::LARGE;
  }
  if (armor.ratio < 2.5)
  {
    return ArmorType::SMALL;
  }
  return ArmorType::INVALID;
}

/**
 * @brief 根据几何比例和编号先验推断装甲板尺寸类型。
 *
 * @tparam FrameLayoutV 编译期帧布局。
 * @param armor 待推断候选。
 * @return 推断出的尺寸类型。
 */
template <CameraTypes::FrameLayout FrameLayoutV>
ArmorType ArmorDetector<FrameLayoutV>::InferArmorType(const CandidateArmor& armor) const
{
  const ArmorType geometry_type = GeometryTypeHint(armor);
  if (geometry_type != ArmorType::INVALID)
  {
    return geometry_type;
  }

  if (ArmorNumberIsLarge(armor.number))
  {
    return ArmorType::LARGE;
  }
  return ArmorType::SMALL;
}

/**
 * @brief 将像素中心归一化到图像宽高范围。
 *
 * @tparam FrameLayoutV 编译期帧布局。
 * @param bgr_img 源图像。
 * @param center 像素中心。
 * @return 归一化中心。
 */
template <CameraTypes::FrameLayout FrameLayoutV>
cv::Point2f ArmorDetector<FrameLayoutV>::GetNormalizedCenter(
    const cv::Mat& bgr_img, const cv::Point2f& center) const
{
  return {center.x / static_cast<float>(std::max(1, bgr_img.cols)),
          center.y / static_cast<float>(std::max(1, bgr_img.rows))};
}
