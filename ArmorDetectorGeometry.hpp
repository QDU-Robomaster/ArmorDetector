#pragma once

/**
 * @file ArmorDetectorGeometry.hpp
 * @brief ArmorDetector 候选几何派生量和尺寸类型判定实现。
 */

/**
 * @brief 检查候选的编号先验和尺寸类型是否冲突。
 *
 * @tparam CameraInfoV 编译期相机参数。
 * @param armor 待检查候选。
 * @return 没有大/小装甲编号冲突时返回 true。
 */
template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorDetector<CameraInfoV>::ValidateArmorType(const CandidateArmor& armor) const
{
  if (network_.OutputLayout() == detail::NetworkOutputLayout::SHTECH22)
  {
    return armor.type != ArmorType::INVALID;
  }

  if (armor.type == ArmorType::SMALL)
  {
    return !ArmorNumberIsLarge(armor.number);
  }

  return !ArmorNumberIsSmall(armor.number);
}

/**
 * @brief 根据编号先验或几何比例更新装甲板尺寸类型。
 *
 * @tparam CameraInfoV 编译期相机参数。
 * @param armor 待更新候选。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorDetector<CameraInfoV>::ApplyNumberTypePrior(
    CandidateArmor& armor) const
{
  if (network_.OutputLayout() == detail::NetworkOutputLayout::SHTECH22)
  {
    const ArmorType shtech_type = detail::shtech_type_from_number(armor.number);
    armor.type = shtech_type != ArmorType::INVALID ? shtech_type
                                                   : InferArmorType(armor);
    return;
  }

  if (ArmorNumberIsLarge(armor.number))
  {
    armor.type = ArmorType::LARGE;
  }
  else if (ArmorNumberIsSmall(armor.number))
  {
    armor.type = ArmorType::SMALL;
  }
  else
  {
    armor.type = InferArmorType(armor);
  }
}

/**
 * @brief 更新候选装甲板的几何比例。
 *
 * ratio 定义为左右竖边中心距除以左右竖边最大长度，用于无法从编号确定大小时的
 * 尺寸类型推断。
 *
 * @tparam CameraInfoV 编译期相机参数。
 * @param armor 待更新候选。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorDetector<CameraInfoV>::UpdateGeometryMetrics(CandidateArmor& armor) const
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
 * @tparam CameraInfoV 编译期相机参数。
 * @param armor 待判断候选。
 * @return ratio 明确时返回 LARGE/SMALL，灰区返回 INVALID。
 */
template <CameraTypes::CameraInfo CameraInfoV>
ArmorType ArmorDetector<CameraInfoV>::GeometryTypeHint(
    const CandidateArmor& armor) const
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
 * @tparam CameraInfoV 编译期相机参数。
 * @param armor 待推断候选。
 * @return 推断出的尺寸类型。
 */
template <CameraTypes::CameraInfo CameraInfoV>
ArmorType ArmorDetector<CameraInfoV>::InferArmorType(const CandidateArmor& armor) const
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
 * @brief 判断 refine 编号是否与明确几何尺寸冲突。
 *
 * @tparam CameraInfoV 编译期相机参数。
 * @param number 分类器输出编号。
 * @param armor 当前候选。
 * @return 未冲突时返回 true。
 */
template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorDetector<CameraInfoV>::RefinedNumberCompatibleWithGeometry(
    ArmorNumber number, const CandidateArmor& armor) const
{
  if (!cfg_.number_refine.enforce_type_compatibility)
  {
    return true;
  }

  const ArmorType geometry_type = GeometryTypeHint(armor);
  if (geometry_type == ArmorType::INVALID)
  {
    return true;
  }
  if (ArmorNumberIsLarge(number))
  {
    return geometry_type == ArmorType::LARGE;
  }
  if (ArmorNumberIsSmall(number))
  {
    return geometry_type == ArmorType::SMALL;
  }
  return true;
}

/**
 * @brief 将像素中心归一化到图像宽高范围。
 *
 * @tparam CameraInfoV 编译期相机参数。
 * @param bgr_img 源图像。
 * @param center 像素中心。
 * @return 归一化中心。
 */
template <CameraTypes::CameraInfo CameraInfoV>
cv::Point2f ArmorDetector<CameraInfoV>::GetNormalizedCenter(
    const cv::Mat& bgr_img, const cv::Point2f& center) const
{
  return {center.x / static_cast<float>(std::max(1, bgr_img.cols)),
          center.y / static_cast<float>(std::max(1, bgr_img.rows))};
}
