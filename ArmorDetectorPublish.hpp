#pragma once

/**
 * @file ArmorDetectorPublish.hpp
 * @brief ArmorDetector 输出消息填充和 PnP 发布阶段实现。
 */

/**
 * @brief 将内部候选填充为对外 detector 结果包。
 *
 * 每个候选都会复制语义和 2D 几何，并使用模块相机参数执行 PnP。
 * PnP 成功时 pose、pnp_valid 和 pnp_reprojection_error_px 才有效。
 *
 * @tparam CameraInfoV 编译期相机参数。
 * @param armors 内部候选列表。
 * @param bgr_img 源图像，用于归一化中心。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorDetector<CameraInfoV>::FillResultMessage(
    const std::vector<CandidateArmor>& armors, const cv::Mat& bgr_img)
{
  armors_packet_.image_timestamp_us = latest_timestamp_us_;
  armors_packet_.results.clear();
  armors_packet_.results.reserve(armors.size());

  for (const auto& armor : armors)
  {
    ArmorDetectorResult result;
    result.color = armor.color;
    result.number = armor.number;
    result.type = armor.type;
    result.priority = GetArmorPriority(armor.number);
    result.confidence = armor.confidence;
    result.box = armor.box;
    result.points = armor.points;
    result.center = armor.center;
    result.center_norm = GetNormalizedCenter(bgr_img, armor.center);
    result.distance_to_image_center = pnp_solver_.CalculateDistanceToCenter(armor.center);

    cv::Mat rvec;
    cv::Mat tvec;
    double pnp_reprojection_error_px = 0.0;
    if (pnp_solver_.SolvePnP(armor.points, armor.type, rvec, tvec,
                             &pnp_reprojection_error_px))
    {
      result.pnp_valid = true;
      result.pnp_reprojection_error_px = pnp_reprojection_error_px;
      result.pose = detail::make_pose(rvec, tvec);
      ++counters_.pnp_success_count;
    }

    armors_packet_.results.emplace_back(std::move(result));
  }
}
