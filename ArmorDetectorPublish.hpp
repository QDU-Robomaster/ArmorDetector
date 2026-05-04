#pragma once

// Output stage: convert accepted candidates to detector topic payloads and run
// PnP using the module camera model.
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
    result.raw_points_valid = armor.raw_points_valid;
    result.refined = armor.refined;
    result.raw_points = armor.raw_points;
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
