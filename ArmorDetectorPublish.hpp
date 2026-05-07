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
      if (cfg_.depth_correction.enabled && tvec.rows >= 3 && tvec.type() == CV_64F)
      {
        const auto& pts = armor.points;
        const double left_height = cv::norm(pts[0] - pts[3]);
        const double right_height = cv::norm(pts[1] - pts[2]);
        const double top_width = cv::norm(pts[0] - pts[1]);
        const double bottom_width = cv::norm(pts[3] - pts[2]);
        const double quad_height = 0.5 * (left_height + right_height);
        const double quad_width = 0.5 * (top_width + bottom_width);
        const cv::Point2f quad_center = detail::quad_center(pts);
        const double raw_z = tvec.at<double>(2);

        if (std::isfinite(raw_z) && std::isfinite(quad_height) &&
            std::isfinite(quad_width) && std::isfinite(pnp_reprojection_error_px) &&
            quad_height >= cfg_.depth_correction.min_quad_height_px)
        {
          const double fx = CameraInfoV.camera_matrix[0];
          const double fy = CameraInfoV.camera_matrix[4];
          const double cx = CameraInfoV.camera_matrix[2];
          const double cy = CameraInfoV.camera_matrix[5];
          const double reprojection_scale = 0.5 * (fx + fy);
          const bool camera_params_valid =
              std::isfinite(fx) && std::isfinite(fy) &&
              std::isfinite(cx) && std::isfinite(cy) &&
              std::isfinite(reprojection_scale) && fx > 1e-6 &&
              fy > 1e-6 && reprojection_scale > 1e-6;

          double feature_height = quad_height;
          double feature_width = quad_width;
          double feature_reprojection = pnp_reprojection_error_px;
          double feature_center_x = static_cast<double>(quad_center.x);
          double feature_center_y = static_cast<double>(quad_center.y);
          bool correction_features_valid = true;

          if (cfg_.depth_correction.camera_normalized_features)
          {
            if (!camera_params_valid)
            {
              correction_features_valid = false;
            }
            else
            {
              feature_height = quad_height / fy;
              feature_width = quad_width / fx;
              feature_reprojection = pnp_reprojection_error_px / reprojection_scale;
              feature_center_x = (static_cast<double>(quad_center.x) - cx) / fx;
              feature_center_y = (static_cast<double>(quad_center.y) - cy) / fy;
            }
          }

          if (correction_features_valid)
          {
            const auto& c = cfg_.depth_correction.coeffs;
            double dz = c[0] + c[1] * raw_z + c[2] * feature_height +
                        c[3] * feature_width +
                        c[4] * feature_reprojection +
                        c[5] * feature_center_x +
                        c[6] * feature_center_y;

            const double correction_limit =
                std::max(0.0, cfg_.depth_correction.max_abs_correction_m);
            if (std::isfinite(dz))
            {
              dz = std::clamp(dz, -correction_limit, correction_limit);
              tvec.at<double>(2) = raw_z - dz;
            }
          }

          if (cfg_.depth_correction.height_fusion_enabled && camera_params_valid)
          {
            const double armor_height_m = cfg_.depth_correction.armor_height_m;
            const double height_weight = cfg_.depth_correction.height_fusion_weight;
            const double current_z = tvec.at<double>(2);
            const double height_z = armor_height_m * fy / quad_height;
            const double height_limit =
                std::max(0.0, cfg_.depth_correction.height_fusion_max_abs_m);
            if (std::isfinite(armor_height_m) && armor_height_m > 0.0 &&
                std::isfinite(height_weight) && height_weight > 0.0 &&
                std::isfinite(current_z) && std::isfinite(height_z))
            {
              const double bounded_delta =
                  std::clamp(height_z - current_z, -height_limit, height_limit);
              tvec.at<double>(2) = current_z + height_weight * bounded_delta;
            }
          }
        }
      }

      result.pnp_valid = true;
      result.pnp_reprojection_error_px = pnp_reprojection_error_px;
      result.pose = detail::make_pose(rvec, tvec);
      ++counters_.pnp_success_count;
    }

    armors_packet_.results.emplace_back(std::move(result));
  }
}
