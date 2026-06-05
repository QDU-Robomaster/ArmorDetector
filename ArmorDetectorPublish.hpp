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

  MaybeDumpResultsTsv(armors);
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorDetector<CameraInfoV>::MaybeDumpResultsTsv(
    const std::vector<CandidateArmor>& armors)
{
  const char* path = std::getenv("ARMOR_DETECTOR_DUMP_TSV");
  if (path == nullptr || path[0] == '\0')
  {
    return;
  }

  static std::string active_path;
  static std::FILE* file = nullptr;
  if (active_path != path)
  {
    if (file != nullptr)
    {
      std::fclose(file);
      file = nullptr;
    }
    active_path = path;
    file = std::fopen(active_path.c_str(), "w");
    if (file == nullptr)
    {
      XR_LOG_ERROR("ArmorDetector failed to open dump TSV: %s",
                   active_path.c_str());
      active_path.clear();
      return;
    }
    std::fprintf(
        file,
        "frame_index\timage_timestamp_us\tresult_index\tcolor\tnumber\tconfidence\tcenter_x\tcenter_y\tp0_x\tp0_y\tp1_x\tp1_y\tp2_x\tp2_y\tp3_x\tp3_y\n");
    std::fflush(file);
  }

  if (file == nullptr)
  {
    return;
  }

  for (std::size_t index = 0; index < armors.size(); ++index)
  {
    const auto& armor = armors[index];
    std::fprintf(
        file,
        "%llu\t%llu\t%zu\t%u\t%u\t%.9f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
        static_cast<unsigned long long>(frame_index_ + 1U),
        static_cast<unsigned long long>(latest_timestamp_us_),
        index,
        static_cast<unsigned>(armor.color),
        static_cast<unsigned>(armor.number),
        static_cast<double>(armor.confidence),
        static_cast<double>(armor.center.x),
        static_cast<double>(armor.center.y),
        static_cast<double>(armor.points[0].x),
        static_cast<double>(armor.points[0].y),
        static_cast<double>(armor.points[1].x),
        static_cast<double>(armor.points[1].y),
        static_cast<double>(armor.points[2].x),
        static_cast<double>(armor.points[2].y),
        static_cast<double>(armor.points[3].x),
        static_cast<double>(armor.points[3].y));
  }
  std::fflush(file);
}
