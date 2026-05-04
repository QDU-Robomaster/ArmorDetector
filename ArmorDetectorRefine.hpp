#pragma once

// Traditional image-processing stage: binary extraction, lightbar fitting, corner
// refinement, geometry classification, and refine diagnostics.
template <CameraTypes::CameraInfo CameraInfoV>
cv::Mat ArmorDetector<CameraInfoV>::BuildTraditionalBinary(
    const cv::Mat& bgr_img, ArmorColor target_color) const
{
  if (bgr_img.empty())
  {
    return {};
  }

  std::vector<cv::Mat> channels;
  cv::split(bgr_img, channels);
  if (channels.size() < 3U)
  {
    cv::Mat gray_img =
        (channels.size() == 1U) ? channels[0] : cv::Mat();
    if (gray_img.empty())
    {
      cv::cvtColor(bgr_img, gray_img, cv::COLOR_BGR2GRAY);
    }
    cv::Mat binary_img;
    cv::threshold(gray_img, binary_img, cfg_.traditional.threshold, 255,
                  cv::THRESH_BINARY);
    return binary_img;
  }

  cv::Mat intensity_img;
  if (target_color == ArmorColor::BLUE)
  {
    cv::subtract(channels[0], channels[2], intensity_img);
  }
  else if (target_color == ArmorColor::RED)
  {
    cv::subtract(channels[2], channels[0], intensity_img);
  }
  else
  {
    cv::absdiff(channels[0], channels[2], intensity_img);
  }

  cv::Mat binary_img;
  cv::threshold(intensity_img, binary_img, cfg_.traditional.threshold, 255,
                cv::THRESH_BINARY);
  return binary_img;
}

template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorDetector<CameraInfoV>::RefineArmorCorners(
    CandidateArmor& armor, const cv::Mat& bgr_img)
{
  ++counters_.refine_attempt_count;
  const cv::Point2f top_left = armor.points[0];
  const cv::Point2f top_right = armor.points[1];
  const cv::Point2f bottom_right = armor.points[2];
  const cv::Point2f bottom_left = armor.points[3];

  const cv::Point2f left_to_bottom = bottom_left - top_left;
  const cv::Point2f right_to_bottom = bottom_right - top_right;
  const cv::Point2f top_left_1 = (top_left + bottom_left) * 0.5F - left_to_bottom;
  const cv::Point2f bottom_left_1 = (top_left + bottom_left) * 0.5F + left_to_bottom;
  const cv::Point2f bottom_right_1 =
      (top_right + bottom_right) * 0.5F + right_to_bottom;
  const cv::Point2f top_right_1 = (top_right + bottom_right) * 0.5F - right_to_bottom;

  const cv::Point2f top_left_to_top_right = top_right_1 - top_left_1;
  const cv::Point2f bottom_left_to_bottom_right = bottom_right_1 - bottom_left_1;
  const cv::Point2f top_left_2 =
      (top_left_1 + top_right) * 0.5F - 0.75F * top_left_to_top_right;
  const cv::Point2f top_right_2 =
      (top_left_1 + top_right) * 0.5F + 0.75F * top_left_to_top_right;
  const cv::Point2f bottom_left_2 =
      (bottom_left_1 + bottom_right) * 0.5F - 0.75F * bottom_left_to_bottom_right;
  const cv::Point2f bottom_right_2 =
      (bottom_left_1 + bottom_right) * 0.5F + 0.75F * bottom_left_to_bottom_right;

  std::vector<cv::Point> points = {top_left_2, top_right_2, bottom_right_2, bottom_left_2};
  const cv::Rect bounding_box = cv::minAreaRect(points).boundingRect();
  if (bounding_box.x < 0 || bounding_box.y < 0 ||
      (bounding_box.x + bounding_box.width) > bgr_img.cols ||
      (bounding_box.y + bounding_box.height) > bgr_img.rows)
  {
    ++counters_.refine_fail_bbox_oob_count;
    return false;
  }

  const cv::Mat armor_roi = bgr_img(bounding_box);
  if (armor_roi.empty())
  {
    ++counters_.refine_fail_roi_empty_count;
    return false;
  }

  ArmorColor refine_target_color = armor.color;
  if (refine_target_color == ArmorColor::UNKNOWN)
  {
    refine_target_color = detail::detect_color_from_config(cfg_.detect_color);
  }
  cv::Mat binary_img = BuildTraditionalBinary(armor_roi, refine_target_color);

  auto lightbars = DetectLightbars(armor_roi, binary_img);
  if (lightbars.size() < 2U)
  {
    if (lightbars.empty())
    {
      ++counters_.refine_fail_lightbar_zero_count;
      MaybeDumpRefineFailure("lightbar_zero", armor, bounding_box, armor_roi,
                             binary_img, lightbars);
    }
    else
    {
      ++counters_.refine_fail_lightbar_one_count;
      MaybeDumpRefineFailure("lightbar_one", armor, bounding_box, armor_roi,
                             binary_img, lightbars);
    }
    return false;
  }

  const cv::Point2f roi_offset(static_cast<float>(bounding_box.x),
                               static_cast<float>(bounding_box.y));
  Lightbar* closest_left = nullptr;
  Lightbar* closest_right = nullptr;
  float best_pair_distance = std::numeric_limits<float>::max();

  for (std::size_t lhs = 0; lhs < lightbars.size(); ++lhs)
  {
    for (std::size_t rhs = lhs + 1; rhs < lightbars.size(); ++rhs)
    {
      Lightbar* left = &lightbars[lhs];
      Lightbar* right = &lightbars[rhs];
      if (left->center.x > right->center.x)
      {
        std::swap(left, right);
      }

      const float left_distance =
          cv::norm(top_left - (left->top + roi_offset)) +
          cv::norm(bottom_left - (left->bottom + roi_offset));
      const float right_distance =
          cv::norm(top_right - (right->top + roi_offset)) +
          cv::norm(bottom_right - (right->bottom + roi_offset));
      const float pair_distance = left_distance + right_distance;
      if (pair_distance < best_pair_distance)
      {
        best_pair_distance = pair_distance;
        closest_left = left;
        closest_right = right;
      }
    }
  }

  if (closest_left == nullptr || closest_right == nullptr)
  {
    ++counters_.refine_fail_pair_distance_count;
    MaybeDumpRefineFailure("pair_distance", armor, bounding_box, armor_roi,
                           binary_img, lightbars);
    return false;
  }

  armor.points[0] = closest_left->top + roi_offset;
  armor.points[1] = closest_right->top + roi_offset;
  armor.points[2] = closest_right->bottom + roi_offset;
  armor.points[3] = closest_left->bottom + roi_offset;
  armor.center = detail::quad_center(armor.points);
  armor.center_norm = GetNormalizedCenter(bgr_img, armor.center);
  armor.box = cv::boundingRect(
      std::vector<cv::Point2f>(armor.points.begin(), armor.points.end()));
  armor.refined = true;
  UpdateGeometryMetrics(armor);
  return true;
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorDetector<CameraInfoV>::MaybeDumpRefineFailure(
    const char* reason, const CandidateArmor& armor, const cv::Rect& bounding_box,
    const cv::Mat& armor_roi, const cv::Mat& binary_img,
    const std::vector<Lightbar>& lightbars)
{
  if (!diagnostics_.dump_refine_fails || armor_roi.empty() || binary_img.empty())
  {
    return;
  }
  if (diagnostics_.dump_refine_fails_count >=
      diagnostics_.dump_refine_fails_max)
  {
    return;
  }

  const uint32_t dump_index = diagnostics_.dump_refine_fails_count++;
  std::error_code ec;
  std::filesystem::create_directories(diagnostics_.dump_refine_fails_dir, ec);
  if (ec)
  {
    XR_LOG_WARN("ArmorDetector failed to create refine dump dir: %s",
                diagnostics_.dump_refine_fails_dir.c_str());
    return;
  }

  std::ostringstream base_name;
  base_name << diagnostics_.dump_refine_fails_dir << "/frame" << std::setw(6)
            << std::setfill('0') << frame_index_ << "_ts"
            << latest_timestamp_us_ << "_dump" << std::setw(2)
            << dump_index << "_" << reason;
  const std::string base_path = base_name.str();

  cv::Mat overlay = armor_roi.clone();
  const cv::Point2f roi_offset(static_cast<float>(bounding_box.x),
                               static_cast<float>(bounding_box.y));
  std::vector<cv::Point> armor_quad;
  armor_quad.reserve(armor.points.size());
  for (const auto& point : armor.points)
  {
    armor_quad.emplace_back(
        cv::Point(cvRound(point.x - roi_offset.x), cvRound(point.y - roi_offset.y)));
  }
  cv::polylines(overlay, std::vector<std::vector<cv::Point>>{armor_quad}, true,
                cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

  for (const auto& lightbar : lightbars)
  {
    const cv::Point top(cvRound(lightbar.top.x), cvRound(lightbar.top.y));
    const cv::Point bottom(cvRound(lightbar.bottom.x), cvRound(lightbar.bottom.y));
    cv::line(overlay, top, bottom, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    cv::circle(overlay, cv::Point(cvRound(lightbar.center.x), cvRound(lightbar.center.y)),
               2, cv::Scalar(255, 0, 255), cv::FILLED, cv::LINE_AA);
    cv::putText(overlay, std::to_string(lightbar.id), top, cv::FONT_HERSHEY_SIMPLEX,
                0.35, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
  }

  cv::imwrite(base_path + "_roi.png", armor_roi);
  cv::imwrite(base_path + "_bin.png", binary_img);
  cv::imwrite(base_path + "_overlay.png", overlay);

  std::ofstream meta(base_path + ".txt");
  meta << "reason=" << reason << '\n';
  meta << "frame_index=" << frame_index_ << '\n';
  meta << "timestamp_us=" << latest_timestamp_us_ << '\n';
  meta << "threshold=" << cfg_.traditional.threshold << '\n';
  meta << "bbox=" << bounding_box.x << "," << bounding_box.y << ","
       << bounding_box.width << "," << bounding_box.height << '\n';
  meta << "armor_box=" << armor.box.x << "," << armor.box.y << "," << armor.box.width
       << "," << armor.box.height << '\n';
  meta << "lightbar_count=" << lightbars.size() << '\n';
  meta << "armor_points=";
  for (std::size_t index = 0; index < armor.points.size(); ++index)
  {
    if (index != 0U)
    {
      meta << ';';
    }
    meta << armor.points[index].x << ',' << armor.points[index].y;
  }
  meta << '\n';
  for (const auto& lightbar : lightbars)
  {
    meta << "lightbar[" << lightbar.id << "]"
         << " center=" << lightbar.center.x << "," << lightbar.center.y
         << " top=" << lightbar.top.x << "," << lightbar.top.y
         << " bottom=" << lightbar.bottom.x << "," << lightbar.bottom.y
         << " len=" << lightbar.length << " width=" << lightbar.width
         << " ratio=" << lightbar.ratio << " angle_deg="
         << (lightbar.angle * 180.0 / CV_PI) << " fill=" << lightbar.fill_ratio
         << '\n';
  }
}

template <CameraTypes::CameraInfo CameraInfoV>
std::vector<typename ArmorDetector<CameraInfoV>::Lightbar>
ArmorDetector<CameraInfoV>::DetectLightbars(const cv::Mat& bgr_img,
                                            const cv::Mat& binary_img) const
{
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  std::vector<Lightbar> lightbars;
  lightbars.reserve(contours.size());
  std::size_t lightbar_id = 0U;

  for (const auto& contour : contours)
  {
    if (contour.size() < 5U)
    {
      continue;
    }

    const auto rotated_rect = cv::minAreaRect(contour);
    const cv::Rect bounding_rect = cv::boundingRect(contour);
    if (bounding_rect.width <= 0 || bounding_rect.height <= 0)
    {
      continue;
    }

    cv::Mat mask = cv::Mat::zeros(bounding_rect.size(), CV_8UC1);
    std::vector<cv::Point> shifted_contour;
    shifted_contour.reserve(contour.size());
    for (const auto& point : contour)
    {
      shifted_contour.emplace_back(point - bounding_rect.tl());
    }
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{shifted_contour}, 255);
    std::vector<cv::Point> fit_points;
    cv::findNonZero(mask, fit_points);
    if (fit_points.size() < 5U)
    {
      continue;
    }

    Lightbar lightbar;
    lightbar.id = lightbar_id++;
    lightbar.rect = rotated_rect;
    lightbar.center = rotated_rect.center;
    lightbar.fill_ratio =
        static_cast<double>(fit_points.size()) /
        std::max(1.0, static_cast<double>(rotated_rect.size.area()));

    cv::Vec4f fit_line;
    cv::fitLine(fit_points, fit_line, cv::DIST_L2, 0, 0.01, 0.01);
    const cv::Point2f line_dir(fit_line[0], fit_line[1]);
    const cv::Point2f line_origin(
        fit_line[2] + static_cast<float>(bounding_rect.x),
        fit_line[3] + static_cast<float>(bounding_rect.y));

    float projection_min = std::numeric_limits<float>::max();
    float projection_max = std::numeric_limits<float>::lowest();
    for (const auto& point : fit_points)
    {
      const cv::Point2f absolute_point(
          static_cast<float>(point.x + bounding_rect.x),
          static_cast<float>(point.y + bounding_rect.y));
      const cv::Point2f delta = absolute_point - line_origin;
      const float projection = delta.x * line_dir.x + delta.y * line_dir.y;
      projection_min = std::min(projection_min, projection);
      projection_max = std::max(projection_max, projection);
    }

    lightbar.top = line_origin + line_dir * projection_min;
    lightbar.bottom = line_origin + line_dir * projection_max;
    if (lightbar.top.y > lightbar.bottom.y)
    {
      std::swap(lightbar.top, lightbar.bottom);
    }
    lightbar.top_to_bottom = lightbar.bottom - lightbar.top;
    lightbar.width =
        std::max(1.0f, std::min(rotated_rect.size.width, rotated_rect.size.height));
    lightbar.length = cv::norm(lightbar.top_to_bottom);
    lightbar.ratio = lightbar.length / std::max(lightbar.width, 1e-6);
    lightbar.angle = std::atan2(lightbar.top_to_bottom.y, lightbar.top_to_bottom.x);
    lightbar.angle_error = std::abs(lightbar.angle - CV_PI / 2.0);

    if (!ValidateLightbar(lightbar))
    {
      continue;
    }

    lightbar.color = GetContourColor(bgr_img, contour);
    lightbars.emplace_back(lightbar);
  }

  std::sort(lightbars.begin(), lightbars.end(),
            [](const Lightbar& lhs, const Lightbar& rhs)
            {
              return lhs.center.x < rhs.center.x;
            });
  return lightbars;
}

template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorDetector<CameraInfoV>::ValidateLightbar(const Lightbar& lightbar) const
{
  const double max_angle_error = cfg_.traditional.max_angle_error_deg * detail::deg2rad;
  return lightbar.angle_error < max_angle_error &&
         lightbar.ratio > cfg_.traditional.min_lightbar_ratio &&
         lightbar.ratio < cfg_.traditional.max_lightbar_ratio &&
         lightbar.length > cfg_.traditional.min_lightbar_length;
}

template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorDetector<CameraInfoV>::ValidateArmorType(const CandidateArmor& armor) const
{
  if (armor.type == ArmorType::SMALL)
  {
    return !ArmorNumberIsLarge(armor.number);
  }

  return !ArmorNumberIsSmall(armor.number);
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorDetector<CameraInfoV>::UpdateGeometryMetrics(CandidateArmor& armor) const
{
  const cv::Point2f left_center = (armor.points[0] + armor.points[3]) * 0.5F;
  const cv::Point2f right_center = (armor.points[1] + armor.points[2]) * 0.5F;
  const cv::Point2f left_light = armor.points[3] - armor.points[0];
  const cv::Point2f right_light = armor.points[2] - armor.points[1];
  const cv::Point2f left_to_right = right_center - left_center;

  const double width = cv::norm(left_to_right);
  const double left_length = cv::norm(left_light);
  const double right_length = cv::norm(right_light);
  const double max_lightbar_length = std::max(left_length, right_length);

  armor.ratio = width / std::max(max_lightbar_length, 1e-6);
}

template <CameraTypes::CameraInfo CameraInfoV>
ArmorType ArmorDetector<CameraInfoV>::InferArmorType(const CandidateArmor& armor) const
{
  if (armor.ratio > 3.0)
  {
    return ArmorType::LARGE;
  }
  if (armor.ratio < 2.5)
  {
    return ArmorType::SMALL;
  }

  if (ArmorNumberIsLarge(armor.number))
  {
    return ArmorType::LARGE;
  }
  return ArmorType::SMALL;
}

template <CameraTypes::CameraInfo CameraInfoV>
cv::Point2f ArmorDetector<CameraInfoV>::GetNormalizedCenter(
    const cv::Mat& bgr_img, const cv::Point2f& center) const
{
  return {center.x / static_cast<float>(std::max(1, bgr_img.cols)),
          center.y / static_cast<float>(std::max(1, bgr_img.rows))};
}

template <CameraTypes::CameraInfo CameraInfoV>
ArmorColor ArmorDetector<CameraInfoV>::GetContourColor(
    const cv::Mat& bgr_img, const std::vector<cv::Point>& contour) const
{
  int red_sum = 0;
  int blue_sum = 0;
  for (const auto& point : contour)
  {
    const cv::Vec3b pixel = bgr_img.at<cv::Vec3b>(point);
    blue_sum += pixel[0];
    red_sum += pixel[2];
  }
  return (blue_sum > red_sum) ? ArmorColor::BLUE : ArmorColor::RED;
}
