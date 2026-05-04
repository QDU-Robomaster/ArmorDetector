#pragma once

// Network stage: ROI selection, model inference, YOLO output decode, semantic
// filtering, and candidate construction.
template <CameraTypes::CameraInfo CameraInfoV>
std::vector<typename ArmorDetector<CameraInfoV>::CandidateArmor>
ArmorDetector<CameraInfoV>::Detect(const cv::Mat& raw_img)
{
  counters_.decoded_count = 0U;
  counters_.nms_count = 0U;
  counters_.semantic_kept_count = 0U;
  counters_.pnp_success_count = 0U;
  counters_.refined_count = 0U;
  counters_.discarded_count = 0U;
  counters_.semantic_discard_count = 0U;
  counters_.type_discard_count = 0U;
  counters_.max_objectness = 0.0;

  if (!network_.Ready())
  {
    XR_LOG_ERROR("ArmorDetector YOLOv5 model is not ready");
    return {};
  }

  cv::Mat detector_img = raw_img;
  cv::Point2f offset(0.0F, 0.0F);
  cv::Rect clipped_roi(0, 0, raw_img.cols, raw_img.rows);

  if (cfg_.yolo.use_roi)
  {
    int roi_width = cfg_.yolo.roi_width;
    int roi_height = cfg_.yolo.roi_height;
    if (roi_width < 0)
    {
      roi_width = raw_img.cols;
    }
    if (roi_height < 0)
    {
      roi_height = raw_img.rows;
    }

    const cv::Rect full_roi(0, 0, raw_img.cols, raw_img.rows);
    const cv::Rect roi(cfg_.yolo.roi_x, cfg_.yolo.roi_y, roi_width, roi_height);
    clipped_roi = roi & full_roi;
    if (clipped_roi.empty())
    {
      ++counters_.discarded_count;
      return {};
    }

    detector_img = raw_img(clipped_roi);
    offset = cv::Point2f(static_cast<float>(clipped_roi.x),
                         static_cast<float>(clipped_roi.y));
  }

  const auto input_shape = network_.InputShape();
  const double height_scale =
      static_cast<double>(input_shape.height) / std::max(1, detector_img.rows);
  const double width_scale =
      static_cast<double>(input_shape.width) / std::max(1, detector_img.cols);
  const double scale = std::min(height_scale, width_scale);
  const int resized_height =
      std::max(1, static_cast<int>(std::round(detector_img.rows * scale)));
  const int resized_width =
      std::max(1, static_cast<int>(std::round(detector_img.cols * scale)));
  const bool centered_letterbox =
      diagnostics_.center_letterbox || diagnostics_.yolo_letterbox;
  const int input_offset_x =
      centered_letterbox ? std::max(0, (input_shape.width - resized_width) / 2) : 0;
  const int input_offset_y =
      centered_letterbox ? std::max(0, (input_shape.height - resized_height) / 2) : 0;
  const cv::Point2f input_offset(static_cast<float>(input_offset_x),
                                 static_cast<float>(input_offset_y));
  const cv::Scalar letterbox_fill =
      diagnostics_.yolo_letterbox ? cv::Scalar(114, 114, 114)
                                  : cv::Scalar(0, 0, 0);

  cv::Mat input(input_shape.height, input_shape.width, CV_8UC3,
                letterbox_fill);
  cv::resize(detector_img,
             input(cv::Rect(input_offset_x, input_offset_y, resized_width, resized_height)),
             cv::Size(resized_width, resized_height));

  cv::Mat output;
  if (!network_.Infer(input, output))
  {
    return {};
  }

  auto armors = DecodeOutput(scale, input_offset, output, detector_img);
  if (armors.empty())
  {
    return armors;
  }

  if (cfg_.yolo.use_roi)
  {
    for (auto& armor : armors)
    {
      for (auto& point : armor.points)
      {
        point += offset;
      }
      armor.center += offset;
      armor.center_norm = GetNormalizedCenter(raw_img, armor.center);
      armor.box.x += static_cast<int>(offset.x);
      armor.box.y += static_cast<int>(offset.y);
    }
  }

  return armors;
}

template <CameraTypes::CameraInfo CameraInfoV>
std::vector<typename ArmorDetector<CameraInfoV>::CandidateArmor>
ArmorDetector<CameraInfoV>::DecodeOutput(double scale, const cv::Point2f& input_offset,
                                         const cv::Mat& output, const cv::Mat& bgr_img)
{
  std::vector<NetworkDetection> detections;
  const ArmorColor target_color = detail::detect_color_from_config(cfg_.detect_color);

  for (int row = 0; row < output.rows; ++row)
  {
    const double score =
        1.0 / (1.0 + std::exp(-output.at<float>(row, detail::OutputLayout::objectness_index)));
    if (score > counters_.max_objectness)
    {
      counters_.max_objectness = score;
    }

    const auto detection = DecodeDetection(scale, input_offset, output, row);
    if (!detection.has_value())
    {
      continue;
    }

    detections.emplace_back(*detection);
  }
  counters_.decoded_count = static_cast<uint32_t>(detections.size());

  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  boxes.reserve(detections.size());
  confidences.reserve(detections.size());
  for (const auto& detection : detections)
  {
    boxes.emplace_back(detection.box);
    confidences.emplace_back(detection.confidence);
  }
  if (boxes.empty())
  {
    return {};
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, static_cast<float>(cfg_.yolo.score_threshold),
                    static_cast<float>(cfg_.yolo.nms_threshold), indices);
  counters_.nms_count = static_cast<uint32_t>(indices.size());

  std::vector<CandidateArmor> armors;
  armors.reserve(indices.size());

  for (const int index : indices)
  {
    CandidateArmor armor = BuildCandidateArmor(detections[index], bgr_img);

    const bool color_mismatch =
        (target_color != ArmorColor::UNKNOWN) && (armor.color != target_color);
    if (color_mismatch ||
        !ArmorNumberIsKnown(armor.number) ||
        armor.confidence < static_cast<float>(cfg_.yolo.min_confidence))
    {
      ++counters_.discarded_count;
      ++counters_.semantic_discard_count;
      continue;
    }
    ++counters_.semantic_kept_count;

    if (cfg_.yolo.use_traditional_refine &&
        !diagnostics_.disable_traditional_refine)
    {
      if (RefineArmorCorners(armor, bgr_img))
      {
        ++counters_.refined_count;
      }
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
    if (!ValidateArmorType(armor))
    {
      ++counters_.discarded_count;
      ++counters_.type_discard_count;
      continue;
    }

    armors.emplace_back(std::move(armor));
  }

  return armors;
}

template <CameraTypes::CameraInfo CameraInfoV>
std::optional<typename ArmorDetector<CameraInfoV>::NetworkDetection>
ArmorDetector<CameraInfoV>::DecodeDetection(double scale,
                                            const cv::Point2f& input_offset,
                                            const cv::Mat& output, int row) const
{
  double score = output.at<float>(row, detail::OutputLayout::objectness_index);
  score = 1.0 / (1.0 + std::exp(-score));
  if (score < cfg_.yolo.score_threshold)
  {
    return std::nullopt;
  }

  const cv::Mat color_scores =
      output.row(row).colRange(detail::OutputLayout::color_begin,
                               detail::OutputLayout::color_end);
  const cv::Mat number_scores =
      output.row(row).colRange(detail::OutputLayout::number_begin,
                               detail::OutputLayout::number_end);
  cv::Point color_id_point;
  cv::Point number_id_point;
  cv::minMaxLoc(color_scores, nullptr, nullptr, nullptr, &color_id_point);
  cv::minMaxLoc(number_scores, nullptr, nullptr, nullptr, &number_id_point);

  const std::array<cv::Point2f, 4> unsorted_points = {
      cv::Point2f((output.at<float>(row, 0) - input_offset.x) / scale,
                  (output.at<float>(row, 1) - input_offset.y) / scale),
      cv::Point2f((output.at<float>(row, 6) - input_offset.x) / scale,
                  (output.at<float>(row, 7) - input_offset.y) / scale),
      cv::Point2f((output.at<float>(row, 4) - input_offset.x) / scale,
                  (output.at<float>(row, 5) - input_offset.y) / scale),
      cv::Point2f((output.at<float>(row, 2) - input_offset.x) / scale,
                  (output.at<float>(row, 3) - input_offset.y) / scale)};

  NetworkDetection detection;
  detection.color = detail::color_from_yolo_id(color_id_point.x);
  detection.number = detail::number_from_yolo_id(number_id_point.x);
  detection.confidence = static_cast<float>(score);
  detection.points = detail::sort_keypoints(unsorted_points);
  detection.box = detail::bounding_rect_from_points(detection.points);
  return detection;
}

template <CameraTypes::CameraInfo CameraInfoV>
typename ArmorDetector<CameraInfoV>::CandidateArmor
ArmorDetector<CameraInfoV>::BuildCandidateArmor(
    const NetworkDetection& detection, const cv::Mat& bgr_img) const
{
  CandidateArmor armor;
  armor.color = detection.color;
  armor.number = detection.number;
  armor.confidence = detection.confidence;
  armor.box = detection.box;
  armor.points = detection.points;
  armor.raw_points_valid = true;
  armor.raw_points = detection.points;
  armor.center = detail::quad_center(armor.points);
  armor.center_norm = GetNormalizedCenter(bgr_img, armor.center);
  UpdateGeometryMetrics(armor);
  return armor;
}
