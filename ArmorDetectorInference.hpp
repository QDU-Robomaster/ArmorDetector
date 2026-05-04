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
    XR_LOG_ERROR("ArmorDetector model is not ready");
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

  detail::NetworkInputMapping input_mapping;
  const cv::Mat input = BuildNetworkInput(detector_img, input_mapping);
  if (input.empty())
  {
    ++counters_.discarded_count;
    return {};
  }

  cv::Mat output;
  if (!network_.Infer(input, output))
  {
    return {};
  }

  auto armors = DecodeOutput(input_mapping, output, detector_img);
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
cv::Mat ArmorDetector<CameraInfoV>::BuildNetworkInput(
    const cv::Mat& detector_img, detail::NetworkInputMapping& mapping) const
{
  if (detector_img.empty())
  {
    return {};
  }

  const auto input_shape = network_.InputShape();
  const auto profile_spec = detail::ProfileSpecFor(cfg_.yolo.model_profile);
  if (profile_spec.resize_mode == detail::ResizeMode::STRETCH)
  {
    mapping.x_scale =
        static_cast<double>(detector_img.cols) / std::max(1, input_shape.width);
    mapping.y_scale =
        static_cast<double>(detector_img.rows) / std::max(1, input_shape.height);
    mapping.input_offset = {};

    cv::Mat input;
    cv::resize(detector_img, input, cv::Size(input_shape.width, input_shape.height));
    return input;
  }

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
  const cv::Scalar letterbox_fill =
      diagnostics_.yolo_letterbox ? cv::Scalar(114, 114, 114)
                                  : cv::Scalar(0, 0, 0);

  mapping.x_scale = 1.0 / std::max(scale, 1e-9);
  mapping.y_scale = mapping.x_scale;
  mapping.input_offset =
      cv::Point2f(static_cast<float>(input_offset_x), static_cast<float>(input_offset_y));

  cv::Mat input(input_shape.height, input_shape.width, CV_8UC3, letterbox_fill);
  cv::resize(detector_img,
             input(cv::Rect(input_offset_x, input_offset_y, resized_width,
                            resized_height)),
             cv::Size(resized_width, resized_height));
  return input;
}

template <CameraTypes::CameraInfo CameraInfoV>
std::vector<typename ArmorDetector<CameraInfoV>::CandidateArmor>
ArmorDetector<CameraInfoV>::DecodeOutput(
    const detail::NetworkInputMapping& mapping, const cv::Mat& output,
    const cv::Mat& bgr_img)
{
  std::vector<NetworkDetection> detections;
  const ArmorColor target_color = detail::detect_color_from_config(cfg_.detect_color);
  const auto profile_spec = detail::ProfileSpecFor(cfg_.yolo.model_profile);

  cv::Mat output_rows = output;
  cv::Mat transposed_output;
  if (output_rows.cols != detail::OutputLayout::number_end &&
      output_rows.rows == detail::OutputLayout::number_end)
  {
    cv::transpose(output_rows, transposed_output);
    output_rows = transposed_output;
  }

  for (int row = 0; row < output_rows.rows; ++row)
  {
    if (output_rows.cols <= detail::OutputLayout::objectness_index)
    {
      continue;
    }

    const double objectness =
        output_rows.at<float>(row, detail::OutputLayout::objectness_index);
    const double score = 1.0 / (1.0 + std::exp(-objectness));
    counters_.max_objectness = std::max(counters_.max_objectness, score);

    const auto detection = DecodeNetworkDetection(mapping, output_rows, row);
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
    boxes.emplace_back(
        detail::ExpandRect(detection.box, profile_spec.nms_box_padding_ratio));
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
ArmorDetector<CameraInfoV>::DecodeNetworkDetection(
    const detail::NetworkInputMapping& mapping, const cv::Mat& output, int row) const
{
  switch (cfg_.yolo.model_profile)
  {
    case DetectorProfile::YOLO_KEYPOINT_640X640:
      return DecodeYoloKeypointDetection(mapping, output, row);
    case DetectorProfile::DIRECT_KEYPOINT_640X512:
      return DecodeDirectKeypointDetection(mapping, output, row);
    default:
      return std::nullopt;
  }
}

template <CameraTypes::CameraInfo CameraInfoV>
std::optional<typename ArmorDetector<CameraInfoV>::NetworkDetection>
ArmorDetector<CameraInfoV>::DecodeYoloKeypointDetection(
    const detail::NetworkInputMapping& mapping, const cv::Mat& output, int row) const
{
  if (output.cols < detail::OutputLayout::number_end)
  {
    return std::nullopt;
  }

  double score = output.at<float>(row, detail::OutputLayout::objectness_index);
  score = 1.0 / (1.0 + std::exp(-score));
  if (score < cfg_.yolo.score_threshold)
  {
    return std::nullopt;
  }

  const int color_id = detail::ArgMaxRowRange(
      output, row, detail::OutputLayout::color_begin, detail::OutputLayout::color_end);
  const int number_id = detail::ArgMaxRowRange(
      output, row, detail::OutputLayout::number_begin, detail::OutputLayout::number_end);

  const std::array<cv::Point2f, 4> unsorted_points = {
      mapping.MapToSource(output.at<float>(row, 0), output.at<float>(row, 1)),
      mapping.MapToSource(output.at<float>(row, 6), output.at<float>(row, 7)),
      mapping.MapToSource(output.at<float>(row, 4), output.at<float>(row, 5)),
      mapping.MapToSource(output.at<float>(row, 2), output.at<float>(row, 3))};

  NetworkDetection detection;
  detection.color = detail::color_from_yolo_id(color_id);
  detection.number = detail::number_from_yolo_id(number_id);
  detection.confidence = static_cast<float>(score);
  detection.points = detail::sort_keypoints(unsorted_points);
  detection.box = detail::bounding_rect_from_points(detection.points);
  return detection;
}

template <CameraTypes::CameraInfo CameraInfoV>
std::optional<typename ArmorDetector<CameraInfoV>::NetworkDetection>
ArmorDetector<CameraInfoV>::DecodeDirectKeypointDetection(
    const detail::NetworkInputMapping& mapping, const cv::Mat& output, int row) const
{
  if (output.cols < detail::OutputLayout::number_end)
  {
    return std::nullopt;
  }

  double score = output.at<float>(row, detail::OutputLayout::objectness_index);
  score = 1.0 / (1.0 + std::exp(-score));
  if (score < cfg_.yolo.score_threshold)
  {
    return std::nullopt;
  }

  const int color_id = detail::ArgMaxRowRange(
      output, row, detail::OutputLayout::color_begin, detail::OutputLayout::color_end);
  const int class_id = detail::ArgMaxRowRange(
      output, row, detail::OutputLayout::number_begin, detail::OutputLayout::number_end);

  const std::array<cv::Point2f, 4> raw_points = {
      mapping.MapToSource(output.at<float>(row, 0), output.at<float>(row, 1)),
      mapping.MapToSource(output.at<float>(row, 2), output.at<float>(row, 3)),
      mapping.MapToSource(output.at<float>(row, 4), output.at<float>(row, 5)),
      mapping.MapToSource(output.at<float>(row, 6), output.at<float>(row, 7))};
  const auto points = detail::sort_keypoints(raw_points);
  if (cfg_.yolo.enable_quad_check &&
      !detail::IsUsableQuad(points, cfg_.yolo.min_quad_area_px))
  {
    return std::nullopt;
  }

  NetworkDetection detection;
  detection.color = detail::color_from_direct_keypoint_id(color_id);
  const int target_id = detail::direct_keypoint_target_id_from_class_id(class_id);
  detection.number = detail::number_from_direct_keypoint_target_id(target_id);
  detection.confidence = static_cast<float>(score);
  detection.points = points;
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
