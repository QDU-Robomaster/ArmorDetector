#pragma once

/**
 * @file ArmorDetectorInference.hpp
 * @brief ArmorDetector 网络输入构建、推理输出解码和网络候选后处理。
 */

/**
 * @brief 对单帧图像执行网络 detector 主流程。
 *
 * 该函数负责 ROI 裁剪、网络输入构建、OpenVINO 推理、输出解码和 ROI 坐标还原。
 * 语义过滤和尺寸类型判定在 DecodeOutput() 内完成。
 *
 * @tparam CameraInfoV 编译期相机参数。
 * @param raw_img 输入 BGR 图像。
 * @return 本帧有效装甲板候选。
 */
template <CameraTypes::CameraInfo CameraInfoV>
std::vector<typename ArmorDetector<CameraInfoV>::CandidateArmor>
ArmorDetector<CameraInfoV>::Detect(const cv::Mat& raw_img)
{
  counters_.decoded_count = 0U;
  counters_.overlap_kept_count = 0U;
  counters_.semantic_kept_count = 0U;
  counters_.pnp_success_count = 0U;
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

  if (cfg_.network.use_roi)
  {
    int roi_width = cfg_.network.roi_width;
    int roi_height = cfg_.network.roi_height;
    if (roi_width < 0)
    {
      roi_width = raw_img.cols;
    }
    if (roi_height < 0)
    {
      roi_height = raw_img.rows;
    }

    const cv::Rect full_roi(0, 0, raw_img.cols, raw_img.rows);
    const cv::Rect roi(cfg_.network.roi_x, cfg_.network.roi_y, roi_width, roi_height);
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

  if (cfg_.network.use_roi)
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

/**
 * @brief 将 detector 图像拉伸成 dense-grid 模型输入。
 *
 * mapping 描述 640x512 模型输入坐标如何还原到 detector_img 坐标。
 *
 * @tparam CameraInfoV 编译期相机参数。
 * @param detector_img detector 处理区域图像。
 * @param mapping 输出的坐标还原映射。
 * @return 网络输入尺寸 BGR8 图像；输入空图时返回空 Mat。
 */
template <CameraTypes::CameraInfo CameraInfoV>
cv::Mat ArmorDetector<CameraInfoV>::BuildNetworkInput(
    const cv::Mat& detector_img, detail::NetworkInputMapping& mapping) const
{
  if (detector_img.empty())
  {
    return {};
  }

  const auto input_shape = network_.InputShape();
  mapping.x_scale =
      static_cast<double>(detector_img.cols) / std::max(1, input_shape.width);
  mapping.y_scale =
      static_cast<double>(detector_img.rows) / std::max(1, input_shape.height);
  mapping.input_offset = {};

  cv::Mat input;
  cv::resize(detector_img, input, cv::Size(input_shape.width, input_shape.height));
  return input;
}

/**
 * @brief 将网络输出矩阵转换成内部装甲板候选。
 *
 * 输出先解码为 NetworkDetection，再执行交叠抑制、颜色/编号/置信度过滤和
 * 尺寸类型一致性检查。
 *
 * @tparam CameraInfoV 编译期相机参数。
 * @param mapping 网络输入到 detector 图像的坐标映射。
 * @param output 网络输出矩阵。
 * @param bgr_img detector 源图像。
 * @return 通过所有 detector 后处理门限的候选列表。
 */
template <CameraTypes::CameraInfo CameraInfoV>
std::vector<typename ArmorDetector<CameraInfoV>::CandidateArmor>
ArmorDetector<CameraInfoV>::DecodeOutput(
    const detail::NetworkInputMapping& mapping, const cv::Mat& output,
    const cv::Mat& bgr_img)
{
  std::vector<NetworkDetection> detections;
  const ArmorColor target_color = detail::detect_color_from_config(cfg_.detect_color);

  const detail::DirectKeypointOutputView output_view(output);
  if (!output_view.Valid())
  {
    XR_LOG_ERROR("ArmorDetector output shape invalid: rows=%d cols=%d type=%d",
                 output.rows, output.cols, output.type());
    ++counters_.discarded_count;
    return {};
  }

  for (int row = 0; row < output_view.CandidateCount(); ++row)
  {
    const auto detection = DecodeDirectKeypointDetection(mapping, output_view, row);
    if (!detection.has_value())
    {
      continue;
    }

    counters_.max_objectness =
        std::max(counters_.max_objectness,
                 static_cast<double>(detection->confidence));
    detections.emplace_back(*detection);
  }
  counters_.decoded_count = static_cast<uint32_t>(detections.size());

  if (detections.empty())
  {
    return {};
  }

  const std::vector<int> indices = SelectDetectionsAfterOverlapSuppression(detections);
  counters_.overlap_kept_count = static_cast<uint32_t>(indices.size());

  std::vector<CandidateArmor> armors;
  armors.reserve(indices.size());

  for (const int index : indices)
  {
    CandidateArmor armor = BuildCandidateArmor(detections[index], bgr_img);

    const bool color_mismatch =
        (target_color != ArmorColor::UNKNOWN) && (armor.color != target_color);
    if (color_mismatch ||
        !ArmorNumberIsKnown(armor.number) ||
        armor.confidence < static_cast<float>(cfg_.network.min_confidence))
    {
      ++counters_.discarded_count;
      ++counters_.semantic_discard_count;
      continue;
    }
    ++counters_.semantic_kept_count;

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

/**
 * @brief 执行 dense-grid 源语义的交叠抑制。
 *
 * 源模型后处理是先按 confidence 降序取前 128 个候选，再丢弃与已保留候选
 * 有任意 bbox 交叠的框；这里复现这个语义，避免把 dense-grid 模型误套成
 * IoU NMS。
 *
 * @tparam CameraInfoV 编译期相机参数。
 * @param detections 已解码候选。
 * @return 保留候选下标。
 */
template <CameraTypes::CameraInfo CameraInfoV>
std::vector<int>
ArmorDetector<CameraInfoV>::SelectDetectionsAfterOverlapSuppression(
    const std::vector<NetworkDetection>& detections) const
{
  if (detections.empty())
  {
    return {};
  }

  std::vector<DetectionSelection> ordered;
  ordered.reserve(detections.size());
  for (std::size_t index = 0; index < detections.size(); ++index)
  {
    ordered.push_back({index, detections[index].confidence, detections[index].box});
  }

  std::sort(ordered.begin(), ordered.end(),
            [](const DetectionSelection& lhs, const DetectionSelection& rhs)
            {
              return lhs.confidence > rhs.confidence;
            });

  std::vector<int> indices;
  indices.reserve(std::min<std::size_t>(
      ordered.size(), static_cast<std::size_t>(detail::direct_keypoint_keep_topk)));
  const std::size_t limit = std::min<std::size_t>(
      ordered.size(), static_cast<std::size_t>(detail::direct_keypoint_keep_topk));
  for (std::size_t ordered_index = 0; ordered_index < limit; ++ordered_index)
  {
    const auto& candidate = ordered[ordered_index];
    bool overlaps = false;
    for (const int kept_index : indices)
    {
      if ((candidate.box & detections[kept_index].box).area() > 0)
      {
        overlaps = true;
        break;
      }
    }
    if (!overlaps)
    {
      indices.push_back(static_cast<int>(candidate.index));
    }
  }

  return indices;
}

/**
 * @brief 解码 dense-grid keypoint 模型的单行输出。
 *
 * dense-grid 模型使用拉伸输入；每行先由网格中心和 stride 还原到模型输入坐标，
 * 再统一成 detector/PnP 使用的左上、右上、右下、左下顺序。
 *
 * @tparam CameraInfoV 编译期相机参数。
 * @param mapping 网络输入到 detector 图像的坐标映射。
 * @param output 网络输出矩阵。
 * @param row 待解码输出行。
 * @return 通过置信度和可选四边形门限时返回网络检测单元。
 */
template <CameraTypes::CameraInfo CameraInfoV>
std::optional<typename ArmorDetector<CameraInfoV>::NetworkDetection>
ArmorDetector<CameraInfoV>::DecodeDirectKeypointDetection(
    const detail::NetworkInputMapping& mapping,
    const detail::DirectKeypointOutputView& output, int row) const
{
  if (!output.Valid() || row >= output.CandidateCount())
  {
    return std::nullopt;
  }

  const double score =
      output.At(row, detail::DirectKeypointOutputLayout::objectness_index);
  if (score < cfg_.network.score_threshold)
  {
    return std::nullopt;
  }

  const int color_id =
      detail::ArgMaxRowRange(output, row,
                             detail::DirectKeypointOutputLayout::color_begin,
                             detail::DirectKeypointOutputLayout::color_end);
  const int class_id = detail::ArgMaxRowRange(
      output, row, detail::DirectKeypointOutputLayout::number_begin,
      detail::DirectKeypointOutputLayout::number_end);

  const auto cell = detail::DirectKeypointGridCellForRow(row);
  std::array<cv::Point2f, 4> declared_points{};
  for (int point_index = 0; point_index < 4; ++point_index)
  {
    const float x =
        output.At(row, detail::DirectKeypointOutputLayout::point_begin +
                           point_index * 2) *
            static_cast<float>(cell.stride * 2) +
        static_cast<float>(cell.center_x);
    const float y =
        output.At(row, detail::DirectKeypointOutputLayout::point_begin +
                           point_index * 2 + 1) *
            static_cast<float>(cell.stride * 2) +
        static_cast<float>(cell.center_y);
    declared_points[static_cast<std::size_t>(point_index)] =
        mapping.MapToSource(x, y);
  }

  std::swap(declared_points[2], declared_points[3]);
  const auto points =
      detail::direct_keypoint_declared_to_canonical(declared_points);
  if (cfg_.network.enable_quad_check &&
      !detail::IsUsableQuad(points, cfg_.network.min_quad_area_px))
  {
    return std::nullopt;
  }

  NetworkDetection detection;
  detection.color = detail::color_from_direct_keypoint_id(color_id);
  detection.number = detail::number_from_direct_keypoint_class_id(class_id);
  detection.confidence = static_cast<float>(score);
  detection.points = points;
  detection.box = detail::bounding_rect_from_points(detection.points);
  return detection;
}

/**
 * @brief 从网络检测单元创建内部候选。
 *
 * 候选会立即计算中心、归一化中心和尺寸比例等几何派生量。
 *
 * @tparam CameraInfoV 编译期相机参数。
 * @param detection 网络检测单元。
 * @param bgr_img detector 源图像。
 * @return 初始化完成的内部候选。
 */
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
  armor.center = detail::quad_center(armor.points);
  armor.center_norm = GetNormalizedCenter(bgr_img, armor.center);
  UpdateGeometryMetrics(armor);
  return armor;
}
