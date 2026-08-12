#pragma once

/**
 * @file ArmorDetectorInference.hpp
 * @brief ArmorDetector 网络输入构建、推理输出解码和网络候选后处理。
 */

/**
 * @brief 对单帧图像执行网络 detector 主流程。
 *
 * 该函数负责网络输入构建、HailoRT 推理和输出解码。
 * 语义过滤和尺寸类型判定在 DecodeOutput() 内完成。
 *
 * @tparam FrameLayoutV 编译期帧布局。
 * @param raw_img 输入 BGR 图像。
 * @return 本帧有效装甲板候选。
 */
template <CameraTypes::FrameLayout FrameLayoutV>
std::vector<typename ArmorDetector<FrameLayoutV>::CandidateArmor>
ArmorDetector<FrameLayoutV>::Detect(const cv::Mat& raw_img)
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

  detail::NetworkInputMapping input_mapping;
  const auto preprocess_begin = std::chrono::steady_clock::now();
  const cv::Mat input = BuildNetworkInput(raw_img, input_mapping);
  const auto preprocess_end = std::chrono::steady_clock::now();
  last_preprocess_latency_ms_ =
      std::chrono::duration<double, std::milli>(preprocess_end - preprocess_begin)
          .count();
  if (input.empty())
  {
    ++counters_.discarded_count;
    return {};
  }

  cv::Mat output;
  const auto infer_begin = std::chrono::steady_clock::now();
  if (!network_.Infer(input, output))
  {
    const auto infer_end = std::chrono::steady_clock::now();
    last_infer_latency_ms_ =
        std::chrono::duration<double, std::milli>(infer_end - infer_begin).count();
    last_postprocess_latency_ms_ = 0.0;
    return {};
  }
  const auto infer_end = std::chrono::steady_clock::now();
  last_infer_latency_ms_ =
      std::chrono::duration<double, std::milli>(infer_end - infer_begin).count();

  MaybeDumpModelOutput(output);

  const auto postprocess_begin = std::chrono::steady_clock::now();
  auto armors = DecodeOutput(raw_img, input_mapping, output);
  const auto postprocess_end = std::chrono::steady_clock::now();
  last_postprocess_latency_ms_ =
      std::chrono::duration<double, std::milli>(postprocess_end - postprocess_begin)
          .count();
  return armors;
}

template <CameraTypes::FrameLayout FrameLayoutV>
void ArmorDetector<FrameLayoutV>::MaybeDumpModelOutput(const cv::Mat& output)
{
  const char* path = std::getenv("ARMOR_DETECTOR_DUMP_OUTPUT_F32");
  if (path == nullptr || path[0] == '\0')
  {
    return;
  }
  if (output.empty() || output.type() != CV_32F || output.dims != 2)
  {
    return;
  }

  const char* frame_env = std::getenv("ARMOR_DETECTOR_DUMP_OUTPUT_FRAME_INDEX");
  if (frame_env != nullptr && frame_env[0] != '\0')
  {
    const unsigned long requested_frame = std::strtoul(frame_env, nullptr, 10);
    const unsigned long current_frame = static_cast<unsigned long>(frame_index_ + 1U);
    if (requested_frame != current_frame)
    {
      return;
    }
  }

  static std::string active_path;
  static bool dumped = false;
  if (active_path != path)
  {
    active_path = path;
    dumped = false;
  }
  if (dumped)
  {
    return;
  }

  std::FILE* file = std::fopen(active_path.c_str(), "wb");
  if (file == nullptr)
  {
    XR_LOG_ERROR("ArmorDetector failed to open model output dump: %s",
                 active_path.c_str());
    return;
  }

  const int32_t rows = output.rows;
  const int32_t cols = output.cols;
  std::fwrite(&rows, sizeof(rows), 1, file);
  std::fwrite(&cols, sizeof(cols), 1, file);
  for (int r = 0; r < rows; ++r)
  {
    const float* ptr = output.ptr<float>(r);
    std::fwrite(ptr, sizeof(float), static_cast<size_t>(cols), file);
  }
  std::fclose(file);
  dumped = true;
}

/**
 * @brief 将原始图像拉伸成网络张量图像。
 *
 * mapping 描述当前网络张量坐标如何还原到原始图像坐标。
 *
 * @tparam FrameLayoutV 编译期帧布局。
 * @param bgr_img 原始 BGR 图像。
 * @param mapping 输出的坐标还原映射。
 * @return 网络张量 RGB8 图像；输入空图时返回空 Mat。
 */
template <CameraTypes::FrameLayout FrameLayoutV>
cv::Mat ArmorDetector<FrameLayoutV>::BuildNetworkInput(
    const cv::Mat& bgr_img, detail::NetworkInputMapping& mapping) const
{
  if (bgr_img.empty())
  {
    return {};
  }

  const auto input_shape = network_.InputShape();
  mapping.x_scale = static_cast<double>(bgr_img.cols) / std::max(1, input_shape.width);
  mapping.y_scale = static_cast<double>(bgr_img.rows) / std::max(1, input_shape.height);

  cv::Mat input;
  cv::resize(bgr_img, input, cv::Size(input_shape.width, input_shape.height));
  cv::Mat rgb_input;
  cv::cvtColor(input, rgb_input, cv::COLOR_BGR2RGB);
  return rgb_input;
}

/**
 * @brief 将网络输出矩阵转换成内部装甲板候选。
 *
 * 输出先解码为 NetworkDetection，再执行交叠抑制、颜色/编号/置信度过滤和
 * 尺寸类型一致性检查。
 *
 * @tparam FrameLayoutV 编译期帧布局。
 * @param mapping 网络输入到原始图像的坐标映射。
 * @param output 网络输出矩阵。
 * @return 通过所有 detector 后处理门限的候选列表。
 */
template <CameraTypes::FrameLayout FrameLayoutV>
std::vector<typename ArmorDetector<FrameLayoutV>::CandidateArmor>
ArmorDetector<FrameLayoutV>::DecodeOutput(const cv::Mat& raw_img,
                                          const detail::NetworkInputMapping& mapping,
                                          const cv::Mat& output)
{
  std::vector<NetworkDetection> detections;

  const auto& adapter = infer::resolve_model_infer_adapter(cfg_.network.model);
  const detail::ModelOutputView output_view(output, adapter.candidate_count,
                                            adapter.output_width);
  if (!output_view.Valid())
  {
    const auto input_shape = network_.InputShape();
    XR_LOG_ERROR(
        "ArmorDetector output shape invalid: rows=%d cols=%d type=%d expected=%dx%d "
        "input=%dx%d",
        output.rows, output.cols, output.type(), adapter.candidate_count,
        adapter.output_width, input_shape.width, input_shape.height);
    ++counters_.discarded_count;
    return {};
  }

  float max_objectness_logit = -1.0e9F;
  std::array<int, 4> raw_color_hist{0, 0, 0, 0};
  uint32_t objectness_pass_count = 0U;
  uint32_t raw_color_keep_count = 0U;
  uint32_t quad_keep_count = 0U;
  for (int row = 0; row < output_view.CandidateCount(); ++row)
  {
    const float objectness_logit = output_view.At(row, 8);
    max_objectness_logit = std::max(max_objectness_logit, objectness_logit);
    const int raw_color_id = detail::ArgMaxOutputRange(
        output_view, row, adapter.raw_color_begin, adapter.raw_color_end);
    if (raw_color_id >= 0 && raw_color_id < static_cast<int>(raw_color_hist.size()))
    {
      ++raw_color_hist[static_cast<std::size_t>(raw_color_id)];
    }
    if (objectness_logit >= static_cast<float>(cfg_.network.logit_threshold > 0.0
                                                   ? cfg_.network.logit_threshold
                                                   : detail::default_logit_threshold))
    {
      ++objectness_pass_count;
      if (!infer::reject_raw_color(adapter, raw_color_id))
      {
        ++raw_color_keep_count;
      }
    }

    const auto detection = DecodeModelDetection(mapping, output_view, row);
    if (!detection.has_value())
    {
      continue;
    }
    ++quad_keep_count;

    counters_.max_objectness =
        std::max(counters_.max_objectness, static_cast<double>(detection->confidence));
    detections.emplace_back(*detection);
  }
  counters_.decoded_count = static_cast<uint32_t>(detections.size());

  if (detections.empty())
  {
    XR_LOG_INFO(
        "ArmorDetector zero-decode diag max_obj_logit=%.3f max_obj_sigmoid=%.3f "
        "raw_color_hist=[%d,%d,%d,%d] pass_obj=%u pass_color=%u pass_quad=%u",
        max_objectness_logit, detail::Sigmoid(max_objectness_logit), raw_color_hist[0],
        raw_color_hist[1], raw_color_hist[2], raw_color_hist[3], objectness_pass_count,
        raw_color_keep_count, quad_keep_count);
  }

  if (detections.empty())
  {
    return {};
  }

  return FinalizeDetections(raw_img, std::move(detections));
}

template <CameraTypes::FrameLayout FrameLayoutV>
std::vector<typename ArmorDetector<FrameLayoutV>::CandidateArmor>
ArmorDetector<FrameLayoutV>::FinalizeDetections(
    const cv::Mat& raw_img, std::vector<NetworkDetection>&& detections)
{
  if (detections.empty())
  {
    return {};
  }

  const ArmorColor target_color = CurrentTargetColor();
  std::vector<int> indices = SelectDetectionsAfterOpenCvNms(detections);
  const bool is_int8_head = cfg_.network.model == ArmorDetectorModel::INT8_HEAD ||
                            cfg_.network.model == ArmorDetectorModel::INT8_HEAD_L;
  if (!is_int8_head)
  {
    SuppressNearDuplicateDetections(detections, indices);
  }
  counters_.overlap_kept_count = static_cast<uint32_t>(indices.size());

  std::vector<CandidateArmor> armors;
  armors.reserve(indices.size());

  for (const int index : indices)
  {
    CandidateArmor armor = BuildCandidateArmor(detections[index]);

    const bool color_mismatch =
        (target_color != ArmorColor::UNKNOWN) && (armor.color != target_color);
    if (color_mismatch || !ArmorNumberIsKnown(armor.number) ||
        armor.confidence < static_cast<float>(cfg_.network.min_confidence))
    {
      ++counters_.discarded_count;
      ++counters_.semantic_discard_count;
      continue;
    }
    ++counters_.semantic_kept_count;

    if (!is_int8_head)
    {
      ApplyNumberTypePrior(armor);
      if (!ValidateArmorType(armor))
      {
        ++counters_.discarded_count;
        ++counters_.type_discard_count;
        continue;
      }
    }

    armors.emplace_back(std::move(armor));
  }

  return armors;
}

template <CameraTypes::FrameLayout FrameLayoutV>
void ArmorDetector<FrameLayoutV>::SuppressNearDuplicateDetections(
    const std::vector<NetworkDetection>& detections, std::vector<int>& indices) const
{
  if (indices.size() <= 1U)
  {
    return;
  }

  constexpr float kDuplicateCenterDistancePx = 24.0F;
  constexpr float kDuplicateIouThreshold = 0.35F;

  std::vector<int> kept;
  kept.reserve(indices.size());
  for (const int index : indices)
  {
    const auto& candidate = detections[static_cast<std::size_t>(index)];
    bool duplicate = false;
    for (const int kept_index : kept)
    {
      const auto& reference = detections[static_cast<std::size_t>(kept_index)];
      if (candidate.color != reference.color || candidate.number != reference.number)
      {
        continue;
      }

      const cv::Point2f candidate_center = detail::quad_center(candidate.points);
      const cv::Point2f reference_center = detail::quad_center(reference.points);
      const float center_distance = cv::norm(candidate_center - reference_center);
      if (center_distance <= kDuplicateCenterDistancePx)
      {
        duplicate = true;
        break;
      }

      const float iou = detail::rect_iou(candidate.box, reference.box);
      if (iou >= kDuplicateIouThreshold)
      {
        duplicate = true;
        break;
      }
    }

    if (!duplicate)
    {
      kept.push_back(index);
    }
  }

  indices = std::move(kept);
}

/**
 * @brief 执行网络候选 NMS。
 *
 * score threshold 使用 network.min_confidence，IoU threshold 使用
 * network.nms_threshold，然后按 confidence 降序截断 max_detections。
 */
template <CameraTypes::FrameLayout FrameLayoutV>
std::vector<int> ArmorDetector<FrameLayoutV>::SelectDetectionsAfterOpenCvNms(
    const std::vector<NetworkDetection>& detections) const
{
  if (detections.empty())
  {
    return {};
  }

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  boxes.reserve(detections.size());
  scores.reserve(detections.size());
  for (const auto& detection : detections)
  {
    boxes.push_back(detection.box);
    scores.push_back(detection.confidence);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, scores, static_cast<float>(cfg_.network.min_confidence),
                    static_cast<float>(cfg_.network.nms_threshold), indices);
  std::sort(indices.begin(), indices.end(),
            [&detections](int lhs, int rhs)
            {
              return detections[static_cast<std::size_t>(lhs)].confidence >
                     detections[static_cast<std::size_t>(rhs)].confidence;
            });
  const int max_detections = cfg_.network.max_detections > 0
                                 ? cfg_.network.max_detections
                                 : detail::default_max_detections;
  if (static_cast<int>(indices.size()) > max_detections)
  {
    indices.resize(static_cast<std::size_t>(max_detections));
  }
  return indices;
}

/**
 * @brief 解码当前 detector 模型的一行输出。
 *
 * 该 decoder 按当前模型适配器解释 objectness、颜色、编号和角点顺序。
 */
template <CameraTypes::FrameLayout FrameLayoutV>
std::optional<typename ArmorDetector<FrameLayoutV>::NetworkDetection>
ArmorDetector<FrameLayoutV>::DecodeModelDetection(
    const detail::NetworkInputMapping& mapping, const detail::ModelOutputView& output,
    int row) const
{
  return DecodeModelDetectionFromFields(
      mapping, [&output, row](int field) { return output.At(row, field); },
      output.OutputWidth(), row);
}

namespace
{

inline std::tuple<float, float, int> Int8GridCellForRow(int row)
{
  const int stride8_cols = detail::model_input_width / 8;
  const int stride8_count = stride8_cols * (detail::model_input_height / 8);
  const int stride16_cols = detail::model_input_width / 16;
  const int stride16_count = stride16_cols * (detail::model_input_height / 16);
  if (row < stride8_count)
  {
    return {static_cast<float>((row % stride8_cols) * 8),
            static_cast<float>((row / stride8_cols) * 8), 8};
  }
  row -= stride8_count;
  if (row < stride16_count)
  {
    return {static_cast<float>((row % stride16_cols) * 16),
            static_cast<float>((row / stride16_cols) * 16), 16};
  }
  row -= stride16_count;
  const int stride32_cols = detail::model_input_width / 32;
  return {static_cast<float>((row % stride32_cols) * 32),
          static_cast<float>((row / stride32_cols) * 32), 32};
}

}  // namespace

template <CameraTypes::FrameLayout FrameLayoutV>
template <typename FieldReader>
std::optional<typename ArmorDetector<FrameLayoutV>::NetworkDetection>
ArmorDetector<FrameLayoutV>::DecodeModelDetectionFromFields(
    const detail::NetworkInputMapping& mapping, FieldReader&& read, int field_count,
    int row) const
{
  const auto& adapter = infer::resolve_model_infer_adapter(cfg_.network.model);
  const float objectness_logit = read(8);
  const float objectness_value = infer::decode_confidence(adapter, objectness_logit);
  const double prefilter_threshold =
      adapter.confidence_is_logit
          ? cfg_.network.min_confidence
          : (cfg_.network.logit_threshold > 0.0 ? cfg_.network.logit_threshold
                                                : detail::default_logit_threshold);
  if (objectness_value < static_cast<float>(prefilter_threshold))
  {
    return std::nullopt;
  }

  const int raw_color_id =
      detail::ArgMaxFieldRange([&read](int field) { return read(field); },
                               adapter.raw_color_begin, adapter.raw_color_end);
  const int raw_class_id = detail::ArgMaxFieldRange(
      [&read](int field) { return read(field); }, adapter.raw_class_begin,
      std::min(adapter.raw_class_end, field_count));

  std::array<cv::Point2f, 4> declared_points{};
  float grid_offset_x = 0.0F;
  float grid_offset_y = 0.0F;
  float point_stride = 1.0F;
  if (cfg_.network.model == ArmorDetectorModel::INT8_GRID ||
      cfg_.network.model == ArmorDetectorModel::INT8_GRID_L)
  {
    const auto [cx, cy, stride] = Int8GridCellForRow(row);
    grid_offset_x = cx;
    grid_offset_y = cy;
    point_stride = static_cast<float>(stride * 2);
  }
  for (int point_index = 0; point_index < 4; ++point_index)
  {
    const float x = read(point_index * 2) * point_stride + grid_offset_x;
    const float y = read(point_index * 2 + 1) * point_stride + grid_offset_y;
    declared_points[static_cast<std::size_t>(point_index)] = mapping.MapToSource(x, y);
  }

  const auto points = infer::canonicalize_points(adapter, declared_points);
  if (cfg_.network.enable_quad_check &&
      !detail::IsUsableQuad(points, cfg_.network.min_quad_area_px))
  {
    return std::nullopt;
  }

  NetworkDetection detection;
  if (infer::reject_raw_color(adapter, raw_color_id))
  {
    return std::nullopt;
  }
  detection.color = adapter.decode_color(raw_color_id);
  detection.number = adapter.decode_number(raw_class_id);
  detection.confidence = objectness_value;
  detection.points = points;
  const double bbox_expand = cfg_.network.bbox_expand >= 0.0
                                 ? cfg_.network.bbox_expand
                                 : detail::default_bbox_expand;
  detection.box =
      detail::expanded_bounding_rect_from_points(detection.points, bbox_expand);
  return detection;
}

/**
 * @brief 从网络检测单元创建内部候选。
 *
 * 候选会立即计算中心和尺寸比例等几何派生量。
 *
 * @tparam FrameLayoutV 编译期帧布局。
 * @param detection 网络检测单元。
 * @return 初始化完成的内部候选。
 */
template <CameraTypes::FrameLayout FrameLayoutV>
typename ArmorDetector<FrameLayoutV>::CandidateArmor
ArmorDetector<FrameLayoutV>::BuildCandidateArmor(const NetworkDetection& detection) const
{
  CandidateArmor armor;
  armor.color = detection.color;
  armor.number = detection.number;
  armor.confidence = detection.confidence;
  armor.box = detection.box;
  armor.points = detection.points;
  armor.center = detail::quad_center(armor.points);
  UpdateGeometryMetrics(armor);
  return armor;
}
