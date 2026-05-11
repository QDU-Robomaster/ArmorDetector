#pragma once

/**
 * @file ArmorDetectorInference.hpp
 * @brief ArmorDetector 网络输入构建、推理输出解码和网络候选后处理。
 */

/**
 * @brief 对单帧图像执行网络 detector 主流程。
 *
 * 该函数负责网络输入构建、OpenVINO 推理和输出解码。
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

  detail::NetworkInputMapping input_mapping;
  const cv::Mat input = BuildNetworkInput(raw_img, input_mapping);
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

  return DecodeOutput(raw_img, input_mapping, output);
}

/**
 * @brief 将原始图像拉伸成网络张量图像。
 *
 * mapping 描述当前网络张量坐标如何还原到原始图像坐标。
 *
 * @tparam CameraInfoV 编译期相机参数。
 * @param bgr_img 原始 BGR 图像。
 * @param mapping 输出的坐标还原映射。
 * @return 网络张量 RGB8 图像；输入空图时返回空 Mat。
 */
template <CameraTypes::CameraInfo CameraInfoV>
cv::Mat ArmorDetector<CameraInfoV>::BuildNetworkInput(
    const cv::Mat& bgr_img, detail::NetworkInputMapping& mapping) const
{
  if (bgr_img.empty())
  {
    return {};
  }

  const auto input_shape = network_.InputShape();
  mapping.x_scale =
      static_cast<double>(bgr_img.cols) / std::max(1, input_shape.width);
  mapping.y_scale =
      static_cast<double>(bgr_img.rows) / std::max(1, input_shape.height);

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
 * @tparam CameraInfoV 编译期相机参数。
 * @param mapping 网络输入到原始图像的坐标映射。
 * @param output 网络输出矩阵。
 * @return 通过所有 detector 后处理门限的候选列表。
 */
template <CameraTypes::CameraInfo CameraInfoV>
std::vector<typename ArmorDetector<CameraInfoV>::CandidateArmor>
ArmorDetector<CameraInfoV>::DecodeOutput(
    const cv::Mat& raw_img, const detail::NetworkInputMapping& mapping,
    const cv::Mat& output)
{
  std::vector<NetworkDetection> detections;
  const ArmorColor target_color = CurrentTargetColor();

  const detail::ModelOutputView output_view(output);
  if (!output_view.Valid())
  {
    const auto input_shape = network_.InputShape();
    XR_LOG_ERROR(
        "ArmorDetector output shape invalid: rows=%d cols=%d type=%d expected=%dx%d input=%dx%d",
        output.rows, output.cols, output.type(), detail::model_candidate_count,
        detail::model_output_width, input_shape.width, input_shape.height);
    ++counters_.discarded_count;
    return {};
  }

  for (int row = 0; row < output_view.CandidateCount(); ++row)
  {
    const auto detection = DecodeModelDetection(mapping, output_view, row);
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

  const std::vector<int> indices = SelectDetectionsAfterOpenCvNms(detections);
  counters_.overlap_kept_count = static_cast<uint32_t>(indices.size());

  std::vector<CandidateArmor> armors;
  armors.reserve(indices.size());

  for (const int index : indices)
  {
    CandidateArmor armor = BuildCandidateArmor(detections[index]);

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

    ApplyNumberTypePrior(armor);
    if (!ValidateArmorType(armor))
    {
      ++counters_.discarded_count;
      ++counters_.type_discard_count;
      continue;
    }

    RefineNumberIfConfident(raw_img, armor);

    armors.emplace_back(std::move(armor));
  }

  return armors;
}

/**
 * @brief 执行网络候选 NMS。
 *
 * score threshold 使用 network.min_confidence，IoU threshold 使用
 * network.nms_threshold，然后按 confidence 降序截断 max_detections。
 */
template <CameraTypes::CameraInfo CameraInfoV>
std::vector<int> ArmorDetector<CameraInfoV>::SelectDetectionsAfterOpenCvNms(
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
  cv::dnn::NMSBoxes(
      boxes, scores, static_cast<float>(cfg_.network.min_confidence),
      static_cast<float>(cfg_.network.nms_threshold), indices);
  std::sort(indices.begin(), indices.end(),
            [&detections](int lhs, int rhs)
            {
              return detections[static_cast<std::size_t>(lhs)].confidence >
                     detections[static_cast<std::size_t>(rhs)].confidence;
            });
  const int max_detections =
      cfg_.network.max_detections > 0
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
 * 该 decoder 使用当前模型约定：RGB HWC UINT8 输入、objectness raw logit、
 * raw color 0/1 映射到公开颜色枚举、raw color 2/3 直接拒绝、9-class tag
 * 映射到公开编号枚举，角点按模型原顺序 [0,3,2,1] 转成 TL/TR/BR/BL。
 */
template <CameraTypes::CameraInfo CameraInfoV>
std::optional<typename ArmorDetector<CameraInfoV>::NetworkDetection>
ArmorDetector<CameraInfoV>::DecodeModelDetection(
    const detail::NetworkInputMapping& mapping,
    const detail::ModelOutputView& output, int row) const
{
  if (!output.Valid() || row >= output.CandidateCount())
  {
    return std::nullopt;
  }

  const float objectness_logit = output.At(row, 8);
  const double logit_threshold =
      cfg_.network.logit_threshold > 0.0
          ? cfg_.network.logit_threshold
          : detail::default_logit_threshold;
  if (objectness_logit < static_cast<float>(logit_threshold))
  {
    return std::nullopt;
  }

  const int raw_color_id = detail::ArgMaxOutputRange(output, row, 9, 13);
  if (raw_color_id == 2 || raw_color_id == 3)
  {
    return std::nullopt;
  }
  const int raw_class_id = detail::ArgMaxOutputRange(output, row, 13, 22);

  std::array<cv::Point2f, 4> declared_points{};
  for (int point_index = 0; point_index < 4; ++point_index)
  {
    const float x = output.At(row, point_index * 2);
    const float y = output.At(row, point_index * 2 + 1);
    declared_points[static_cast<std::size_t>(point_index)] =
        mapping.MapToSource(x, y);
  }

  const auto points = detail::model_points_to_canonical(declared_points);
  if (cfg_.network.enable_quad_check &&
      !detail::IsUsableQuad(points, cfg_.network.min_quad_area_px))
  {
    return std::nullopt;
  }

  NetworkDetection detection;
  detection.color = detail::color_from_model_id(raw_color_id);
  detection.number = detail::number_from_model_class_id(raw_class_id);
  detection.confidence = detail::Sigmoid(objectness_logit);
  detection.points = points;
  const double bbox_expand =
      cfg_.network.bbox_expand >= 0.0 ? cfg_.network.bbox_expand
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
 * @tparam CameraInfoV 编译期相机参数。
 * @param detection 网络检测单元。
 * @return 初始化完成的内部候选。
 */
template <CameraTypes::CameraInfo CameraInfoV>
typename ArmorDetector<CameraInfoV>::CandidateArmor
ArmorDetector<CameraInfoV>::BuildCandidateArmor(
    const NetworkDetection& detection) const
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

/**
 * @brief 使用 MLP 对 detector 编号做置信度门控后的覆盖。
 *
 * @tparam CameraInfoV 编译期相机参数。
 * @param bgr_img 当前源图像。
 * @param armor 待 refine 候选。
 * @return 成功覆盖编号时返回 true。
 */
template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorDetector<CameraInfoV>::RefineNumberIfConfident(
    const cv::Mat& bgr_img, CandidateArmor& armor)
{
  if (!cfg_.number_refine.enabled || !number_refiner_.Ready())
  {
    return false;
  }
  if (armor.confidence <
      static_cast<float>(cfg_.number_refine.detector_min_confidence))
  {
    return false;
  }

  const auto prediction =
      number_refiner_.Predict(bgr_img, armor.points, armor.type);
  if (!prediction.valid ||
      prediction.confidence <
          static_cast<float>(cfg_.number_refine.classifier_min_confidence))
  {
    return false;
  }
  if (prediction.number == ArmorNumber::NEGATIVE ||
      !ArmorNumberIsKnown(prediction.number))
  {
    return false;
  }
  if (!RefinedNumberCompatibleWithGeometry(prediction.number, armor))
  {
    return false;
  }

  armor.number = prediction.number;
  ApplyNumberTypePrior(armor);
  return true;
}
