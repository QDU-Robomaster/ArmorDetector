#pragma once

// 仅供 ArmorDetector.hpp 在类声明之后包含。
template <CameraTypes::CameraInfo CameraInfoV>
ArmorDetector<CameraInfoV>::ArmorDetector(LibXR::HardwareContainer&,
                                          LibXR::ApplicationManager& app,
                                          Config cfg,
                                          Sync& sync)
    : sync_(sync)
{
  SetConfig(cfg);

  sync_frame_thread_.Create(this, SyncFrameThreadFun, "ArmorDetSync",
                            detail::sync_frame_thread_stack_size,
                            LibXR::Thread::Priority::HIGH);

  app.Register(*this);
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorDetector<CameraInfoV>::SetConfig(const Config& cfg)
{
  auto parse_env_flag = [](const char* name) -> bool
  {
    if (const char* env = std::getenv(name))
    {
      return env[0] != '\0' && env[0] != '0';
    }
    return false;
  };
  auto parse_env_u32 = [](const char* name, uint32_t fallback) -> uint32_t
  {
    if (const char* env = std::getenv(name))
    {
      if (env[0] == '\0')
      {
        return fallback;
      }
      const unsigned long parsed = std::strtoul(env, nullptr, 10);
      if (parsed > 0UL)
      {
        return static_cast<uint32_t>(parsed);
      }
    }
    return fallback;
  };

  cfg_ = cfg;
  counters_ = {};
  diagnostics_ = {};
  model_ready_ = false;

  diagnostics_.audit_every_frame =
      parse_env_flag("ARMOR_DETECTOR_AUDIT_EVERY_FRAME");
  diagnostics_.audit_zero_frames =
      parse_env_flag("ARMOR_DETECTOR_AUDIT_ZERO_FRAMES");
  diagnostics_.disable_traditional_refine =
      parse_env_flag("XR_ARMOR_DETECTOR_DISABLE_TRADITIONAL_REFINE");
  diagnostics_.center_letterbox =
      parse_env_flag("XR_ARMOR_DETECTOR_CENTER_LETTERBOX");
  diagnostics_.yolo_letterbox =
      parse_env_flag("XR_ARMOR_DETECTOR_YOLO_LETTERBOX");
  diagnostics_.dump_refine_fails =
      parse_env_flag("XR_ARMOR_DETECTOR_DUMP_REFINE_FAILS");
  if (diagnostics_.dump_refine_fails)
  {
    diagnostics_.dump_refine_fails_max =
        parse_env_u32("XR_ARMOR_DETECTOR_DUMP_REFINE_FAILS_MAX", 12U);
    if (const char* env = std::getenv("XR_ARMOR_DETECTOR_DUMP_REFINE_FAILS_DIR"))
    {
      if (env[0] != '\0')
      {
        diagnostics_.dump_refine_fails_dir = env;
      }
    }
    if (diagnostics_.dump_refine_fails_dir.empty())
    {
      diagnostics_.dump_refine_fails_dir = "/tmp/xr_armor_refine_fails";
    }
    std::error_code ec;
    std::filesystem::create_directories(diagnostics_.dump_refine_fails_dir, ec);
    XR_LOG_INFO(
        "ArmorDetector refine failure dump enabled: dir=%s max=%u create_ok=%d",
        diagnostics_.dump_refine_fails_dir.c_str(),
        diagnostics_.dump_refine_fails_max, ec ? 0 : 1);
  }

  try
  {
    auto model = ov_core_.read_model(ARMOR_DETECTOR_MODEL_PATH);
    ov::preprocess::PrePostProcessor post_processor(model);
    auto& input = post_processor.input();

    input.tensor()
        .set_element_type(ov::element::u8)
        .set_shape(
            ov::PartialShape{1, detail::yolo_input_size, detail::yolo_input_size, 3})
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::BGR);
    input.model().set_layout("NCHW");
    input.preprocess()
        .convert_element_type(ov::element::f32)
        .convert_color(ov::preprocess::ColorFormat::RGB)
        .scale(255.0);

    model = post_processor.build();
    compiled_model_ = ov_core_.compile_model(
        model, "CPU",
        ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    infer_request_ = compiled_model_.create_infer_request();
    model_ready_ = true;
  }
  catch (const std::exception& exception)
  {
    XR_LOG_ERROR("ArmorDetector failed to load YOLOv5 model: %s", exception.what());
    compiled_model_ = ov::CompiledModel();
    infer_request_ = ov::InferRequest();
  }
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorDetector<CameraInfoV>::ProcessImage(const cv::Mat& img_msg,
                                              const SyncedFrame& synced_frame)
{
  if (img_msg.empty())
  {
    return;
  }

  const auto* image_frame = synced_frame.GetImageFrame();
  if (image_frame == nullptr)
  {
    XR_LOG_ERROR("ArmorDetector received null synced image");
    return;
  }

  const uint64_t image_timestamp_us = image_frame->timestamp_us;
  armors_frame_packet_.source_frame.image_timestamp_us = image_timestamp_us;
  armors_frame_packet_.source_frame.image_frame = image_frame;
  armors_frame_packet_.source_frame.imu = &synced_frame.imu;
  armors_frame_packet_.detections = &armors_packet_;
  latest_timestamp_us_ = image_timestamp_us;

  const auto start_time = std::chrono::steady_clock::now();
  const cv::Mat& bgr_img = img_msg;

  const auto armors = Detect(bgr_img);
  const auto detector_finish = std::chrono::steady_clock::now();

  FillResultMessage(armors, bgr_img);
  const auto publish_finish = std::chrono::steady_clock::now();

  ++frame_index_;
  metrics_msg_.frame_index = frame_index_;
  metrics_msg_.image_timestamp_us = latest_timestamp_us_;
  metrics_msg_.decoded_count = counters_.decoded_count;
  metrics_msg_.nms_count = counters_.nms_count;
  metrics_msg_.semantic_kept_count = counters_.semantic_kept_count;
  metrics_msg_.armor_count = static_cast<uint32_t>(armors_packet_.results.size());
  metrics_msg_.pnp_success_count = counters_.pnp_success_count;
  metrics_msg_.refined_count = counters_.refined_count;
  metrics_msg_.discarded_count = counters_.discarded_count;
  metrics_msg_.semantic_discard_count = counters_.semantic_discard_count;
  metrics_msg_.type_discard_count = counters_.type_discard_count;
  metrics_msg_.max_objectness = counters_.max_objectness;
  metrics_msg_.detector_latency_ms =
      std::chrono::duration<double, std::milli>(detector_finish - start_time).count();
  metrics_msg_.publish_latency_ms =
      std::chrono::duration<double, std::milli>(publish_finish - detector_finish).count();

  DetectionMessage armors_frame_msg = &armors_frame_packet_;
  ArmorDetectionsMessage armors_msg = &armors_packet_;
  armors_frame_topic_.Publish(armors_frame_msg);
  armors_topic_.Publish(armors_msg);
  metrics_topic_.Publish(metrics_msg_);

  const uint32_t log_frame = detail::to_log_u32(metrics_msg_.frame_index);
  const uint32_t log_timestamp_ms =
      detail::to_log_u32(metrics_msg_.image_timestamp_us / 1000ULL);
  const uint32_t log_max_objectness_x1000 =
      detail::scaled_log_u32(metrics_msg_.max_objectness, 1000.0);
  const uint32_t log_detector_ms_x100 =
      detail::scaled_log_u32(metrics_msg_.detector_latency_ms, 100.0);
  const uint32_t log_publish_ms_x100 =
      detail::scaled_log_u32(metrics_msg_.publish_latency_ms, 100.0);

  if (diagnostics_.audit_every_frame)
  {
    XR_LOG_INFO(
        "ArmorDetector audit frame=%u ts_ms=%u decoded=%u nms=%u semantic_kept=%u armors=%u",
        log_frame, log_timestamp_ms,
        metrics_msg_.decoded_count, metrics_msg_.nms_count,
        metrics_msg_.semantic_kept_count, metrics_msg_.armor_count);
    XR_LOG_INFO(
        "ArmorDetector audit pnp=%u refined=%u semantic_discard=%u type_discard=%u discarded=%u max_obj_x1000=%u",
        metrics_msg_.pnp_success_count, metrics_msg_.refined_count,
        metrics_msg_.semantic_discard_count, metrics_msg_.type_discard_count,
        metrics_msg_.discarded_count, log_max_objectness_x1000);
    XR_LOG_INFO("ArmorDetector audit detector_ms_x100=%u publish_ms_x100=%u",
                log_detector_ms_x100, log_publish_ms_x100);
  }
  else if (diagnostics_.audit_zero_frames && metrics_msg_.armor_count == 0U)
  {
    XR_LOG_WARN(
        "ArmorDetector zero frame=%u ts_ms=%u decoded=%u nms=%u semantic_kept=%u pnp=%u",
        log_frame, log_timestamp_ms,
        metrics_msg_.decoded_count, metrics_msg_.nms_count,
        metrics_msg_.semantic_kept_count, metrics_msg_.pnp_success_count);
    XR_LOG_WARN(
        "ArmorDetector zero refined=%u semantic_discard=%u type_discard=%u discarded=%u max_obj_x1000=%u",
        metrics_msg_.refined_count, metrics_msg_.semantic_discard_count,
        metrics_msg_.type_discard_count, metrics_msg_.discarded_count,
        log_max_objectness_x1000);
    XR_LOG_WARN("ArmorDetector zero detector_ms_x100=%u publish_ms_x100=%u",
                log_detector_ms_x100, log_publish_ms_x100);
  }
  else if ((frame_index_ % detail::metrics_log_period) == 0U)
  {
    XR_LOG_INFO(
        "ArmorDetector frame=%u armors=%u decoded=%u nms=%u semantic_kept=%u refined=%u",
        log_frame,
        metrics_msg_.armor_count, metrics_msg_.decoded_count,
        metrics_msg_.nms_count, metrics_msg_.semantic_kept_count,
        metrics_msg_.refined_count);
    XR_LOG_INFO(
        "ArmorDetector refine_attempt=%u fail_bbox=%u fail_roi=%u fail_l0=%u fail_l1=%u fail_pair=%u",
        counters_.refine_attempt_count,
        counters_.refine_fail_bbox_oob_count,
        counters_.refine_fail_roi_empty_count,
        counters_.refine_fail_lightbar_zero_count,
        counters_.refine_fail_lightbar_one_count,
        counters_.refine_fail_pair_distance_count);
    XR_LOG_INFO(
        "ArmorDetector semantic_discard=%u type_discard=%u discarded=%u max_obj_x1000=%u detector_ms_x100=%u publish_ms_x100=%u",
        metrics_msg_.semantic_discard_count,
        metrics_msg_.type_discard_count, metrics_msg_.discarded_count,
        log_max_objectness_x1000,
        log_detector_ms_x100, log_publish_ms_x100);
  }
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorDetector<CameraInfoV>::ProcessSyncedFrame(const SyncedFrame& synced_frame)
{
  const auto* image_frame = synced_frame.GetImageFrame();
  if (image_frame == nullptr)
  {
    XR_LOG_ERROR("ArmorDetector received null synced image");
    return;
  }

  const int cv_type = detail::CvTypeFromEncoding(camera_info.encoding);
  if (cv_type < 0)
  {
    XR_LOG_WARN("ArmorDetector sync frame encoding unsupported: %u",
                static_cast<unsigned>(camera_info.encoding));
    return;
  }

  cv::Mat img(static_cast<int>(camera_info.height), static_cast<int>(camera_info.width),
              cv_type, const_cast<uint8_t*>(image_frame->data.data()),
              static_cast<size_t>(camera_info.step));
  const cv::Mat bgr_img =
      detail::ConvertToBgrWithEncoding(img, camera_info.encoding);
  if (bgr_img.empty())
  {
    return;
  }
  ProcessImage(bgr_img, synced_frame);
}

template <CameraTypes::CameraInfo CameraInfoV>
void ArmorDetector<CameraInfoV>::SyncFrameThreadFun(ArmorDetector<CameraInfoV>* self)
{
  XR_LOG_INFO("ArmorDetector sync worker starting: image=%s imu=%s",
              self->sync_.ImageTopicName(), self->sync_.ImuTopicName());

  typename Sync::Subscriber subscriber(self->sync_);
  if (!subscriber.Valid())
  {
    XR_LOG_ERROR("ArmorDetector failed to attach sync image topic: %s",
                 self->sync_.ImageTopicName());
    return;
  }

  XR_LOG_PASS("ArmorDetector attached sync stream: image=%s imu=%s",
              self->sync_.ImageTopicName(), self->sync_.ImuTopicName());

  SyncedFrame synced_frame;
  while (true)
  {
    const auto wait_ans =
        subscriber.Wait(synced_frame, detail::sync_frame_wait_timeout_ms);
    if (wait_ans == LibXR::ErrorCode::TIMEOUT)
    {
      continue;
    }
    if (wait_ans != LibXR::ErrorCode::OK)
    {
      XR_LOG_ERROR("ArmorDetector sync wait failed (err=%d), worker stopped.",
                   static_cast<int>(wait_ans));
      return;
    }

    self->ProcessSyncedFrame(synced_frame);
  }
}

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

  if (!model_ready_)
  {
    XR_LOG_ERROR("ArmorDetector YOLOv5 model is not ready");
    return {};
  }

  cv::Mat detector_img = raw_img;
  cv::Point2f offset(0.0F, 0.0F);
  cv::Rect clipped_roi(0, 0, raw_img.cols, raw_img.rows);

  // 1. 根据配置裁剪出真正送入网络的图像区域。
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

  const double height_scale =
      static_cast<double>(detail::yolo_input_size) / std::max(1, detector_img.rows);
  const double width_scale =
      static_cast<double>(detail::yolo_input_size) / std::max(1, detector_img.cols);
  const double scale = std::min(height_scale, width_scale);
  const int resized_height =
      std::max(1, static_cast<int>(std::round(detector_img.rows * scale)));
  const int resized_width =
      std::max(1, static_cast<int>(std::round(detector_img.cols * scale)));
  const bool centered_letterbox =
      diagnostics_.center_letterbox || diagnostics_.yolo_letterbox;
  const int input_offset_x =
      centered_letterbox ? std::max(0, (detail::yolo_input_size - resized_width) / 2) : 0;
  const int input_offset_y =
      centered_letterbox ? std::max(0, (detail::yolo_input_size - resized_height) / 2) : 0;
  const cv::Point2f input_offset(static_cast<float>(input_offset_x),
                                 static_cast<float>(input_offset_y));
  const cv::Scalar letterbox_fill =
      diagnostics_.yolo_letterbox ? cv::Scalar(114, 114, 114)
                                  : cv::Scalar(0, 0, 0);

  // 3. letterbox 到固定输入尺寸，并复用持久化 infer request。
  cv::Mat input(detail::yolo_input_size, detail::yolo_input_size, CV_8UC3,
                letterbox_fill);
  cv::resize(detector_img,
             input(cv::Rect(input_offset_x, input_offset_y, resized_width, resized_height)),
             cv::Size(resized_width, resized_height));

  ov::Tensor input_tensor(
      ov::element::u8,
      ov::Shape{1, static_cast<size_t>(detail::yolo_input_size),
                static_cast<size_t>(detail::yolo_input_size), 3},
      input.data);
  infer_request_.set_input_tensor(input_tensor);
  infer_request_.infer();

  auto output_tensor = infer_request_.get_output_tensor();
  const auto output_shape = output_tensor.get_shape();
  cv::Mat output(static_cast<int>(output_shape[1]), static_cast<int>(output_shape[2]),
                 CV_32F, output_tensor.data<float>());
  auto armors = DecodeOutput(scale, input_offset, output, detector_img);
  if (armors.empty())
  {
    return armors;
  }

  // 4. 如果网络只看了 ROI，这里再把检测结果平移回整幅原图坐标系。
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
