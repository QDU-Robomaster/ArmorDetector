#pragma once

// Runtime glue: configuration, frame conversion, worker thread, and publishing
// order. Keep image-processing stages in the dedicated implementation headers.
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

  network_.Configure(ARMOR_DETECTOR_MODEL_PATH);
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
