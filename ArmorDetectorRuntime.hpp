#pragma once

#include <chrono>

/**
 * @file ArmorDetectorRuntime.hpp
 * @brief ArmorDetector 配置、同步帧回调和帧级运行时实现。
 */

/**
 * @brief 构造 detector，加载模型并启动推理流水线。
 *
 * detector 当前不直接访问 HardwareContainer；图像和 IMU 由 CameraFrameSync
 * 输入。
 *
 * @tparam FrameLayoutV 编译期帧布局。
 * @param app 应用管理器，用于注册本模块。
 * @param cfg detector 初始配置。
 * @param sync 同步帧来源。
 */
template <CameraTypes::FrameLayout FrameLayoutV>
ArmorDetector<FrameLayoutV>::ArmorDetector(LibXR::HardwareContainer&,
                                           LibXR::ApplicationManager& app, Config cfg,
                                           Sync& sync)
    : sync_(sync), pnp_solver_(sync.Calibration())
{
  SetConfig(cfg);

  if (const char* mode = std::getenv("ARMOR_DETECTOR_INFERENCE_MODE");
      mode != nullptr && mode[0] != '\0')
  {
    if (std::string(mode) == "sync")
    {
      async_inference_enabled_ = false;
    }
    else if (std::string(mode) != "async")
    {
      XR_LOG_ERROR("ArmorDetector invalid inference mode '%s'", mode);
      return;
    }
  }

  if (cfg_.referee_auto_detect_color)
  {
    const char* domain_name = cfg_.referee_domain;
    if (domain_name == nullptr || domain_name[0] == '\0')
    {
      domain_name = "host";
    }
    const char* topic_name = cfg_.referee_topic;
    if (topic_name == nullptr || topic_name[0] == '\0')
    {
      topic_name = "robot_game_ref";
    }

    referee_domain_ = LibXR::Topic::Domain(domain_name);
    referee_topic_ =
        LibXR::Topic(LibXR::Topic::WaitTopic(topic_name, UINT32_MAX, &referee_domain_));
    referee_callback_ = LibXR::Topic::Callback::Create(
        [](bool, ArmorDetector* self, const LibXR::ConstRawData& data)
        { self->OnRefereeRobotGame(data); }, this);
    referee_topic_.RegisterCallback(referee_callback_);
  }

  for (auto& buffers : hailo_buffer_pool_)
  {
    if (!network_.InitRawOutputSlot(buffers.raw_output))
    {
      XR_LOG_ERROR("ArmorDetector failed to initialize a pipeline raw-output slot");
      return;
    }
    if (!network_.InitRawInputView(buffers.raw_output, buffers.network_input))
    {
      XR_LOG_ERROR("ArmorDetector failed to initialize a pipeline raw-input view");
      return;
    }
  }
  for (std::size_t index = 0; index < infer_slots_.size(); ++index)
  {
    infer_slots_[index].hailo_buffer_id = index;
  }
  for (std::size_t index = 0; index < post_slots_.size(); ++index)
  {
    post_slots_[index].hailo_buffer_id = infer_slots_.size() + index;
  }
  if (async_inference_enabled_ && network_.AsyncQueueSize() < 2U)
  {
    XR_LOG_ERROR("ArmorDetector Hailo async queue is too small: %zu",
                 network_.AsyncQueueSize());
    return;
  }
  XR_LOG_PASS(
      "ArmorDetector inference mode=%s infer_slots=%zu post_slots=%zu "
      "async_depth=%u",
      async_inference_enabled_ ? "async" : "sync", infer_slots_.size(),
      post_slots_.size(), async_inference_enabled_ ? async_inflight_limit : 1U);

  inference_thread_ = std::thread(InferenceThreadFun, this);
  inference_thread_.detach();
  output_fusion_thread_ = std::thread(OutputFusionThreadFun, this);
  output_fusion_thread_.detach();
  workers_started_.store(true, std::memory_order_release);

  synced_frame_topic_ = LibXR::Topic(
      LibXR::Topic::FindOrCreate<SyncedFrameTopicPayload>(sync_.SyncedFrameTopicName()));
  synced_frame_callback_ = LibXR::Topic::Callback::Create(OnSyncedFrameStatic, this);
  synced_frame_topic_.RegisterCallback(synced_frame_callback_);

  app.Register(*this);
}

/**
 * @brief 更新配置并重新加载模型。
 *
 * @tparam FrameLayoutV 编译期帧布局。
 * @param cfg 新 detector 配置。
 */
template <CameraTypes::FrameLayout FrameLayoutV>
void ArmorDetector<FrameLayoutV>::SetConfig(const Config& cfg)
{
  if (workers_started_.load(std::memory_order_acquire))
  {
    XR_LOG_ERROR("ArmorDetector runtime SetConfig rejected after pipeline start");
    return;
  }
  cfg_ = cfg;
  counters_ = {};
  preview_.Stop();
  preview_.Start(cfg_.preview);

  const auto* resolved_model = infer::resolve_detector_model(cfg_.network.model);
  if (resolved_model == nullptr)
  {
    XR_LOG_ERROR("ArmorDetector unknown model enum=%u, fallback to default %s",
                 static_cast<unsigned>(cfg_.network.model),
                 infer::detector_model_name(infer::default_detector_model));
  }
  const auto& resolved = infer::resolve_detector_model_or_default(cfg_.network.model);

  XR_LOG_INFO("ArmorDetector model=%s line=%s hef_path=%s", resolved.canonical_name,
              infer::model_line_name(resolved.line), resolved.hailort_hef_path);
  XR_LOG_INFO(
      "ArmorDetector decode logit=%.3f confidence=%.3f nms=%.3f "
      "bbox_expand=%.3f "
      "max_det=%d",
      cfg_.network.logit_threshold, cfg_.network.min_confidence,
      cfg_.network.nms_threshold, cfg_.network.bbox_expand, cfg_.network.max_detections);
  network_.Configure(resolved);
}

template <CameraTypes::FrameLayout FrameLayoutV>
bool ArmorDetector<FrameLayoutV>::PipelineDrained() const noexcept
{
  std::lock_guard<std::mutex> lock(pipeline_mutex_);
  if (inference_worker_active_.load(std::memory_order_acquire) ||
      output_worker_active_.load(std::memory_order_acquire) ||
      !inference_queue_.Empty() || !output_queue_.Empty() ||
      !async_completions_.Empty() || async_inflight_ != 0U)
  {
    return false;
  }
  for (const auto& slot : infer_slots_)
  {
    if (slot.state.load(std::memory_order_acquire) !=
        armor_detector_pipeline::InferSlotState::FREE)
    {
      return false;
    }
  }
  for (const auto& slot : post_slots_)
  {
    if (slot.state.load(std::memory_order_acquire) !=
        armor_detector_pipeline::PostSlotState::FREE)
    {
      return false;
    }
  }
  const uint64_t admitted = pipeline_admitted_count_.load(std::memory_order_acquire);
  const uint64_t completed = pipeline_completed_count_.load(std::memory_order_acquire);
  const bool drained = admitted == completed;
  return drained;
}

template <CameraTypes::FrameLayout FrameLayoutV>
void ArmorDetector<FrameLayoutV>::OnMonitor()
{
  const auto preprocess = preprocess_duration_.GetSummary();
  const auto inference_worker = inference_worker_duration_.GetSummary();
  const auto postprocess = postprocess_duration_.GetSummary();
  const auto result = result_duration_.GetSummary();
  XR_LOG_INFO(
      "ArmorDetector duration count/average/minimum/maximum_us "
      "preprocess=%llu/%llu/%llu/%llu inference_worker=%llu/%llu/%llu/%llu",
      static_cast<unsigned long long>(preprocess.sample_count),
      static_cast<unsigned long long>(preprocess.average_us),
      static_cast<unsigned long long>(preprocess.minimum_us),
      static_cast<unsigned long long>(preprocess.maximum_us),
      static_cast<unsigned long long>(inference_worker.sample_count),
      static_cast<unsigned long long>(inference_worker.average_us),
      static_cast<unsigned long long>(inference_worker.minimum_us),
      static_cast<unsigned long long>(inference_worker.maximum_us));
  XR_LOG_INFO(
      "ArmorDetector duration count/average/minimum/maximum_us "
      "postprocess=%llu/%llu/%llu/%llu result=%llu/%llu/%llu/%llu",
      static_cast<unsigned long long>(postprocess.sample_count),
      static_cast<unsigned long long>(postprocess.average_us),
      static_cast<unsigned long long>(postprocess.minimum_us),
      static_cast<unsigned long long>(postprocess.maximum_us),
      static_cast<unsigned long long>(result.sample_count),
      static_cast<unsigned long long>(result.average_us),
      static_cast<unsigned long long>(result.minimum_us),
      static_cast<unsigned long long>(result.maximum_us));
}

template <CameraTypes::FrameLayout FrameLayoutV>
void ArmorDetector<FrameLayoutV>::OnSyncedFrameStatic(bool,
                                                      ArmorDetector<FrameLayoutV>* self,
                                                      SyncedFrameTopicPayload borrowed)
{
  if (borrowed == nullptr || !borrowed->Valid())
  {
    return;
  }
  self->AdmitSyncedFrame(*borrowed);
}

template <CameraTypes::FrameLayout FrameLayoutV>
bool ArmorDetector<FrameLayoutV>::AdmitSyncedFrame(const SyncedFrame& synced_frame)
{
  if (!synced_frame.Valid())
  {
    pipeline_prepare_drop_count_.fetch_add(1U, std::memory_order_relaxed);
    return false;
  }

  uint8_t slot_id = 0U;
  uint64_t generation = 0U;
  {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    bool found = false;
    uint8_t candidate_slot_id = 0U;
    for (auto& slot : infer_slots_)
    {
      const uint8_t current_slot_id = candidate_slot_id++;
      if (slot.state.load(std::memory_order_relaxed) !=
          armor_detector_pipeline::InferSlotState::FREE)
      {
        continue;
      }

      slot_id = current_slot_id;
      generation = armor_detector_pipeline::NextGeneration(
          slot.generation.load(std::memory_order_relaxed));
      slot.generation.store(generation, std::memory_order_relaxed);
      slot.context = {};
      auto& context = slot.context;
      context.synced_frame = synced_frame;
      const ImageFrame* image = context.synced_frame.GetImageFrame();
      ASSERT(image != nullptr);
      context.frame_timestamp_us =
          static_cast<uint64_t>(context.synced_frame.imu.timestamp_us);
      context.camera_timestamp_us = static_cast<uint64_t>(image->timestamp_us);
      slot.state.store(armor_detector_pipeline::InferSlotState::PREPARING,
                       std::memory_order_release);
      found = true;
      break;
    }

    if (!found)
    {
      pipeline_no_free_count_.fetch_add(1U, std::memory_order_relaxed);
      const uint64_t drops =
          pipeline_prepare_drop_count_.fetch_add(1U, std::memory_order_relaxed) + 1U;
      if (drops <= 5U || (drops % 100U) == 0U)
      {
        XR_LOG_WARN("ArmorDetector pipeline full: prepare_drop=%llu",
                    static_cast<unsigned long long>(drops));
      }
      return false;
    }
  }

  bool prepared = false;
  auto& slot = infer_slots_[slot_id];
  auto& context = slot.context;
  const ImageFrame* image = context.synced_frame.GetImageFrame();
  ASSERT(image != nullptr);
  ASSERT(slot.hailo_buffer_id < hailo_buffer_pool_.size());
  auto& hailo_buffers = hailo_buffer_pool_[slot.hailo_buffer_id];
  const auto preprocess_begin = std::chrono::steady_clock::now();
  const bool input_view_valid_before =
      network_.IsRawInputView(hailo_buffers.raw_output, hailo_buffers.network_input);
  cv::Mat working_network_input = hailo_buffers.network_input;
  {
    auto preprocess_measurement = preprocess_duration_.Measure();
    try
    {
      if (input_view_valid_before)
      {
        const auto& geometry = image->geometry;
        const int cv_type = detail::CvTypeFromEncoding(frame_layout.encoding);
        if (cv_type >= 0)
        {
          cv::Mat source(static_cast<int>(geometry.height),
                         static_cast<int>(geometry.width), cv_type,
                         const_cast<uint8_t*>(image->data.data()),
                         static_cast<std::size_t>(geometry.step));
          const cv::Mat bgr =
              detail::ConvertToBgrWithEncoding(source, frame_layout.encoding);
          prepared = BuildNetworkInput(bgr, context.input_mapping,
                                       slot.preprocess_scratch, working_network_input);
        }
      }
    }
    catch (const cv::Exception& exception)
    {
      XR_LOG_ERROR("ArmorDetector preprocess exception: %s", exception.what());
    }
    catch (const std::exception& exception)
    {
      XR_LOG_ERROR("ArmorDetector preprocess exception: %s", exception.what());
    }
    catch (...)
    {
      XR_LOG_ERROR("ArmorDetector preprocess exception: unknown");
    }
  }
  const auto preprocess_end = std::chrono::steady_clock::now();
  const bool input_view_valid_after =
      network_.IsRawInputView(hailo_buffers.raw_output, hailo_buffers.network_input) &&
      network_.IsRawInputView(hailo_buffers.raw_output, working_network_input);
  if (!input_view_valid_before || !input_view_valid_after)
  {
    XR_LOG_ERROR("ArmorDetector pipeline raw-input view identity changed");
    prepared = false;
  }

  std::lock_guard<std::mutex> lock(pipeline_mutex_);
  if (slot.generation.load(std::memory_order_relaxed) != generation ||
      slot.state.load(std::memory_order_acquire) !=
          armor_detector_pipeline::InferSlotState::PREPARING)
  {
    pipeline_prepare_drop_count_.fetch_add(1U, std::memory_order_relaxed);
    return false;
  }
  if (!prepared)
  {
    pipeline_prepare_drop_count_.fetch_add(1U, std::memory_order_relaxed);
    ReleaseInferSlotLocked({slot_id, generation});
    return false;
  }

  context.preprocess_latency_ms =
      std::chrono::duration<double, std::milli>(preprocess_end - preprocess_begin)
          .count();
  context.admission_counted = true;
  pipeline_admitted_count_.fetch_add(1U, std::memory_order_relaxed);
  slot.state.store(armor_detector_pipeline::InferSlotState::INFER_QUEUED,
                   std::memory_order_release);
  const bool infer_queued = inference_queue_.TryPush({slot_id, generation});
  ASSERT(infer_queued);
  (void)infer_queued;
  pipeline_cv_.notify_all();
  return true;
}

/**
 * @brief 裁判系统摘要包回调。
 *
 * 当前只依赖 RobotGameRefereePack 第一个字段 RobotStatus 的首字节 robot_id。
 *
 * @tparam FrameLayoutV 编译期帧布局。
 * @param data 裁判系统摘要包原始数据。
 */
template <CameraTypes::FrameLayout FrameLayoutV>
void ArmorDetector<FrameLayoutV>::OnRefereeRobotGame(const LibXR::ConstRawData& data)
{
  if (data.addr_ == nullptr || data.size_ < sizeof(uint8_t))
  {
    return;
  }

  const auto robot_id = *reinterpret_cast<const uint8_t*>(data.addr_);
  const int target_color = TargetColorFromRobotId(robot_id);
  if (target_color < 0)
  {
    return;
  }

  referee_target_color_.store(target_color, std::memory_order_relaxed);
}

/**
 * @brief 当前有效目标颜色。
 *
 * 自动颜色启用且已收到有效裁判系统 robot_id 时使用动态敌方颜色，否则使用
 * cfg.detect_color。
 *
 * @tparam FrameLayoutV 编译期帧布局。
 * @return 当前用于语义过滤的目标颜色。
 */
template <CameraTypes::FrameLayout FrameLayoutV>
ArmorColor ArmorDetector<FrameLayoutV>::CurrentTargetColor() const
{
  if (cfg_.referee_auto_detect_color)
  {
    const int target_color = referee_target_color_.load(std::memory_order_relaxed);
    if (target_color == 0 || target_color == 1)
    {
      return detail::detect_color_from_config(target_color);
    }
  }

  return detail::detect_color_from_config(cfg_.detect_color);
}

/**
 * @brief 由机器人 ID 推导敌方颜色。
 *
 * RoboMaster ID 1~99 视为红方，本机红方时目标为蓝色；ID 101~199 视为蓝方，
 * 本机蓝方时目标为红色。
 *
 * @tparam FrameLayoutV 编译期帧布局。
 * @param robot_id 本机机器人 ID。
 * @return 0=红，1=蓝，-1=不确定。
 */
template <CameraTypes::FrameLayout FrameLayoutV>
int ArmorDetector<FrameLayoutV>::TargetColorFromRobotId(uint8_t robot_id)
{
  if (robot_id >= 1U && robot_id < 100U)
  {
    return 1;
  }
  if (robot_id >= 101U && robot_id < 200U)
  {
    return 0;
  }
  return -1;
}

/**
 * @brief 处理已经转换为 BGR Mat 的同步帧。
 *
 * 该函数把共享图像所有权、同步 IMU 和当前检测结果一起发布给下游。
 *
 * @tparam FrameLayoutV 编译期帧布局。
 * @param img_msg BGR 图像。
 * @param synced_frame 原始同步帧。
 */
template <CameraTypes::FrameLayout FrameLayoutV>
void ArmorDetector<FrameLayoutV>::ProcessImage(
    const cv::Mat& img_msg, SyncedFrame& synced_frame,
    std::vector<CandidateArmor>&& armors, uint64_t frame_timestamp_us,
    uint64_t camera_timestamp_us,
    const detail::ArmorDetectorNetwork::HailoRawTimingSnapshot& infer_timing,
    const detail::ArmorDetectorNetwork::HailoDecodeTimingSnapshot& decode_timing,
    double preprocess_latency_ms, double postprocess_latency_ms)
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

  detected_frame_.sequence = synced_frame.sequence;
  detected_frame_.image = std::move(synced_frame.image);
  struct ImageResetGuard
  {
    SharedFrame& image;
    ~ImageResetGuard() { image.Reset(); }
  } image_reset_guard{detected_frame_.image};
  detected_frame_.imu = synced_frame.imu;
  latest_frame_timestamp_us_ = frame_timestamp_us;
  latest_camera_timestamp_us_ = camera_timestamp_us;

  const cv::Mat& bgr_img = img_msg;
  const uint64_t next_frame_index = frame_index_ + 1U;
  const bool trace_frame = (next_frame_index <= 5U) || (next_frame_index % 30U == 0U);

  if (trace_frame)
  {
    XR_LOG_INFO("ArmorDetector trace frame=%llu step=detect_begin frame_ts=%llu",
                static_cast<unsigned long long>(next_frame_index),
                static_cast<unsigned long long>(frame_timestamp_us));
  }

  if (trace_frame)
  {
    XR_LOG_INFO(
        "ArmorDetector trace frame=%llu step=detect_end armors=%zu decoded=%u "
        "semantic_kept=%u",
        static_cast<unsigned long long>(next_frame_index), armors.size(),
        counters_.decoded_count, counters_.semantic_kept_count);
  }

  if (trace_frame)
  {
    XR_LOG_INFO("ArmorDetector trace frame=%llu step=fill_begin",
                static_cast<unsigned long long>(next_frame_index));
  }
  const auto result_begin = std::chrono::steady_clock::now();
  {
    auto result_measurement = result_duration_.Measure();
    FillResultMessage(armors, bgr_img, image_frame->geometry, detected_frame_.detections);
    detected_frame_.detections.erase(
        std::remove_if(
            detected_frame_.detections.begin(), detected_frame_.detections.end(),
            [](const ArmorDetectorResult& armor)
            { return armor.number == ArmorNumber::OUTPOST; }),
        detected_frame_.detections.end());
  }
  const auto result_finish = std::chrono::steady_clock::now();
  if (trace_frame)
  {
    XR_LOG_INFO("ArmorDetector trace frame=%llu step=fill_end results=%zu pnp=%u",
                static_cast<unsigned long long>(next_frame_index),
                detected_frame_.detections.size(), counters_.pnp_success_count);
  }

  ++frame_index_;
  metrics_msg_.frame_index = frame_index_;
  metrics_msg_.frame_timestamp_us = latest_frame_timestamp_us_;
  metrics_msg_.camera_timestamp_us = latest_camera_timestamp_us_;
  metrics_msg_.decoded_count = counters_.decoded_count;
  metrics_msg_.overlap_kept_count = counters_.overlap_kept_count;
  metrics_msg_.semantic_kept_count = counters_.semantic_kept_count;
  metrics_msg_.armor_count = static_cast<uint32_t>(detected_frame_.detections.size());
  metrics_msg_.pnp_success_count = counters_.pnp_success_count;
  metrics_msg_.discarded_count = counters_.discarded_count;
  metrics_msg_.semantic_discard_count = counters_.semantic_discard_count;
  metrics_msg_.type_discard_count = counters_.type_discard_count;
  metrics_msg_.max_objectness = counters_.max_objectness;
  metrics_msg_.preprocess_latency_ms = preprocess_latency_ms;
  metrics_msg_.infer_latency_ms = infer_timing.valid ? infer_timing.infer_ms : 0.0;
  metrics_msg_.postprocess_latency_ms = postprocess_latency_ms;
  metrics_msg_.hailo_infer_latency_ms = infer_timing.valid ? infer_timing.infer_ms : 0.0;
  metrics_msg_.hailo_tail_latency_ms = decode_timing.valid ? decode_timing.tail_ms : 0.0;
  metrics_msg_.detector_latency_ms =
      metrics_msg_.preprocess_latency_ms + metrics_msg_.hailo_infer_latency_ms +
      metrics_msg_.hailo_tail_latency_ms + metrics_msg_.postprocess_latency_ms;
  metrics_msg_.result_latency_ms =
      std::chrono::duration<double, std::milli>(result_finish - result_begin).count();

  DetectionMessage armors_frame_msg = &detected_frame_;
  const LibXR::MicrosecondTimestamp publish_timestamp(frame_timestamp_us);
  if (trace_frame)
  {
    XR_LOG_INFO("ArmorDetector trace frame=%llu step=publish_begin",
                static_cast<unsigned long long>(frame_index_));
  }
  armors_frame_topic_.Publish(armors_frame_msg, publish_timestamp);
  if (trace_frame)
  {
    XR_LOG_INFO("ArmorDetector trace frame=%llu step=publish_end",
                static_cast<unsigned long long>(frame_index_));
  }
  if (trace_frame)
  {
    XR_LOG_INFO("ArmorDetector trace frame=%llu step=preview_begin",
                static_cast<unsigned long long>(frame_index_));
  }
  SubmitPreview(bgr_img, armors);
  if (trace_frame)
  {
    XR_LOG_INFO("ArmorDetector trace frame=%llu step=preview_end",
                static_cast<unsigned long long>(frame_index_));
  }

  if ((frame_index_ % detail::metrics_log_period) == 0U)
  {
    XR_LOG_INFO(
        "ArmorDetector frame=%llu armors=%u decoded=%u overlap_kept=%u "
        "semantic_kept=%u "
        "pnp=%u",
        static_cast<unsigned long long>(metrics_msg_.frame_index),
        metrics_msg_.armor_count, metrics_msg_.decoded_count,
        metrics_msg_.overlap_kept_count, metrics_msg_.semantic_kept_count,
        metrics_msg_.pnp_success_count);
    XR_LOG_INFO(
        "ArmorDetector semantic_discard=%u type_discard=%u "
        "discarded=%u max_obj=%.3f "
        "detector_ms=%.3f result_ms=%.3f",
        metrics_msg_.semantic_discard_count, metrics_msg_.type_discard_count,
        metrics_msg_.discarded_count, metrics_msg_.max_objectness,
        metrics_msg_.detector_latency_ms, metrics_msg_.result_latency_ms);
    XR_LOG_INFO(
        "ArmorDetector split_ms preprocess=%.3f infer_call=%.3f "
        "postprocess=%.3f "
        "hailo_infer=%.3f hailo_tail=%.3f",
        metrics_msg_.preprocess_latency_ms, metrics_msg_.infer_latency_ms,
        metrics_msg_.postprocess_latency_ms, metrics_msg_.hailo_infer_latency_ms,
        metrics_msg_.hailo_tail_latency_ms);
    XR_LOG_INFO(
        "ArmorDetector pipeline admitted=%llu completed=%llu prepare_drop=%llu "
        "no_free=%llu infer_fail=%llu post_fail=%llu infer_q=%zu output_q=%zu",
        static_cast<unsigned long long>(
            pipeline_admitted_count_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(
            pipeline_completed_count_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(
            pipeline_prepare_drop_count_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(
            pipeline_no_free_count_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(
            pipeline_infer_fail_count_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(
            pipeline_post_fail_count_.load(std::memory_order_relaxed)),
        inference_queue_.Size(), output_queue_.Size());
  }
}

/**
 * @brief 提交 detector 预览帧。
 *
 * Submit 会先深拷贝图像；回调捕获的是检测结果快照，避免预览线程读到下一帧复用的
 * detected_frame_ / metrics_msg_。
 *
 * @tparam FrameLayoutV 编译期帧布局。
 * @param bgr_img 当前 BGR 图像。
 */
template <CameraTypes::FrameLayout FrameLayoutV>
void ArmorDetector<FrameLayoutV>::SubmitPreview(const cv::Mat& bgr_img,
                                                const std::vector<CandidateArmor>& armors)
{
  if (!preview_.Running())
  {
    return;
  }

  const std::vector<CandidateArmor> local_armors = armors;
  const FrameMetrics metrics = metrics_msg_;
  preview_.Submit(
      bgr_img,
      [local_armors, metrics](cv::Mat& canvas)
      {
        const auto preview_color_for = [](ArmorColor color) -> cv::Scalar
        {
          if (color == ArmorColor::RED)
          {
            return cv::Scalar(0, 0, 255);
          }
          if (color == ArmorColor::BLUE)
          {
            return cv::Scalar(255, 80, 0);
          }
          return cv::Scalar(0, 220, 255);
        };
        const auto number_name_for = [](ArmorNumber number) -> std::string
        {
          const auto index = static_cast<std::size_t>(number);
          return index < ARMOR_NUMBER_NAMES.size()
                     ? std::string(ARMOR_NUMBER_NAMES[index])
                     : std::string("unknown");
        };

        for (const auto& armor : local_armors)
        {
          const cv::Scalar color = preview_color_for(armor.color);
          for (std::size_t i = 0; i < armor.points.size(); ++i)
          {
            cv::line(canvas, armor.points[i],
                     armor.points[(i + 1U) % armor.points.size()], color, 2, cv::LINE_AA);
            cv::circle(canvas, armor.points[i], 3, color, -1, cv::LINE_AA);
          }
          cv::circle(canvas, armor.center, 4, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);

          const std::string label =
              number_name_for(armor.number) + " " +
              std::to_string(static_cast<int>(armor.confidence * 100.0F)) + "%";
          const cv::Point text_origin(std::max(0, static_cast<int>(armor.box.x)),
                                      std::max(18, static_cast<int>(armor.box.y) - 6));
          cv::putText(canvas, label, text_origin, cv::FONT_HERSHEY_SIMPLEX, 0.55, color,
                      2, cv::LINE_AA);
        }

        const std::string header =
            "detector frame=" + std::to_string(metrics.frame_index) +
            " armors=" + std::to_string(metrics.armor_count) +
            " pnp=" + std::to_string(metrics.pnp_success_count);
        cv::putText(canvas, header, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.75,
                    cv::Scalar(40, 240, 40), 2, cv::LINE_AA);
      });
}

template <CameraTypes::FrameLayout FrameLayoutV>
void ArmorDetector<FrameLayoutV>::RunInference(armor_detector_pipeline::WorkItem item)
{
  auto inference_worker_measurement = inference_worker_duration_.Measure();
  const uint8_t slot_id = item.slot_id;
  if (slot_id >= infer_slots_.size())
  {
    pipeline_infer_fail_count_.fetch_add(1U, std::memory_order_relaxed);
    return;
  }
  auto& slot = infer_slots_[slot_id];
  if (slot.generation.load(std::memory_order_acquire) != item.generation)
  {
    return;
  }
  auto expected = armor_detector_pipeline::InferSlotState::INFER_QUEUED;
  if (!slot.state.compare_exchange_strong(
          expected, armor_detector_pipeline::InferSlotState::INFER_RUNNING,
          std::memory_order_acq_rel, std::memory_order_acquire))
  {
    pipeline_infer_fail_count_.fetch_add(1U, std::memory_order_relaxed);
    return;
  }

  auto& context = slot.context;
  ASSERT(slot.hailo_buffer_id < hailo_buffer_pool_.size());
  auto& hailo_buffers = hailo_buffer_pool_[slot.hailo_buffer_id];
  bool ok = false;
  detail::ArmorDetectorNetwork::HailoRawTimingSnapshot sync_timing{};
  if (!network_.IsRawInputView(hailo_buffers.raw_output, hailo_buffers.network_input))
  {
    XR_LOG_ERROR("ArmorDetector inference rejected a non-slot raw-input view");
    pipeline_infer_fail_count_.fetch_add(1U, std::memory_order_relaxed);
    ReleaseInferSlot(item);
    return;
  }
  try
  {
    ok = async_inference_enabled_
             ? network_.PrepareRawAsync(hailo_buffers.network_input,
                                        hailo_buffers.raw_output)
             : network_.InferRaw(hailo_buffers.network_input, hailo_buffers.raw_output,
                                 sync_timing);
  }
  catch (const std::exception& exception)
  {
    XR_LOG_ERROR("ArmorDetector inference worker exception: %s", exception.what());
  }
  if (!ok)
  {
    pipeline_infer_fail_count_.fetch_add(1U, std::memory_order_relaxed);
    ReleaseInferSlot(item);
    return;
  }

  if (!async_inference_enabled_)
  {
    std::unique_lock<std::mutex> lock(pipeline_mutex_);
    if (slot.generation.load(std::memory_order_relaxed) != item.generation ||
        slot.state.load(std::memory_order_relaxed) !=
            armor_detector_pipeline::InferSlotState::INFER_RUNNING)
    {
      pipeline_infer_fail_count_.fetch_add(1U, std::memory_order_relaxed);
      return;
    }
    if (!sync_timing.valid)
    {
      pipeline_infer_fail_count_.fetch_add(1U, std::memory_order_relaxed);
      ReleaseInferSlotLocked(item);
      return;
    }
    context.infer_timing = sync_timing;

    pipeline_cv_.wait(
        lock,
        [this, &slot, item]()
        {
          return slot.generation.load(std::memory_order_relaxed) != item.generation ||
                 slot.state.load(std::memory_order_relaxed) !=
                     armor_detector_pipeline::InferSlotState::INFER_RUNNING ||
                 HasFreePostSlotLocked();
        });
    if (slot.generation.load(std::memory_order_relaxed) != item.generation ||
        slot.state.load(std::memory_order_relaxed) !=
            armor_detector_pipeline::InferSlotState::INFER_RUNNING)
    {
      pipeline_infer_fail_count_.fetch_add(1U, std::memory_order_relaxed);
      return;
    }

    const auto post_item = AcquirePostSlotLocked();
    ASSERT(post_item.has_value());
    if (!post_item.has_value() || !HandoffInferToPostLocked(item, *post_item))
    {
      pipeline_post_fail_count_.fetch_add(1U, std::memory_order_relaxed);
      return;
    }
    const bool output_queued = output_queue_.TryPush(*post_item);
    ASSERT(output_queued);
    (void)output_queued;
    return;
  }

  {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    if (slot.generation.load(std::memory_order_relaxed) != item.generation)
    {
      pipeline_infer_fail_count_.fetch_add(1U, std::memory_order_relaxed);
      return;
    }
    if (slot.state.load(std::memory_order_relaxed) !=
            armor_detector_pipeline::InferSlotState::INFER_RUNNING ||
        async_inflight_ >= async_inflight_limit || !async_completions_.TryRegister(item))
    {
      pipeline_infer_fail_count_.fetch_add(1U, std::memory_order_relaxed);
      ReleaseInferSlotLocked(item);
      return;
    }
    ++async_inflight_;
  }

  int64_t submit_return_ns = 0;
  const bool submitted = network_.SubmitRawAsync(
      hailo_buffers.raw_output,
      [this, item](bool completion_ok,
                   detail::ArmorDetectorNetwork::HailoRawTimingSnapshot timing) noexcept
      {
        try
        {
          HandleInferenceCompletion(item, completion_ok, timing);
        }
        catch (...)
        {
          XR_LOG_ERROR("ArmorDetector inference completion handoff threw");
        }
      },
      submit_return_ns);
  if (submitted)
  {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    if (slot.generation.load(std::memory_order_relaxed) != item.generation ||
        slot.state.load(std::memory_order_relaxed) !=
            armor_detector_pipeline::InferSlotState::INFER_RUNNING)
    {
      pipeline_infer_fail_count_.fetch_add(1U, std::memory_order_relaxed);
      return;
    }
  }
  if (!submitted)
  {
    HandleInferenceCompletion(item, false, {});
  }
}

template <CameraTypes::FrameLayout FrameLayoutV>
void ArmorDetector<FrameLayoutV>::HandleInferenceCompletion(
    armor_detector_pipeline::WorkItem item, bool ok,
    detail::ArmorDetectorNetwork::HailoRawTimingSnapshot timing)
{
  {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    if (item.slot_id >= infer_slots_.size())
    {
      return;
    }
    auto& slot = infer_slots_[item.slot_id];
    auto& context = slot.context;
    if (slot.generation.load(std::memory_order_relaxed) != item.generation ||
        slot.state.load(std::memory_order_relaxed) !=
            armor_detector_pipeline::InferSlotState::INFER_RUNNING ||
        context.async_completed)
    {
      return;
    }

    const bool completion_ok = ok && timing.valid;
    if (async_completions_.MarkCompleted(item, completion_ok) !=
        armor_detector_pipeline::CompletionMark::MARKED)
    {
      return;
    }

    if (async_inflight_ > 0U)
    {
      --async_inflight_;
    }
    context.infer_timing = timing;
    context.async_ok = completion_ok;
    context.async_completed = true;
  }
  pipeline_cv_.notify_all();
}

template <CameraTypes::FrameLayout FrameLayoutV>
void ArmorDetector<FrameLayoutV>::DrainCompletedInferencesLocked()
{
  while (const auto completed = async_completions_.PeekCompleted())
  {
    const auto item = completed->item;
    if (item.slot_id >= infer_slots_.size())
    {
      (void)async_completions_.PopCompleted();
      pipeline_infer_fail_count_.fetch_add(1U, std::memory_order_relaxed);
      continue;
    }
    auto& slot = infer_slots_[item.slot_id];
    auto& context = slot.context;
    if (slot.generation.load(std::memory_order_relaxed) != item.generation ||
        slot.state.load(std::memory_order_relaxed) !=
            armor_detector_pipeline::InferSlotState::INFER_RUNNING ||
        !context.async_completed)
    {
      (void)async_completions_.PopCompleted();
      pipeline_infer_fail_count_.fetch_add(1U, std::memory_order_relaxed);
      ReleaseInferSlotLocked(item);
      continue;
    }
    if (!completed->ok || !context.async_ok)
    {
      (void)async_completions_.PopCompleted();
      pipeline_infer_fail_count_.fetch_add(1U, std::memory_order_relaxed);
      ReleaseInferSlotLocked(item);
      continue;
    }

    const auto post_item = AcquirePostSlotLocked();
    if (!post_item.has_value())
    {
      return;
    }
    const auto retired = async_completions_.PopCompleted();
    ASSERT(retired.has_value() &&
           armor_detector_pipeline::SameWorkItem(retired->item, item));
    if (!retired.has_value() ||
        !armor_detector_pipeline::SameWorkItem(retired->item, item))
    {
      pipeline_infer_fail_count_.fetch_add(1U, std::memory_order_relaxed);
      ReleasePostSlotLocked(*post_item);
      return;
    }
    if (!HandoffInferToPostLocked(item, *post_item))
    {
      pipeline_post_fail_count_.fetch_add(1U, std::memory_order_relaxed);
      continue;
    }
    const bool output_queued = output_queue_.TryPush(*post_item);
    ASSERT(output_queued);
    (void)output_queued;
  }
}

template <CameraTypes::FrameLayout FrameLayoutV>
bool ArmorDetector<FrameLayoutV>::HasFreePostSlotLocked() const
{
  return std::any_of(post_slots_.begin(), post_slots_.end(),
                     [](const PostSlot& slot)
                     {
                       return slot.state.load(std::memory_order_relaxed) ==
                              armor_detector_pipeline::PostSlotState::FREE;
                     });
}

template <CameraTypes::FrameLayout FrameLayoutV>
std::optional<armor_detector_pipeline::WorkItem>
ArmorDetector<FrameLayoutV>::AcquirePostSlotLocked()
{
  uint8_t candidate_slot_id = 0U;
  for (auto& slot : post_slots_)
  {
    const uint8_t current_slot_id = candidate_slot_id++;
    if (slot.state.load(std::memory_order_relaxed) !=
        armor_detector_pipeline::PostSlotState::FREE)
    {
      continue;
    }

    const uint64_t generation = armor_detector_pipeline::NextGeneration(
        slot.generation.load(std::memory_order_relaxed));
    slot.generation.store(generation, std::memory_order_relaxed);
    slot.context = {};
    slot.state.store(armor_detector_pipeline::PostSlotState::FUSING,
                     std::memory_order_release);
    return armor_detector_pipeline::WorkItem{current_slot_id, generation};
  }

  return std::nullopt;
}

template <CameraTypes::FrameLayout FrameLayoutV>
bool ArmorDetector<FrameLayoutV>::HandoffInferToPostLocked(
    armor_detector_pipeline::WorkItem infer_item,
    armor_detector_pipeline::WorkItem post_item)
{
  if (infer_item.slot_id >= infer_slots_.size() ||
      post_item.slot_id >= post_slots_.size())
  {
    ReleasePostSlotLocked(post_item);
    ReleaseInferSlotLocked(infer_item);
    return false;
  }

  auto& infer_slot = infer_slots_[infer_item.slot_id];
  auto& post_slot = post_slots_[post_item.slot_id];
  if (infer_slot.generation.load(std::memory_order_relaxed) != infer_item.generation ||
      infer_slot.state.load(std::memory_order_relaxed) !=
          armor_detector_pipeline::InferSlotState::INFER_RUNNING ||
      post_slot.generation.load(std::memory_order_relaxed) != post_item.generation ||
      post_slot.state.load(std::memory_order_relaxed) !=
          armor_detector_pipeline::PostSlotState::FUSING)
  {
    ReleasePostSlotLocked(post_item);
    ReleaseInferSlotLocked(infer_item);
    return false;
  }

  const std::size_t infer_buffer_id = infer_slot.hailo_buffer_id;
  const std::size_t post_buffer_id = post_slot.hailo_buffer_id;
  const bool buffer_ids_valid = infer_buffer_id < hailo_buffer_pool_.size() &&
                                post_buffer_id < hailo_buffer_pool_.size() &&
                                infer_buffer_id != post_buffer_id;
  const bool infer_buffers_valid =
      buffer_ids_valid &&
      network_.IsRawInputView(hailo_buffer_pool_[infer_buffer_id].raw_output,
                              hailo_buffer_pool_[infer_buffer_id].network_input);
  const bool post_buffers_valid =
      buffer_ids_valid &&
      network_.IsRawInputView(hailo_buffer_pool_[post_buffer_id].raw_output,
                              hailo_buffer_pool_[post_buffer_id].network_input);
  if (!infer_buffers_valid || !post_buffers_valid)
  {
    XR_LOG_ERROR("ArmorDetector output handoff rejected an invalid Hailo buffer pair");
    ReleasePostSlotLocked(post_item);
    ReleaseInferSlotLocked(infer_item);
    return false;
  }

  if (!armor_detector_pipeline::TryHandoffBufferOwnership(
          infer_slot.hailo_buffer_id, post_slot.hailo_buffer_id,
          hailo_buffer_pool_.size(), infer_slot.context, post_slot.context))
  {
    XR_LOG_ERROR("ArmorDetector output handoff rejected buffer ownership");
    ReleasePostSlotLocked(post_item);
    ReleaseInferSlotLocked(infer_item);
    return false;
  }

  ReleaseInferSlotLocked(infer_item);
  return true;
}

template <CameraTypes::FrameLayout FrameLayoutV>
void ArmorDetector<FrameLayoutV>::RunOutputFusion(armor_detector_pipeline::WorkItem item)
{
  if (item.slot_id >= post_slots_.size())
  {
    pipeline_post_fail_count_.fetch_add(1U, std::memory_order_relaxed);
    return;
  }
  auto& post_slot = post_slots_[item.slot_id];
  if (post_slot.generation.load(std::memory_order_acquire) != item.generation ||
      post_slot.state.load(std::memory_order_acquire) !=
          armor_detector_pipeline::PostSlotState::FUSING)
  {
    return;
  }

  auto& post_context = post_slot.context;
  ASSERT(post_slot.hailo_buffer_id < hailo_buffer_pool_.size());
  auto& post_buffers = hailo_buffer_pool_[post_slot.hailo_buffer_id];
  bool ok = false;
  try
  {
    ok = network_.DecodeRaw(post_buffers.raw_output, post_slot.decoded_output,
                            post_context.decode_timing);
  }
  catch (const std::exception& exception)
  {
    XR_LOG_ERROR("ArmorDetector output worker exception: %s", exception.what());
  }
  catch (...)
  {
    XR_LOG_ERROR("ArmorDetector output worker exception: unknown");
  }
  if (!ok || !post_context.decode_timing.valid)
  {
    pipeline_post_fail_count_.fetch_add(1U, std::memory_order_relaxed);
    ReleasePostSlot(item);
    return;
  }

  {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    if (post_slot.generation.load(std::memory_order_relaxed) != item.generation ||
        post_slot.state.load(std::memory_order_relaxed) !=
            armor_detector_pipeline::PostSlotState::FUSING)
    {
      pipeline_post_fail_count_.fetch_add(1U, std::memory_order_relaxed);
      ReleasePostSlotLocked(item);
      return;
    }

    post_slot.state.store(armor_detector_pipeline::PostSlotState::POST_QUEUED,
                          std::memory_order_release);
  }
  RunPostprocess(item);
}

template <CameraTypes::FrameLayout FrameLayoutV>
void ArmorDetector<FrameLayoutV>::RunPostprocess(armor_detector_pipeline::WorkItem item)
{
  const uint8_t post_slot_id = item.slot_id;
  if (post_slot_id >= post_slots_.size())
  {
    pipeline_post_fail_count_.fetch_add(1U, std::memory_order_relaxed);
    return;
  }
  auto& slot = post_slots_[post_slot_id];
  if (slot.generation.load(std::memory_order_acquire) != item.generation)
  {
    return;
  }
  auto expected = armor_detector_pipeline::PostSlotState::POST_QUEUED;
  if (!slot.state.compare_exchange_strong(
          expected, armor_detector_pipeline::PostSlotState::POST_RUNNING,
          std::memory_order_acq_rel, std::memory_order_acquire))
  {
    pipeline_post_fail_count_.fetch_add(1U, std::memory_order_relaxed);
    return;
  }
  auto& context = slot.context;
  bool ok = false;
  try
  {
    const auto* image = context.synced_frame.GetImageFrame();
    if (image != nullptr)
    {
      const auto& geometry = image->geometry;
      const int cv_type = detail::CvTypeFromEncoding(frame_layout.encoding);
      if (cv_type >= 0)
      {
        cv::Mat source(static_cast<int>(geometry.height),
                       static_cast<int>(geometry.width), cv_type,
                       const_cast<uint8_t*>(image->data.data()),
                       static_cast<std::size_t>(geometry.step));
        const cv::Mat bgr =
            detail::ConvertToBgrWithEncoding(source, frame_layout.encoding);
        if (!bgr.empty())
        {
          counters_ = {};
          MaybeDumpModelOutput(slot.decoded_output);
          const auto postprocess_begin = std::chrono::steady_clock::now();
          std::vector<CandidateArmor> armors;
          {
            auto postprocess_measurement = postprocess_duration_.Measure();
            armors = DecodeOutput(bgr, context.input_mapping, slot.decoded_output);
          }
          const auto postprocess_end = std::chrono::steady_clock::now();
          context.postprocess_latency_ms = std::chrono::duration<double, std::milli>(
                                               postprocess_end - postprocess_begin)
                                               .count();
          ProcessImage(bgr, context.synced_frame, std::move(armors),
                       context.frame_timestamp_us, context.camera_timestamp_us,
                       context.infer_timing, context.decode_timing,
                       context.preprocess_latency_ms, context.postprocess_latency_ms);
          ok = true;
        }
      }
    }
  }
  catch (const std::exception& exception)
  {
    XR_LOG_ERROR("ArmorDetector postprocess worker exception: %s", exception.what());
  }
  catch (...)
  {
    XR_LOG_ERROR("ArmorDetector postprocess worker exception: unknown");
  }

  if (!ok)
  {
    pipeline_post_fail_count_.fetch_add(1U, std::memory_order_relaxed);
  }
  slot.state.store(armor_detector_pipeline::PostSlotState::POST_DONE,
                   std::memory_order_release);
  ReleasePostSlot(item);
}

template <CameraTypes::FrameLayout FrameLayoutV>
void ArmorDetector<FrameLayoutV>::ReleaseInferSlot(armor_detector_pipeline::WorkItem item)
{
  std::lock_guard<std::mutex> lock(pipeline_mutex_);
  ReleaseInferSlotLocked(item);
}

template <CameraTypes::FrameLayout FrameLayoutV>
void ArmorDetector<FrameLayoutV>::ReleaseInferSlotLocked(
    armor_detector_pipeline::WorkItem item)
{
  if (item.slot_id >= infer_slots_.size())
  {
    return;
  }
  auto& slot = infer_slots_[item.slot_id];
  if (slot.generation.load(std::memory_order_relaxed) != item.generation ||
      slot.state.load(std::memory_order_relaxed) ==
          armor_detector_pipeline::InferSlotState::FREE)
  {
    return;
  }

  const bool completed = slot.context.admission_counted;
  slot.context = {};
  if (completed)
  {
    pipeline_completed_count_.fetch_add(1U, std::memory_order_release);
  }
  slot.state.store(armor_detector_pipeline::InferSlotState::FREE,
                   std::memory_order_release);
  pipeline_cv_.notify_all();
}

template <CameraTypes::FrameLayout FrameLayoutV>
void ArmorDetector<FrameLayoutV>::ReleasePostSlot(
    armor_detector_pipeline::WorkItem item)
{
  std::lock_guard<std::mutex> lock(pipeline_mutex_);
  ReleasePostSlotLocked(item);
}

template <CameraTypes::FrameLayout FrameLayoutV>
void ArmorDetector<FrameLayoutV>::ReleasePostSlotLocked(
    armor_detector_pipeline::WorkItem item)
{
  if (item.slot_id >= post_slots_.size())
  {
    return;
  }
  auto& slot = post_slots_[item.slot_id];
  if (slot.generation.load(std::memory_order_relaxed) != item.generation ||
      slot.state.load(std::memory_order_relaxed) ==
          armor_detector_pipeline::PostSlotState::FREE)
  {
    return;
  }

  const bool completed = slot.context.admission_counted;
  slot.context = {};
  if (completed)
  {
    pipeline_completed_count_.fetch_add(1U, std::memory_order_release);
  }
  slot.state.store(armor_detector_pipeline::PostSlotState::FREE,
                   std::memory_order_release);
  pipeline_cv_.notify_all();
}

template <CameraTypes::FrameLayout FrameLayoutV>
void ArmorDetector<FrameLayoutV>::InferenceThreadFun(ArmorDetector<FrameLayoutV>* self)
{
#if defined(__linux__)
  (void)pthread_setname_np(pthread_self(), "armor-infer");
  if (const char* cpu_text = std::getenv("ARMOR_DETECTOR_INFERENCE_CPU");
      cpu_text != nullptr && cpu_text[0] != '\0')
  {
    char* end = nullptr;
    const long cpu = std::strtol(cpu_text, &end, 10);
    if (end == cpu_text || end == nullptr || end[0] != '\0' || cpu < 0 ||
        cpu >= CPU_SETSIZE)
    {
      XR_LOG_ERROR("ArmorDetector invalid inference CPU '%s'", cpu_text);
    }
    else
    {
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(static_cast<int>(cpu), &set);
      const int affinity_result =
          pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
      if (affinity_result == 0)
      {
        XR_LOG_PASS("ArmorDetector inference worker pinned to CPU %ld", cpu);
      }
      else
      {
        XR_LOG_ERROR("ArmorDetector failed to pin inference worker to CPU %ld (err=%d)",
                     cpu, affinity_result);
      }
    }
  }
#endif
  XR_LOG_INFO("ArmorDetector inference worker starting");
  while (true)
  {
    armor_detector_pipeline::WorkItem item{};
    bool have_item = false;
    {
      std::unique_lock<std::mutex> lock(self->pipeline_mutex_);
      self->DrainCompletedInferencesLocked();
      if (self->async_inflight_ < async_inflight_limit)
      {
        have_item = self->inference_queue_.TryPop(item);
      }
      if (!have_item)
      {
        self->inference_worker_active_.store(false, std::memory_order_release);
        self->pipeline_cv_.wait(
            lock,
            [self]()
            {
              const auto completed = self->async_completions_.PeekCompleted();
              return (completed.has_value() &&
                      (!completed->ok || self->HasFreePostSlotLocked())) ||
                     (self->async_inflight_ < async_inflight_limit &&
                      !self->inference_queue_.Empty());
            });
        continue;
      }
    }
    self->inference_worker_active_.store(true, std::memory_order_release);
    self->RunInference(item);
  }
}

template <CameraTypes::FrameLayout FrameLayoutV>
void ArmorDetector<FrameLayoutV>::OutputFusionThreadFun(ArmorDetector<FrameLayoutV>* self)
{
#if defined(__linux__)
  (void)pthread_setname_np(pthread_self(), "armor-output");
#endif
  XR_LOG_INFO("ArmorDetector output fusion worker starting");
  while (true)
  {
    const auto item = self->output_queue_.WaitPop();
    self->output_worker_active_.store(true, std::memory_order_release);
    self->RunOutputFusion(item);
    self->output_worker_active_.store(false, std::memory_order_release);
  }
}
