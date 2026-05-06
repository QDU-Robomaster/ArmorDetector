#pragma once

#include <chrono>

/**
 * @file ArmorDetectorRuntime.hpp
 * @brief ArmorDetector 配置、同步帧消费线程和帧级运行时 glue。
 */

/**
 * @brief 构造 detector，加载模型并启动同步帧 worker。
 *
 * detector 当前不直接访问 HardwareContainer；图像和 IMU 由 CameraFrameSync 输入。
 *
 * @tparam CameraInfoV 编译期相机参数。
 * @param app 应用管理器，用于注册本模块。
 * @param cfg detector 初始配置。
 * @param sync 同步帧来源。
 */
template <CameraTypes::CameraInfo CameraInfoV>
ArmorDetector<CameraInfoV>::ArmorDetector(LibXR::HardwareContainer&,
                                          LibXR::ApplicationManager& app,
                                          Config cfg,
                                          Sync& sync)
    : sync_(sync)
{
  SetConfig(cfg);

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
        [](bool, ArmorDetector* self, LibXR::RawData& data)
        {
          self->OnRefereeRobotGame(data);
        },
        this);
    referee_topic_.RegisterCallback(referee_callback_);
  }

  sync_frame_thread_.Create(this, SyncFrameThreadFun, "ArmorDetSync",
                            detail::sync_frame_thread_stack_size,
                            LibXR::Thread::Priority::HIGH);

  app.Register(*this);
}

/**
 * @brief 更新配置并重新加载模型。
 *
 * @tparam CameraInfoV 编译期相机参数。
 * @param cfg 新 detector 配置。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorDetector<CameraInfoV>::SetConfig(const Config& cfg)
{
  cfg_ = cfg;
  counters_ = {};

  const char* model_path = ARMOR_DETECTOR_MODEL_PATH;

  const char* openvino_device = cfg_.network.openvino_device;
  if (openvino_device == nullptr || openvino_device[0] == '\0')
  {
    openvino_device = "CPU";
  }
  const char* openvino_performance_mode = cfg_.network.openvino_performance_mode;
  if (openvino_performance_mode == nullptr ||
      openvino_performance_mode[0] == '\0')
  {
    openvino_performance_mode = "LATENCY";
  }

  XR_LOG_INFO(
      "ArmorDetector model=%s device=%s mode=%s path=%s",
      detail::detector_model_name, openvino_device, openvino_performance_mode,
      model_path);
  network_.Configure(model_path, openvino_device, openvino_performance_mode);
}

/**
 * @brief 裁判系统摘要包回调。
 *
 * 当前只依赖 RobotGameRefereePack 第一个字段 RobotStatus 的首字节 robot_id。
 *
 * @tparam CameraInfoV 编译期相机参数。
 * @param data 裁判系统摘要包原始 payload。
 */
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorDetector<CameraInfoV>::OnRefereeRobotGame(const LibXR::RawData& data)
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
 * @tparam CameraInfoV 编译期相机参数。
 * @return 当前用于语义过滤的目标颜色。
 */
template <CameraTypes::CameraInfo CameraInfoV>
ArmorColor ArmorDetector<CameraInfoV>::CurrentTargetColor() const
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
 * @tparam CameraInfoV 编译期相机参数。
 * @param robot_id 本机机器人 ID。
 * @return 0=红，1=蓝，-1=不确定。
 */
template <CameraTypes::CameraInfo CameraInfoV>
int ArmorDetector<CameraInfoV>::TargetColorFromRobotId(uint8_t robot_id)
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
 * 该函数保持 armors_frame 中的 source_frame 指针与当前检测结果同步，随后执行
 * detector 主链路、填充 metrics，并按 armors_frame、armors_result、metrics 的顺序发布。
 *
 * @tparam CameraInfoV 编译期相机参数。
 * @param img_msg BGR 图像。
 * @param synced_frame 原始同步帧。
 */
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
  metrics_msg_.overlap_kept_count = counters_.overlap_kept_count;
  metrics_msg_.semantic_kept_count = counters_.semantic_kept_count;
  metrics_msg_.armor_count = static_cast<uint32_t>(armors_packet_.results.size());
  metrics_msg_.pnp_success_count = counters_.pnp_success_count;
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

  if ((frame_index_ % detail::metrics_log_period) == 0U)
  {
    XR_LOG_INFO(
        "ArmorDetector frame=%llu armors=%u decoded=%u overlap_kept=%u semantic_kept=%u pnp=%u",
        static_cast<unsigned long long>(metrics_msg_.frame_index),
        metrics_msg_.armor_count, metrics_msg_.decoded_count,
        metrics_msg_.overlap_kept_count, metrics_msg_.semantic_kept_count,
        metrics_msg_.pnp_success_count);
    XR_LOG_INFO(
        "ArmorDetector semantic_discard=%u type_discard=%u discarded=%u max_obj=%.3f detector_ms=%.3f publish_ms=%.3f",
        metrics_msg_.semantic_discard_count,
        metrics_msg_.type_discard_count, metrics_msg_.discarded_count,
        metrics_msg_.max_objectness,
        metrics_msg_.detector_latency_ms, metrics_msg_.publish_latency_ms);
  }
}

/**
 * @brief 将同步帧中的原始图像数据包装成 OpenCV Mat 并转换为 BGR。
 *
 * CameraInfoV 的 encoding/width/height/step 必须与图像帧内存布局一致，否则
 * 后续 detector 的像素解释和 PnP 内参都会失配。
 *
 * @tparam CameraInfoV 编译期相机参数。
 * @param synced_frame CameraFrameSync 输出的同步帧。
 */
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

/**
 * @brief 同步帧 worker 主循环。
 *
 * worker 持有 CameraFrameSync subscriber，持续等待同步帧；超时只继续等待，非
 * OK 错误会停止 worker 并打印错误。
 *
 * @tparam CameraInfoV 编译期相机参数。
 * @param self detector 实例。
 */
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
