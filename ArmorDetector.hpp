#pragma once

// clang-format off
/* === MODULE MANIFEST V2 ===
module_description: 装甲板检测：v4 模型直接吃原始 Bayer，解码、NMS、编号分类、颜色过滤，发布检测帧 / Armor detection: the v4 model takes raw Bayer; decode, NMS, number classification and colour filtering, then publish detected frames
depends:
- id: QDU-Robomaster/CameraBase
  ref: same-or-dev
- id: QDU-Robomaster/AutoAimTypes
  ref: same-or-dev
- id: xrobot-org/DurationStatistics
  ref: same-or-dev
=== END MANIFEST === */
// clang-format on

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "ArmorV4Decoder.hpp"
#include "AutoAimTypes.hpp"
#include "DurationStatistics.hpp"
#include "InferenceBackend.hpp"
#include "NumberClassifier.hpp"
#include "Sha256.hpp"
#include "libxr_def.hpp"
#include "logger.hpp"
#include "message.hpp"
#ifdef ARMOR_DETECTOR_HAVE_HAILORT
#include "HailoBackend.hpp"
#endif
#ifdef ARMOR_DETECTOR_HAVE_OPENVINO
#include "OpenVinoBackend.hpp"
#endif

/// 要打的颜色 / Colour to engage.
enum class TargetColor : uint8_t
{
  RED,
  BLUE,
  FROM_REFEREE,  ///< 按裁判系统的本机 ID 取对方颜色 / Opponent of the referee robot ID
};

/// 检测设置，与 YAML 一一对应 / Detector settings, one-to-one with the YAML.
struct DetectorSettings
{
  std::string_view camera_name;
  std::string_view model_dir;  ///< armor-models 的下载目录 / armor-models download dir
  std::string_view
      detector_model;  ///< .hef 用 Hailo，.onnx 用 OpenVINO / Backend by suffix
  std::string_view number_model;     ///< armor_num_v1.onnx 或 _slim
  std::string_view openvino_device;  ///< CPU、GPU、NPU；Hailo 不用 / Unused with Hailo
  float min_confidence;              ///< 0.4
  float nms_iou;                     ///< 0.3
  TargetColor target_color;
  uint32_t inflight;  ///< 同时在推理的帧数 / Frames in inference at once
};

/**
 * @brief 装甲板检测。订阅 `<相机名>_synced`，发布 `<相机名>_detected`。
 *        Armor detection. Subscribes to `<camera>_synced`, publishes `<camera>_detected`.
 *
 * 同步帧进一格信箱（满则丢弃新帧，不阻塞上游）。提交线程把原始 Bayer
 * 拷进空闲推理槽并提交；
 * 后处理线程按提交顺序等推理完成，解码、NMS、颜色过滤、编号分类，把角点换到原生坐标后发布。
 * 收进的每一帧都发布一次，没有装甲板也发。编号为非装甲板的检测丢弃。
 * A synced frame goes into a one-frame mailbox (a new frame is dropped when it is full,
 * without blocking upstream). The submit thread copies the raw Bayer into a free slot
 * and submits it; the post thread waits for inferences in submission order, decodes,
 * runs NMS, filters colour, classifies numbers, maps corners to native pixels and
 * publishes. Every accepted frame is published once, with or without armors.
 * Detections classified as non-armor are dropped.
 */
class ArmorDetector
{
 public:
  /// 已知模型文件及其 SHA256 / Known model files and their SHA256.
  struct KnownModel
  {
    const char* file;
    const char* sha256;
  };
  static constexpr std::array<KnownModel, 5> KNOWN_MODELS = {{
      {"armor_det_v4.hef",
       "dc085979972e579e7fcd7e641899abb98fd72851823d9bc1c4cc57c3ecf848b0"},
      {"armor_det_v4_near.hef",
       "3a108d0910911fee6b1bdcfd1f38709e364c28279e0d0a6570f5eb4c3e04d9ac"},
      {"armor_det_v4.onnx",
       "0e4a789706bdd4be525cbc65d7481dfa9b6512f4016eccd0b356e554998d4974"},
      {"armor_num_v1.onnx",
       "9d91e105450a22f5f6e475c7a2efe51f77e305a21fe0a653ae481f35986d0603"},
      {"armor_num_v1_slim.onnx",
       "3d90b154c44b89896c904a36f0a5947f0ed27439a350692b13dc328a684c17d8"},
  }};
  /// 裁判系统摘要包 Topic，首字节为本机 robot_id / Referee summary Topic; its first byte
  /// is the robot_id.
  static constexpr const char* REFEREE_TOPIC = "robot_game_ref";
  static constexpr const char* REFEREE_DOMAIN = "host";

  explicit ArmorDetector(const DetectorSettings& settings)
      : settings_(settings),
        camera_name_(settings.camera_name),
        detected_topic_(LibXR::Topic::CreateTopic<const AutoAim::DetectedFrame*>(
            StageTopicName(camera_name_, AutoAim::STAGE_DETECTED).c_str()))
  {
    REQUIRE(settings.inflight >= 1);
    const std::string dir(settings.model_dir);
    const std::string detector_path = dir + "/" + std::string(settings.detector_model);
    const std::string number_path = dir + "/" + std::string(settings.number_model);
    REQUIRE(CheckModel(detector_path) && CheckModel(number_path));
    backend_ = MakeBackend(detector_path, settings.inflight,
                           std::string(settings.openvino_device));
    REQUIRE(backend_ != nullptr);
    numbers_ = std::make_unique<NumberClassifier>(number_path);
    target_color_.store(settings.target_color == TargetColor::RED ? int(ArmorColor::RED)
                        : settings.target_color == TargetColor::BLUE
                            ? int(ArmorColor::BLUE)
                            : -1);
    if (settings.target_color == TargetColor::FROM_REFEREE)
    {
      SubscribeReferee();
    }
    for (std::size_t slot = 0; slot < backend_->Slots(); ++slot)
    {
      free_slots_.push_back(slot);
    }

    auto on_synced = LibXR::Topic::Callback::Create(
        [](bool, ArmorDetector* self, const AutoAim::SyncedFrame* frame)
        { self->OnSynced(*frame); }, this);
    AutoAim::RequireTopic<const AutoAim::SyncedFrame*>(
        StageTopicName(camera_name_, AutoAim::STAGE_SYNCED))
        .RegisterCallback(on_synced);
    running_.store(true);
    submit_thread_ = std::thread([this]() { SubmitLoop(); });
    post_thread_ = std::thread([this]() { PostLoop(); });
  }

  ~ArmorDetector()
  {
    running_.store(false);
    mailbox_cv_.notify_all();
    slots_cv_.notify_all();
    inflight_cv_.notify_all();
    submit_thread_.join();
    post_thread_.join();
  }

  ArmorDetector(const ArmorDetector&) = delete;
  ArmorDetector& operator=(const ArmorDetector&) = delete;

  /// 打印周期摘要 / Print the periodic summary.
  void OnMonitor()
  {
    XR_LOG_INFO("%s detector: published=%u busy_drop=%u armors=%u negative=%u failed=%u",
                camera_name_.c_str(), stats_.published.exchange(0),
                stats_.busy_drop.exchange(0), stats_.armors.exchange(0),
                stats_.negative.exchange(0), stats_.failed.exchange(0));
    for (const auto& [name, d] :
         {std::pair{"preprocess", &preprocess_}, std::pair{"inference", &inference_},
          std::pair{"postprocess", &postprocess_}, std::pair{"result", &result_}})
    {
      const XRobot::DurationStatistics::Summary s = d->GetSummary();
      XR_LOG_INFO(
          "%s detector %s: n=%u avg=%uus min=%uus max=%uus", camera_name_.c_str(), name,
          static_cast<unsigned>(s.sample_count), static_cast<unsigned>(s.average_us),
          static_cast<unsigned>(s.minimum_us), static_cast<unsigned>(s.maximum_us));
    }
  }

 private:
  struct InFlight
  {
    AutoAim::SyncedFrame frame;
    std::size_t slot;  ///< NOT_STARTED 表示提交失败 / NOT_STARTED when submission failed
  };

  struct Stats
  {
    std::atomic<uint32_t> published{0};
    std::atomic<uint32_t> busy_drop{0};
    std::atomic<uint32_t> armors{0};
    std::atomic<uint32_t> negative{0};
    std::atomic<uint32_t> failed{0};
  };

  static bool CheckModel(const std::string& path)
  {
    const std::string file = std::filesystem::path(path).filename().string();
    for (const KnownModel& known : KNOWN_MODELS)
    {
      if (file == known.file)
      {
        const std::string sha = Sha256::OfFile(path);
        if (sha == known.sha256)
        {
          return true;
        }
        XR_LOG_ERROR(
            "model %s is missing or has SHA256 %s; fetch it with armor-models "
            "scripts/fetch_model.sh",
            path.c_str(), sha.empty() ? "(unreadable)" : sha.c_str());
        return false;
      }
    }
    XR_LOG_ERROR("model %s is not a known armor-models file", file.c_str());
    return false;
  }

  static std::unique_ptr<InferenceBackend> MakeBackend(const std::string& path,
                                                       std::size_t slots,
                                                       const std::string& device)
  {
    const std::string ext = std::filesystem::path(path).extension().string();
#ifdef ARMOR_DETECTOR_HAVE_HAILORT
    if (ext == ".hef")
    {
      return std::make_unique<HailoBackend>(path, slots);
    }
#endif
#ifdef ARMOR_DETECTOR_HAVE_OPENVINO
    if (ext == ".onnx")
    {
      return std::make_unique<OpenVinoBackend>(path, slots, device);
    }
#endif
    XR_LOG_ERROR("no inference backend for %s in this build", path.c_str());
    UNUSED(slots);
    UNUSED(device);
    return nullptr;
  }

  void SubscribeReferee()
  {
    referee_domain_.emplace(REFEREE_DOMAIN);
    LibXR::Topic::TopicHandle topic =
        LibXR::Topic::Find(REFEREE_TOPIC, &*referee_domain_);
    if (topic == nullptr)
    {
      XR_LOG_ERROR("target_color FROM_REFEREE needs the Topic %s/%s", REFEREE_DOMAIN,
                   REFEREE_TOPIC);
      REQUIRE(false);
    }
    auto on_referee = LibXR::Topic::Callback::Create(
        [](bool, ArmorDetector* self, const LibXR::ConstRawData& data)
        {
          if (data.addr_ == nullptr || data.size_ < 1)
          {
            return;
          }
          // 1–99 为红方，101–199 为蓝方 / 1–99 red, 101–199 blue.
          const uint8_t id = *static_cast<const uint8_t*>(data.addr_);
          if (id >= 1 && id < 100)
          {
            self->target_color_.store(int(ArmorColor::BLUE));
          }
          else if (id >= 101 && id < 200)
          {
            self->target_color_.store(int(ArmorColor::RED));
          }
        },
        this);
    LibXR::Topic(topic).RegisterCallback(on_referee);
  }

  void OnSynced(const AutoAim::SyncedFrame& frame)
  {
    {
      std::lock_guard<std::mutex> lock(mailbox_mutex_);
      if (mailbox_)
      {
        stats_.busy_drop.fetch_add(1, std::memory_order_relaxed);
        return;
      }
      mailbox_ = frame;
    }
    mailbox_cv_.notify_one();
  }

  void SubmitLoop()
  {
    while (true)
    {
      AutoAim::SyncedFrame frame;
      {
        std::unique_lock<std::mutex> lock(mailbox_mutex_);
        mailbox_cv_.wait(lock, [this]() { return mailbox_ || !running_.load(); });
        if (!running_.load())
        {
          return;
        }
        frame = std::move(*mailbox_);
        mailbox_.reset();
      }
      std::size_t slot = 0;
      {
        std::unique_lock<std::mutex> lock(slots_mutex_);
        slots_cv_.wait(lock,
                       [this]() { return !free_slots_.empty() || !running_.load(); });
        if (!running_.load())
        {
          return;
        }
        slot = free_slots_.front();
        free_slots_.pop_front();
      }
      {
        auto measure = preprocess_.Measure();
        std::memcpy(backend_->Input(slot), frame.image->data.data(),
                    CameraTypes::FRAME_BYTES);
      }
      const bool started = backend_->Start(slot);
      if (!started)
      {
        ReturnSlot(slot);
      }
      {
        std::lock_guard<std::mutex> lock(inflight_mutex_);
        inflight_.push_back({std::move(frame), started ? slot : NOT_STARTED});
      }
      inflight_cv_.notify_one();
    }
  }

  void PostLoop()
  {
    while (true)
    {
      InFlight item;
      {
        std::unique_lock<std::mutex> lock(inflight_mutex_);
        inflight_cv_.wait(lock,
                          [this]() { return !inflight_.empty() || !running_.load(); });
        if (inflight_.empty())
        {
          return;  // 停止且已清空 / Stopped and drained
        }
        item = std::move(inflight_.front());
        inflight_.pop_front();
      }
      AutoAim::DetectedFrame detected{std::move(item.frame), {}};
      if (item.slot == NOT_STARTED)
      {
        CountFailure("submit");
      }
      else
      {
        bool done = false;
        {
          auto measure = inference_.Measure();  // 后处理线程等结果的时间 / Wait time
          done = backend_->Wait(item.slot);
        }
        if (done)
        {
          auto measure = postprocess_.Measure();
          detected.armors = Process(*detected.synced.image, backend_->Outputs(item.slot));
        }
        else
        {
          CountFailure("wait");
        }
        ReturnSlot(item.slot);
      }
      auto measure = result_.Measure();
      const AutoAim::DetectedFrame* payload = &detected;
      detected_topic_.Publish(payload);
      stats_.published.fetch_add(1, std::memory_order_relaxed);
      stats_.armors.fetch_add(static_cast<uint32_t>(detected.armors.size()),
                              std::memory_order_relaxed);
    }
  }

  std::vector<AutoAim::Armor> Process(const ImageFrame& image,
                                      const ArmorV4::Outputs& outputs)
  {
    const std::vector<ArmorV4::Detection> detections =
        ArmorV4::Decode(outputs, settings_.min_confidence, settings_.nms_iou);
    std::vector<AutoAim::Armor> armors;
    const int target = target_color_.load();
    cv::Mat gray;
    for (const ArmorV4::Detection& d : detections)
    {
      const ArmorColor color = d.color == 1 ? ArmorColor::RED : ArmorColor::BLUE;
      if (target < 0 || int(color) != target)
      {
        continue;  // 颜色未定或不是对方 / Colour unknown or not the opponent
      }
      if (gray.empty())
      {
        gray = NumberClassifier::Gray(image.data.data(), CameraTypes::FRAME_WIDTH,
                                      CameraTypes::FRAME_HEIGHT);
      }
      const ArmorNumber number = numbers_->Classify(gray, d.corners);
      if (number == ArmorNumber::NEGATIVE)
      {
        stats_.negative.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      AutoAim::Armor armor{
          color, number, d.size == 1 ? ArmorType::LARGE : ArmorType::SMALL, d.score, {}};
      for (int k = 0; k < 4; ++k)
      {
        const CameraTypes::Point2d p =
            CameraTypes::FrameToNative(image.geometry, {d.corners[k].x, d.corners[k].y});
        armor.corners[k] = {static_cast<float>(p.x), static_cast<float>(p.y)};
      }
      armors.push_back(armor);
    }
    return armors;
  }

  void ReturnSlot(std::size_t slot)
  {
    {
      std::lock_guard<std::mutex> lock(slots_mutex_);
      free_slots_.push_back(slot);
    }
    slots_cv_.notify_one();
  }

  void CountFailure(const char* stage)
  {
    const uint32_t n = ++total_failures_;
    stats_.failed.fetch_add(1, std::memory_order_relaxed);
    if (AutoAim::ShouldLog(n))
    {
      XR_LOG_ERROR("%s detector: inference %s failed (%u so far)", camera_name_.c_str(),
                   stage, n);
    }
  }

  static constexpr std::size_t NOT_STARTED = SIZE_MAX;

  const DetectorSettings settings_;
  const std::string camera_name_;
  LibXR::Topic detected_topic_;
  std::optional<LibXR::Topic::Domain> referee_domain_;
  std::unique_ptr<InferenceBackend> backend_;
  std::unique_ptr<NumberClassifier> numbers_;
  std::atomic<int> target_color_{-1};

  std::mutex mailbox_mutex_;
  std::condition_variable mailbox_cv_;
  std::optional<AutoAim::SyncedFrame> mailbox_;

  std::mutex slots_mutex_;
  std::condition_variable slots_cv_;
  std::deque<std::size_t> free_slots_;

  std::mutex inflight_mutex_;
  std::condition_variable inflight_cv_;
  std::deque<InFlight> inflight_;

  std::atomic<bool> running_{false};
  std::atomic<uint32_t> total_failures_{0};
  Stats stats_;
  XRobot::DurationStatistics preprocess_;
  XRobot::DurationStatistics inference_;
  XRobot::DurationStatistics postprocess_;
  XRobot::DurationStatistics result_;
  std::thread submit_thread_;
  std::thread post_thread_;
};
