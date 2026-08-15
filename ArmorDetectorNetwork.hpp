#pragma once

/**
 * @file ArmorDetectorNetwork.hpp
 * @brief ArmorDetector 的 HailoRT 推理封装和模型输出适配。
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include "ArmorDetectorInputView.hpp"
#include "infer/ArmorDetectorModelAdapter.hpp"

#if defined(ARMOR_DETECTOR_HAVE_HAILORT)
#include <hailo/hailort.hpp>
#include <hailo/hef.hpp>
#include <hailo/inference_pipeline.hpp>
#include <hailo/network_group.hpp>
#include <hailo/vdevice.hpp>
#endif

namespace armor_detector_detail
{

template <typename T>
class HailoPageAllocator
{
 public:
  using value_type = T;

  HailoPageAllocator() noexcept = default;
  template <typename U>
  HailoPageAllocator(const HailoPageAllocator<U>&) noexcept
  {
  }

  [[nodiscard]] T* allocate(std::size_t count)
  {
    if (count == 0U)
    {
      return nullptr;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
    {
      throw std::bad_array_new_length();
    }

    std::size_t page_size = 4096U;
#if defined(__linux__)
    const long system_page_size = ::sysconf(_SC_PAGESIZE);
    if (system_page_size > 0)
    {
      page_size = static_cast<std::size_t>(system_page_size);
    }
#endif
    const std::size_t bytes = count * sizeof(T);
    if (bytes > std::numeric_limits<std::size_t>::max() - (page_size - 1U))
    {
      throw std::bad_array_new_length();
    }
    const std::size_t allocated_bytes =
        ((bytes + page_size - 1U) / page_size) * page_size;
    void* memory = std::aligned_alloc(page_size, allocated_bytes);
    if (memory == nullptr)
    {
      throw std::bad_alloc();
    }
    return static_cast<T*>(memory);
  }

  void deallocate(T* memory, std::size_t) noexcept { std::free(memory); }

  template <typename U>
  bool operator==(const HailoPageAllocator<U>&) const noexcept
  {
    return true;
  }
};

template <typename T>
using HailoPageVector = std::vector<T, HailoPageAllocator<T>>;

/**
 * @brief detector 推理后端类型。
 */
enum class DetectorBackendKind
{
  NONE,
  HAILORT,
};

/**
 * @brief ArmorDetector 推理后端封装。
 *
 * 对外保持单一接口：输入为 resize 后的 RGB8 图像，输出为当前模型适配器
 * 对应的 `CV_32F` 候选矩阵。当前稳定集合包含 `int8` 和 `int16` 两条线。
 */
class ArmorDetectorNetwork
{
 public:
  struct HailoTimingSnapshot
  {
    bool valid{false};
    double infer_ms{0.0};
    double tail_ms{0.0};
    double total_ms{0.0};
  };

  struct HailoRawTimingSnapshot
  {
    bool valid{false};
    double infer_ms{0.0};
    int64_t call_begin_ns{0};
    int64_t complete_ns{0};
  };

  struct HailoDecodeTimingSnapshot
  {
    bool valid{false};
    double tail_ms{0.0};
  };

  using HailoAsyncCompletion =
      std::function<void(bool, HailoRawTimingSnapshot)>;

  /** Per-frame Hailo output storage. Initialize once, then reuse without resizing. */
  class RawOutputSlot
  {
   public:
    RawOutputSlot() = default;
    RawOutputSlot(const RawOutputSlot&) = delete;
    RawOutputSlot& operator=(const RawOutputSlot&) = delete;
    RawOutputSlot(RawOutputSlot&&) noexcept = default;
    RawOutputSlot& operator=(RawOutputSlot&&) noexcept = default;

    [[nodiscard]] bool Valid() const { return valid_; }

   private:
    struct Buffer
    {
      std::variant<HailoPageVector<uint8_t>, HailoPageVector<uint16_t>,
                   HailoPageVector<float>> storage{};
    };

    HailoPageVector<uint8_t> input_buffer_{};
    std::vector<Buffer> buffers_{};
#if defined(ARMOR_DETECTOR_HAVE_HAILORT)
    hailort::ConfiguredInferModel::Bindings bindings_{};
#endif
    uint64_t descriptor_generation_{0};
    uint64_t infer_index_{0};
    bool bindings_initialized_{false};
    bool initialized_{false};
    bool valid_{false};

    friend class ArmorDetectorNetwork;
  };

  /**
   * @brief 初始化 HailoRT 后端并绑定当前模型适配器。
   * @param model 当前 detector 模型解析结果。
   * @return 后端可用且张量约定检查通过时返回 true。
   */
  bool Configure(const infer::ResolvedDetectorModel& model)
  {
    Reset();
    model_line_ = model.line;
    hailort_hef_path_ = NormalizeOptionalPath(model.hailort_hef_path);
    if (const char* env_hef = std::getenv("XR_ARMOR_HEF_PATH"); env_hef != nullptr &&
                                                            env_hef[0] != '\0')
    {
      hailort_hef_path_ = env_hef;
    }
#if defined(ARMOR_DETECTOR_HAVE_HAILORT)
    backend_kind_ = DetectorBackendKind::HAILORT;
    return ConfigureHailoRt(hailort_hef_path_.c_str());
#else
    input_shape_ = {model_input_width, model_input_height};
    XR_LOG_ERROR("ArmorDetector was built without HailoRT support");
    return false;
#endif
  }

  /**
   * @brief 当前后端是否已经成功初始化并可推理。
   */
  [[nodiscard]] bool Ready() const { return model_ready_; }

  /**
   * @brief 当前网络张量宽高。
   */
  [[nodiscard]] NetworkInputShape InputShape() const { return input_shape_; }

  /**
   * @brief 当前实际启用的后端名字。
   */
  [[nodiscard]] const std::string& BackendName() const { return backend_name_; }

  /**
   * @brief 当前后端是否为 HailoRT。
   */
  [[nodiscard]] bool UsesHailoRt() const
  {
    return backend_kind_ == DetectorBackendKind::HAILORT;
  }

  [[nodiscard]] std::size_t AsyncQueueSize() const
  {
#if defined(ARMOR_DETECTOR_HAVE_HAILORT)
    return hailo_async_queue_size_;
#else
    return 0U;
#endif
  }

  [[nodiscard]] HailoTimingSnapshot LastHailoTiming() const
  {
    return last_hailo_timing_;
  }

  /** Allocate one raw-output slot from the configured immutable descriptors. */
  bool InitRawOutputSlot(RawOutputSlot& slot)
  {
#if defined(ARMOR_DETECTOR_HAVE_HAILORT)
    if (!model_ready_ || backend_kind_ != DetectorBackendKind::HAILORT)
    {
      return false;
    }
    return InitRawOutputSlotHailoRt(slot);
#else
    (void)slot;
    return false;
#endif
  }

  /** Wrap a slot's page-aligned Hailo input buffer as an RGB8 OpenCV matrix. */
  bool InitRawInputView(RawOutputSlot& slot, cv::Mat& input) const
  {
    input.release();
#if defined(ARMOR_DETECTOR_HAVE_HAILORT)
    if (!model_ready_ || backend_kind_ != DetectorBackendKind::HAILORT ||
        !RawOutputSlotMatches(slot))
    {
      return false;
    }
    return BindRawRgbInputView({slot.input_buffer_.data(), slot.input_buffer_.size(),
                                input_shape_.width, input_shape_.height},
                               input);
#else
    (void)slot;
    return false;
#endif
  }

  [[nodiscard]] bool IsRawInputView(const RawOutputSlot& slot, const cv::Mat& input) const
  {
#if defined(ARMOR_DETECTOR_HAVE_HAILORT)
    return RawOutputSlotMatches(slot) &&
           input.data == slot.input_buffer_.data() &&
           MatchesRawRgbInputView({input.data, slot.input_buffer_.size(),
                                   input_shape_.width, input_shape_.height},
                                  input);
#else
    (void)slot;
    (void)input;
    return false;
#endif
  }

  /** Run only the synchronous Hailo inference call into a caller-owned slot. */
  bool InferRaw(const cv::Mat& input, RawOutputSlot& slot,
                HailoRawTimingSnapshot& timing)
  {
    timing = {};
    slot.valid_ = false;
    if (!model_ready_ || input.empty() ||
        backend_kind_ != DetectorBackendKind::HAILORT)
    {
      return false;
    }
#if defined(ARMOR_DETECTOR_HAVE_HAILORT)
    return InferRawHailoRt(input, slot, timing);
#else
    return false;
#endif
  }

  /** Wait for Hailo capacity and prepare a slot without submitting it. */
  bool PrepareRawAsync(const cv::Mat& input, RawOutputSlot& slot)
  {
    slot.valid_ = false;
    if (!model_ready_ || input.empty() ||
        backend_kind_ != DetectorBackendKind::HAILORT)
    {
      return false;
    }
#if defined(ARMOR_DETECTOR_HAVE_HAILORT)
    return PrepareRawAsyncHailoRt(input, slot);
#else
    return false;
#endif
  }

  /** Submit a prepared request; completion executes on a HailoRT callback thread. */
  bool SubmitRawAsync(RawOutputSlot& slot, HailoAsyncCompletion completion,
                      int64_t& submit_return_ns)
  {
    submit_return_ns = 0;
    if (!model_ready_ || backend_kind_ != DetectorBackendKind::HAILORT || !completion)
    {
      return false;
    }
#if defined(ARMOR_DETECTOR_HAVE_HAILORT)
    return SubmitRawAsyncHailoRt(slot, std::move(completion), submit_return_ns);
#else
    (void)slot;
    (void)completion;
    return false;
#endif
  }

  /** Fuse/dequantize a completed raw slot into the legacy CV_32F output matrix. */
  bool DecodeRaw(const RawOutputSlot& slot, cv::Mat& output,
                 HailoDecodeTimingSnapshot& timing) const
  {
    timing = {};
    if (!model_ready_ || backend_kind_ != DetectorBackendKind::HAILORT)
    {
      return false;
    }
#if defined(ARMOR_DETECTOR_HAVE_HAILORT)
    if (!RawOutputSlotMatches(slot) || !slot.valid_)
    {
      return false;
    }

    MaybeDumpHailoHeadTensors(slot, slot.infer_index_);
    const auto tail_begin = std::chrono::steady_clock::now();
    const bool ok = (model_line_ == infer::ModelLine::INT8)
                        ? BuildInt8HailoOutput(slot, output)
                        : BuildInt16HailoOutput(slot, output);
    const auto tail_end = std::chrono::steady_clock::now();
    if (ok)
    {
      timing.valid = true;
      timing.tail_ms =
          std::chrono::duration<double, std::milli>(tail_end - tail_begin).count();
    }
    return ok;
#else
    (void)slot;
    return false;
#endif
  }

  /**
   * @brief 对一帧 RGB8 网络输入执行推理。
   * @param input 已经 resize 到网络宽高的 RGB8 图像。
   * @param output 输出候选矩阵，格式与当前模型适配器约定一致。
   * @return 成功时返回 true。
   */
  bool Infer(const cv::Mat& input, cv::Mat& output)
  {
    output.release();
    if (!model_ready_ || input.empty())
    {
      return false;
    }

    if (backend_kind_ != DetectorBackendKind::HAILORT)
    {
      XR_LOG_ERROR("ArmorDetector backend is not configured");
      return false;
    }
#if defined(ARMOR_DETECTOR_HAVE_HAILORT)
    return InferHailoRt(input, output);
#else
    return false;
#endif
  }

  template <typename DetectionT, typename DetectionBuilder>
  std::optional<std::vector<DetectionT>> InferHailoDetections(
      const cv::Mat& input, DetectionBuilder&& build_detection)
  {
#if defined(ARMOR_DETECTOR_HAVE_HAILORT)
    if (!model_ready_ || backend_kind_ != DetectorBackendKind::HAILORT ||
        input.empty())
    {
      last_hailo_timing_ = {};
      return std::nullopt;
    }

    cv::Mat continuous_input = input;
    if (!continuous_input.isContinuous())
    {
      continuous_input = input.clone();
    }

    HailoRawTimingSnapshot infer_timing{};
    if (!InferRaw(continuous_input, compatibility_raw_output_slot_, infer_timing))
    {
      last_hailo_timing_ = {};
      return std::nullopt;
    }

    try
    {
      MaybeDumpHailoHeadTensors(compatibility_raw_output_slot_,
                                compatibility_raw_output_slot_.infer_index_);
      const auto tail_begin = std::chrono::steady_clock::now();
      auto detections = DecodeHailoDetections<DetectionT>(
          compatibility_raw_output_slot_,
          std::forward<DetectionBuilder>(build_detection));
      const auto tail_end = std::chrono::steady_clock::now();
      if (detections.has_value())
      {
        ++hailo_timing_count_;
        const double infer_ms = infer_timing.infer_ms;
        const double tail_ms =
            std::chrono::duration<double, std::milli>(tail_end - tail_begin).count();
        last_hailo_timing_ = {
            true,
            infer_ms,
            tail_ms,
            infer_ms + tail_ms,
        };
        if (hailo_timing_count_ <= 5U || (hailo_timing_count_ % 30U) == 0U)
        {
          XR_LOG_INFO(
              "ArmorDetector Hailo direct timing count=%u infer_ms=%.3f tail_ms=%.3f total_ms=%.3f",
              hailo_timing_count_, infer_ms, tail_ms, infer_ms + tail_ms);
        }
      }
      else
      {
        last_hailo_timing_ = {};
      }
      return detections;
    }
    catch (const std::exception& exception)
    {
      XR_LOG_ERROR("ArmorDetector Hailo inference exception: %s",
                   exception.what());
      last_hailo_timing_ = {};
      return std::nullopt;
    }
#else
    (void)input;
    (void)build_detection;
    return std::nullopt;
#endif
  }

 private:
  int CandidateCount() const
  {
    return infer::resolve_model_infer_adapter(model_line_).candidate_count;
  }

  int OutputWidth() const
  {
    return infer::resolve_model_infer_adapter(model_line_).output_width;
  }

  int HailortOutputFieldCount() const
  {
    return OutputWidth();
  }

  void Reset()
  {
    model_ready_ = false;
    backend_kind_ = DetectorBackendKind::NONE;
    backend_name_ = "NONE";
    input_shape_ = {};
    hailort_hef_path_.clear();
    last_hailo_timing_ = {};
#if defined(ARMOR_DETECTOR_HAVE_HAILORT)
    hailo_infer_call_count_ = 0;
    hailo_dump_active_prefix_.clear();
    hailo_dumped_ = false;
    hailo_tail_call_count_ = 0;
    hailo_fuse_snapshot_count_ = 0;
    ++descriptor_generation_;
    if (descriptor_generation_ == 0U)
    {
      ++descriptor_generation_;
    }

    hailo_input_name_.clear();
    hailo_input_frame_size_ = 0U;
    hailo_output_infos_.clear();
    hailo_output_descriptors_.clear();
    hailo_async_queue_size_ = 0U;
    compatibility_raw_output_slot_ = {};
    hailo_configured_infer_model_.reset();
    hailo_infer_model_.reset();
    hailo_vdevice_.reset();
#endif
  }

  static std::string NormalizeOptionalPath(const char* path)
  {
    if (path == nullptr)
    {
      return {};
    }
    return std::string(path);
  }

  [[nodiscard]] bool OutputShapeMatches(const cv::Mat& output) const
  {
    return (output.rows == CandidateCount() &&
            output.cols == OutputWidth()) ||
           (output.rows == OutputWidth() &&
            output.cols == CandidateCount());
  }

#if defined(ARMOR_DETECTOR_HAVE_HAILORT)
  struct HailoOutputDescriptor
  {
    std::string name{};
    hailo_format_type_t format_type{HAILO_FORMAT_TYPE_FLOAT32};
    size_t frame_size{0U};
    std::vector<hailo_quant_info_t> quant_infos{};
  };

  bool ConfigureHailoRt(const char* hef_path)
  {
    if (hef_path == nullptr || hef_path[0] == '\0')
    {
      XR_LOG_ERROR("ArmorDetector HailoRT HEF path is empty");
      return false;
    }

    try
    {
      auto vdevice_expected = hailort::VDevice::create();
      if (!vdevice_expected.has_value())
      {
        XR_LOG_ERROR("ArmorDetector failed to create Hailo VDevice status=%d",
                     static_cast<int>(vdevice_expected.status()));
        return false;
      }
      hailo_vdevice_ = vdevice_expected.release();

      auto infer_model_expected =
          hailo_vdevice_->create_infer_model(std::string(hef_path));
      if (!infer_model_expected.has_value())
      {
        XR_LOG_ERROR("ArmorDetector failed to create Hailo infer model status=%d",
                     static_cast<int>(infer_model_expected.status()));
        return false;
      }
      hailo_infer_model_ = infer_model_expected.release();
      hailo_infer_model_->set_batch_size(1U);

      const auto& input_names = hailo_infer_model_->get_input_names();
      const auto& output_names = hailo_infer_model_->get_output_names();
      if (input_names.size() != 1U || output_names.empty())
      {
        XR_LOG_ERROR("ArmorDetector invalid Hailo stream counts input=%zu output=%zu",
                     input_names.size(), output_names.size());
        return false;
      }

      auto input_infos_expected = hailo_infer_model_->hef().get_input_vstream_infos();
      auto output_infos_expected = hailo_infer_model_->hef().get_output_vstream_infos();
      if (!input_infos_expected.has_value() || !output_infos_expected.has_value() ||
          input_infos_expected.value().empty() ||
          output_infos_expected.value().empty())
      {
        XR_LOG_ERROR("ArmorDetector failed to query Hailo vstream infos");
        return false;
      }
      const auto input_infos = input_infos_expected.release();
      hailo_output_infos_ = output_infos_expected.release();

      input_shape_ = {static_cast<int>(input_infos.front().shape.width),
                      static_cast<int>(input_infos.front().shape.height)};
      if (!IsValidNetworkInputShape(input_shape_) ||
          input_infos.front().shape.features != 3U)
      {
        XR_LOG_ERROR("ArmorDetector invalid Hailo input shape: %ux%ux%u",
                     input_infos.front().shape.width,
                     input_infos.front().shape.height,
                     input_infos.front().shape.features);
        return false;
      }

      if (!HailoOutputShapeSupported())
      {
        XR_LOG_ERROR("ArmorDetector Hailo output shape is unsupported");
        return false;
      }

      auto input_stream_expected = hailo_infer_model_->input(input_names.front());
      if (!input_stream_expected.has_value())
      {
        XR_LOG_ERROR("ArmorDetector failed to access Hailo input stream status=%d",
                     static_cast<int>(input_stream_expected.status()));
        return false;
      }
      input_stream_expected.value().set_format_type(HAILO_FORMAT_TYPE_UINT8);
      if (model_line_ == infer::ModelLine::INT8)
      {
        for (const auto& output_name : output_names)
        {
          auto output_stream_expected = hailo_infer_model_->output(output_name);
          if (!output_stream_expected.has_value())
          {
            XR_LOG_ERROR("ArmorDetector failed to access Hailo output stream status=%d",
                         static_cast<int>(output_stream_expected.status()));
            return false;
          }
          output_stream_expected.value().set_format_type(HAILO_FORMAT_TYPE_FLOAT32);
        }
      }

      auto configured_expected = hailo_infer_model_->configure();
      if (!configured_expected.has_value())
      {
        XR_LOG_ERROR("ArmorDetector failed to configure Hailo infer model status=%d",
                     static_cast<int>(configured_expected.status()));
        return false;
      }
      hailo_configured_infer_model_ =
          std::make_unique<hailort::ConfiguredInferModel>(
              configured_expected.release());

      auto async_queue_size_expected =
          hailo_configured_infer_model_->get_async_queue_size();
      if (!async_queue_size_expected.has_value() ||
          async_queue_size_expected.value() == 0U)
      {
        XR_LOG_ERROR("ArmorDetector failed to query a usable Hailo async queue");
        return false;
      }
      hailo_async_queue_size_ = async_queue_size_expected.release();

      if (!PrepareHailoIoDescriptors())
      {
        return false;
      }

      backend_name_ = "HAILORT";
      backend_kind_ = DetectorBackendKind::HAILORT;
      model_ready_ = true;
      XR_LOG_PASS(
          "ArmorDetector loaded HailoRT HEF=%s input=%dx%d outputs=%zu async_queue=%zu",
          hef_path, input_shape_.width, input_shape_.height,
          hailo_output_infos_.size(), hailo_async_queue_size_);
      return true;
    }
    catch (const std::exception& exception)
    {
      XR_LOG_ERROR("ArmorDetector failed to initialize HailoRT backend: %s",
                   exception.what());
      return false;
    }
  }

  bool HailoOutputShapeSupported() const
  {
    if (model_line_ == infer::ModelLine::INT8)
    {
      if (hailo_output_infos_.size() == 1U)
      {
        const auto& info = hailo_output_infos_.front();
        const size_t total_values =
            static_cast<size_t>(info.shape.height) *
            static_cast<size_t>(info.shape.width) *
            static_cast<size_t>(info.shape.features);
        return total_values ==
               static_cast<size_t>(detail::int8_candidate_count *
                                   detail::int8_output_width);
      }

      if (hailo_output_infos_.size() == 6U)
      {
        return true;
      }
      return false;
    }

    size_t total_candidates = 0;
    for (const auto& info : hailo_output_infos_)
    {
      if (info.shape.features == 0U ||
          (info.shape.features % static_cast<uint32_t>(HailortOutputFieldCount())) != 0U)
      {
        XR_LOG_ERROR("ArmorDetector Hailo output %s has invalid feature count %u",
                     info.name, info.shape.features);
        return false;
      }
      total_candidates += static_cast<size_t>(info.shape.height) *
                          static_cast<size_t>(info.shape.width) *
                          static_cast<size_t>(info.shape.features /
                                              HailortOutputFieldCount());
    }
    return total_candidates == static_cast<size_t>(CandidateCount());
  }

  bool PrepareHailoIoDescriptors()
  {
    if (!hailo_infer_model_ || !hailo_configured_infer_model_ ||
        hailo_infer_model_->get_input_names().size() != 1U)
    {
      XR_LOG_ERROR("ArmorDetector requires exactly one configured Hailo input");
      return false;
    }

    hailo_input_name_ = hailo_infer_model_->get_input_names().front();
    auto input_stream_expected = hailo_infer_model_->input(hailo_input_name_);
    if (!input_stream_expected.has_value())
    {
      XR_LOG_ERROR("ArmorDetector cannot find Hailo input stream %s",
                   hailo_input_name_.c_str());
      return false;
    }
    hailo_input_frame_size_ = input_stream_expected.value().get_frame_size();
    const size_t expected_input_size =
        static_cast<size_t>(input_shape_.width) *
        static_cast<size_t>(input_shape_.height) * 3U;
    if (hailo_input_frame_size_ != expected_input_size)
    {
      XR_LOG_ERROR(
          "ArmorDetector Hailo input frame size mismatch: got=%zu expected=%zu",
          hailo_input_frame_size_, expected_input_size);
      return false;
    }

    hailo_output_descriptors_.clear();
    hailo_output_descriptors_.reserve(hailo_output_infos_.size());
    for (const auto& info : hailo_output_infos_)
    {
      auto output_stream_expected = hailo_infer_model_->output(info.name);
      if (!output_stream_expected.has_value())
      {
        XR_LOG_ERROR("ArmorDetector cannot find Hailo output stream %s",
                     info.name);
        return false;
      }
      const auto output_stream = output_stream_expected.release();
      const size_t frame_size = output_stream.get_frame_size();
      const auto format = output_stream.format();

      HailoOutputDescriptor descriptor{};
      descriptor.name = info.name;
      descriptor.format_type = format.type;
      descriptor.frame_size = frame_size;
      descriptor.quant_infos = output_stream.get_quant_infos();

      switch (format.type)
      {
        case HAILO_FORMAT_TYPE_UINT8:
          break;
        case HAILO_FORMAT_TYPE_UINT16:
          if ((frame_size % sizeof(uint16_t)) != 0U)
          {
            XR_LOG_ERROR("ArmorDetector Hailo output frame size invalid for UINT16: %s size=%zu",
                         info.name, frame_size);
            return false;
          }
          break;
        case HAILO_FORMAT_TYPE_FLOAT32:
          if ((frame_size % sizeof(float)) != 0U)
          {
            XR_LOG_ERROR("ArmorDetector Hailo output frame size invalid for FLOAT32: %s size=%zu",
                         info.name, frame_size);
            return false;
          }
          break;
        default:
          XR_LOG_ERROR("ArmorDetector unsupported Hailo output format type=%d for %s",
                       static_cast<int>(format.type), info.name);
          return false;
      }

      hailo_output_descriptors_.emplace_back(std::move(descriptor));
    }
    return InitRawOutputSlotHailoRt(compatibility_raw_output_slot_);
  }

  bool InitRawOutputSlotHailoRt(RawOutputSlot& slot)
  {
    slot = {};
    if (!hailo_configured_infer_model_ || hailo_output_descriptors_.empty() ||
        descriptor_generation_ == 0U)
    {
      return false;
    }

    slot.buffers_.reserve(hailo_output_descriptors_.size());
    for (const auto& descriptor : hailo_output_descriptors_)
    {
      RawOutputSlot::Buffer buffer{};
      switch (descriptor.format_type)
      {
        case HAILO_FORMAT_TYPE_UINT8:
          buffer.storage = HailoPageVector<uint8_t>(descriptor.frame_size, 0U);
          break;
        case HAILO_FORMAT_TYPE_UINT16:
          buffer.storage =
              HailoPageVector<uint16_t>(descriptor.frame_size / sizeof(uint16_t),
                                        0U);
          break;
        case HAILO_FORMAT_TYPE_FLOAT32:
          buffer.storage =
              HailoPageVector<float>(descriptor.frame_size / sizeof(float), 0.0F);
          break;
        default:
          slot = {};
          return false;
      }
      slot.buffers_.emplace_back(std::move(buffer));
    }

    slot.input_buffer_ = HailoPageVector<uint8_t>(hailo_input_frame_size_, 0U);
    auto bindings_expected = hailo_configured_infer_model_->create_bindings();
    if (!bindings_expected.has_value())
    {
      XR_LOG_ERROR("ArmorDetector failed to create Hailo bindings status=%d",
                   static_cast<int>(bindings_expected.status()));
      slot = {};
      return false;
    }
    slot.bindings_ = bindings_expected.release();

    auto input_binding_expected = slot.bindings_.input(hailo_input_name_);
    if (!input_binding_expected.has_value() ||
        HAILO_SUCCESS != input_binding_expected.value().set_buffer(
                             hailort::MemoryView(slot.input_buffer_.data(),
                                                hailo_input_frame_size_)))
    {
      XR_LOG_ERROR("ArmorDetector failed to bind Hailo input buffer");
      slot = {};
      return false;
    }
    for (size_t index = 0; index < hailo_output_descriptors_.size(); ++index)
    {
      const auto& descriptor = hailo_output_descriptors_[index];
      auto output_binding_expected = slot.bindings_.output(descriptor.name);
      if (!output_binding_expected.has_value())
      {
        XR_LOG_ERROR("ArmorDetector failed to access output binding %s",
                     descriptor.name.c_str());
        slot = {};
        return false;
      }
      void* data = nullptr;
      auto& storage = slot.buffers_[index].storage;
      switch (descriptor.format_type)
      {
        case HAILO_FORMAT_TYPE_UINT8:
          data = std::get<HailoPageVector<uint8_t>>(storage).data();
          break;
        case HAILO_FORMAT_TYPE_UINT16:
          data = std::get<HailoPageVector<uint16_t>>(storage).data();
          break;
        case HAILO_FORMAT_TYPE_FLOAT32:
          data = std::get<HailoPageVector<float>>(storage).data();
          break;
        default:
          slot = {};
          return false;
      }
      const auto status = output_binding_expected.value().set_buffer(
          hailort::MemoryView(data, descriptor.frame_size));
      if (HAILO_SUCCESS != status)
      {
        XR_LOG_ERROR("ArmorDetector failed to bind Hailo output %s status=%d",
                     descriptor.name.c_str(), static_cast<int>(status));
        slot = {};
        return false;
      }
    }

    slot.descriptor_generation_ = descriptor_generation_;
    slot.bindings_initialized_ = true;
    slot.initialized_ = true;
    return true;
  }

  [[nodiscard]] bool RawOutputSlotMatches(const RawOutputSlot& slot) const
  {
    if (!slot.initialized_ || !slot.bindings_initialized_ ||
        slot.descriptor_generation_ != descriptor_generation_ ||
        slot.input_buffer_.size() != hailo_input_frame_size_ ||
        slot.buffers_.size() != hailo_output_descriptors_.size())
    {
      return false;
    }

    for (size_t index = 0; index < hailo_output_descriptors_.size(); ++index)
    {
      const auto& descriptor = hailo_output_descriptors_[index];
      const auto& storage = slot.buffers_[index].storage;
      switch (descriptor.format_type)
      {
        case HAILO_FORMAT_TYPE_UINT8:
          if (!std::holds_alternative<HailoPageVector<uint8_t>>(storage) ||
              std::get<HailoPageVector<uint8_t>>(storage).size() !=
                  descriptor.frame_size)
          {
            return false;
          }
          break;
        case HAILO_FORMAT_TYPE_UINT16:
          if (!std::holds_alternative<HailoPageVector<uint16_t>>(storage) ||
              std::get<HailoPageVector<uint16_t>>(storage).size() * sizeof(uint16_t) !=
                  descriptor.frame_size)
          {
            return false;
          }
          break;
        case HAILO_FORMAT_TYPE_FLOAT32:
          if (!std::holds_alternative<HailoPageVector<float>>(storage) ||
              std::get<HailoPageVector<float>>(storage).size() * sizeof(float) !=
                  descriptor.frame_size)
          {
            return false;
          }
          break;
        default:
          return false;
      }
    }
    return true;
  }

  bool InferRawHailoRt(const cv::Mat& input, RawOutputSlot& slot,
                       HailoRawTimingSnapshot& timing)
  {
    const size_t input_bytes = input.total() * input.elemSize();
    if (input.type() != CV_8UC3 || input.rows != input_shape_.height ||
        input.cols != input_shape_.width || !input.isContinuous() ||
        input_bytes != hailo_input_frame_size_)
    {
      XR_LOG_ERROR(
          "ArmorDetector Hailo input invalid: type=%d size=%dx%d continuous=%d bytes=%zu expected=%dx%d/%zu",
          input.type(), input.cols, input.rows, input.isContinuous() ? 1 : 0,
          input_bytes, input_shape_.width, input_shape_.height,
          hailo_input_frame_size_);
      return false;
    }
    if (!RawOutputSlotMatches(slot))
    {
      XR_LOG_ERROR("ArmorDetector Hailo raw output slot is not initialized for this model");
      return false;
    }

    try
    {
      ++hailo_infer_call_count_;
      if (input.data != slot.input_buffer_.data())
      {
        std::memcpy(slot.input_buffer_.data(), input.data, input_bytes);
      }
      const auto infer_begin = std::chrono::steady_clock::now();
      timing.call_begin_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              infer_begin.time_since_epoch())
              .count();
      const auto status = hailo_configured_infer_model_->run(
          slot.bindings_, std::chrono::milliseconds(HAILO_DEFAULT_VSTREAM_TIMEOUT_MS));
      const auto infer_end = std::chrono::steady_clock::now();
      timing.complete_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              infer_end.time_since_epoch())
              .count();
      if (HAILO_SUCCESS != status)
      {
        XR_LOG_ERROR("ArmorDetector Hailo inference failed status=%d",
                     static_cast<int>(status));
        return false;
      }

      slot.infer_index_ = hailo_infer_call_count_;
      slot.valid_ = true;
      timing.valid = true;
      timing.infer_ms =
          std::chrono::duration<double, std::milli>(infer_end - infer_begin).count();
      return true;
    }
    catch (const std::exception& exception)
    {
      XR_LOG_ERROR("ArmorDetector Hailo inference exception: %s", exception.what());
      return false;
    }
  }

  bool PrepareRawAsyncHailoRt(const cv::Mat& input, RawOutputSlot& slot)
  {
    const size_t input_bytes = input.total() * input.elemSize();
    if (input.type() != CV_8UC3 || input.rows != input_shape_.height ||
        input.cols != input_shape_.width || !input.isContinuous() ||
        input_bytes != hailo_input_frame_size_ || !RawOutputSlotMatches(slot))
    {
      return false;
    }

    try
    {
      const auto ready_status = hailo_configured_infer_model_->wait_for_async_ready(
          std::chrono::milliseconds(HAILO_DEFAULT_VSTREAM_TIMEOUT_MS), 1U);
      if (HAILO_SUCCESS != ready_status)
      {
        XR_LOG_ERROR("ArmorDetector Hailo async readiness failed status=%d",
                     static_cast<int>(ready_status));
        return false;
      }

      if (input.data != slot.input_buffer_.data())
      {
        std::memcpy(slot.input_buffer_.data(), input.data, input_bytes);
      }
      return true;
    }
    catch (const std::exception& exception)
    {
      XR_LOG_ERROR("ArmorDetector Hailo async readiness exception: %s",
                   exception.what());
      return false;
    }
  }

  bool SubmitRawAsyncHailoRt(RawOutputSlot& slot,
                             HailoAsyncCompletion completion,
                             int64_t& submit_return_ns)
  {
    if (!RawOutputSlotMatches(slot) || !completion)
    {
      return false;
    }

    try
    {
      const uint64_t infer_index = ++hailo_infer_call_count_;
      const auto infer_begin = std::chrono::steady_clock::now();
      const int64_t infer_submit_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              infer_begin.time_since_epoch())
              .count();
      auto job_expected = hailo_configured_infer_model_->run_async(
          slot.bindings_,
          [&slot, completion = std::move(completion), infer_begin,
           infer_index,
           infer_submit_ns](const hailort::AsyncInferCompletionInfo& info) mutable noexcept
           {
            try
            {
               const auto infer_end = std::chrono::steady_clock::now();
               HailoRawTimingSnapshot timing{};
               timing.call_begin_ns = infer_submit_ns;
               timing.complete_ns =
                   std::chrono::duration_cast<std::chrono::nanoseconds>(
                       infer_end.time_since_epoch())
                       .count();
              const bool ok = HAILO_SUCCESS == info.status;
              if (ok)
              {
                slot.infer_index_ = infer_index;
                slot.valid_ = true;
                timing.valid = true;
                timing.infer_ms =
                    std::chrono::duration<double, std::milli>(infer_end - infer_begin)
                        .count();
              }
              completion(ok, timing);
            }
            catch (...)
            {
              XR_LOG_ERROR("ArmorDetector Hailo async completion threw");
            }
           });
      submit_return_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now().time_since_epoch())
              .count();
      if (!job_expected.has_value())
      {
        XR_LOG_ERROR("ArmorDetector Hailo async submit failed status=%d",
                     static_cast<int>(job_expected.status()));
        return false;
      }
      auto job = job_expected.release();
      job.detach();
      return true;
    }
    catch (const std::exception& exception)
    {
      XR_LOG_ERROR("ArmorDetector Hailo async exception: %s", exception.what());
      return false;
    }
  }

  static constexpr std::array<std::array<float, 8>, 3> kP3AnchorMul{{
      {{10.0F, 13.0F, 10.0F, 13.0F, 10.0F, 13.0F, 10.0F, 13.0F}},
      {{16.0F, 30.0F, 16.0F, 30.0F, 16.0F, 30.0F, 16.0F, 30.0F}},
      {{33.0F, 23.0F, 33.0F, 23.0F, 33.0F, 23.0F, 33.0F, 23.0F}},
  }};
  static constexpr std::array<std::array<float, 8>, 3> kP4AnchorMul{{
      {{30.0F, 61.0F, 30.0F, 61.0F, 30.0F, 61.0F, 30.0F, 61.0F}},
      {{62.0F, 45.0F, 62.0F, 45.0F, 62.0F, 45.0F, 62.0F, 45.0F}},
      {{59.0F, 119.0F, 59.0F, 119.0F, 59.0F, 119.0F, 59.0F, 119.0F}},
  }};
  static constexpr std::array<std::array<float, 8>, 3> kP5AnchorMul{{
      {{116.0F, 90.0F, 116.0F, 90.0F, 116.0F, 90.0F, 116.0F, 90.0F}},
      {{156.0F, 198.0F, 156.0F, 198.0F, 156.0F, 198.0F, 156.0F, 198.0F}},
      {{373.0F, 326.0F, 373.0F, 326.0F, 373.0F, 326.0F, 373.0F, 326.0F}},
  }};

  static constexpr std::array<const char*, 6> kInt8HostTailOutputNames{{
      "skd_host_tail_a8/conv45",
      "skd_host_tail_a8/concat11",
      "skd_host_tail_a8/conv59",
      "skd_host_tail_a8/concat14",
      "skd_host_tail_a8/conv72",
      "skd_host_tail_a8/concat16",
  }};

  static const std::array<std::array<float, 8>, 3>& AnchorMulForShape(
      const hailo_vstream_info_t& info)
  {
    if (info.shape.height == 64U && info.shape.width == 80U)
    {
      return kP3AnchorMul;
    }
    if (info.shape.height == 32U && info.shape.width == 40U)
    {
      return kP4AnchorMul;
    }
    return kP5AnchorMul;
  }

  static float DequantizeValue(uint8_t value, const hailo_quant_info_t& qi)
  {
    return (static_cast<float>(value) - qi.qp_zp) * qi.qp_scale;
  }

  static float DequantizeValue(uint16_t value, const hailo_quant_info_t& qi)
  {
    return (static_cast<float>(value) - qi.qp_zp) * qi.qp_scale;
  }

  static float DequantizeValue(float value, const hailo_quant_info_t&)
  {
    return value;
  }

  static float RoundToTailInputPrecision(float value)
  {
#if defined(__aarch64__) || defined(__ARM_FP16_FORMAT_IEEE) || \
    defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
    const __fp16 fp16_value = static_cast<__fp16>(value);
    return static_cast<float>(fp16_value);
#else
    return value;
#endif
  }

  template <typename T>
  float LoadHailoValue(const HailoPageVector<T>& storage, size_t index,
                       const std::vector<hailo_quant_info_t>& quant_infos,
                       int feature) const
  {
    if (index >= storage.size())
    {
      return 0.0F;
    }
    if (quant_infos.empty())
    {
      return static_cast<float>(storage[index]);
    }
    const auto& qi =
        quant_infos.size() == 1U
            ? quant_infos.front()
            : quant_infos[static_cast<size_t>(feature) % quant_infos.size()];
    return DequantizeValue(storage[index], qi);
  }

  template <typename T>
  bool FuseOneHailoOutput(const hailo_vstream_info_t& info,
                          const HailoPageVector<T>& storage,
                          const std::vector<hailo_quant_info_t>& quant_infos,
                          cv::Mat& output, int& row,
                          const std::array<std::array<float, 8>, 3>& anchor_mul) const
  {
    const int height = static_cast<int>(info.shape.height);
    const int width = static_cast<int>(info.shape.width);
    const int features = static_cast<int>(info.shape.features);
    const int anchors = features / HailortOutputFieldCount();
    const float stride_x =
        static_cast<float>(model_input_width) / static_cast<float>(width);
    const float stride_y =
        static_cast<float>(model_input_height) / static_cast<float>(height);

    for (int anchor = 0; anchor < anchors; ++anchor)
    {
      for (int y = 0; y < height; ++y)
      {
        for (int x = 0; x < width; ++x)
        {
          const size_t cell_offset =
              (static_cast<size_t>(y) * static_cast<size_t>(width) +
               static_cast<size_t>(x)) *
              static_cast<size_t>(features);
          if (row >= CandidateCount())
          {
            XR_LOG_ERROR("ArmorDetector Hailo fused too many rows");
            output.release();
            return false;
          }
          Eigen::Map<Eigen::Array<float, 1, detail::int16_output_width>> dst(
              output.ptr<float>(row++));
          const size_t anchor_offset =
              cell_offset +
              static_cast<size_t>(anchor * HailortOutputFieldCount());
          const auto point_block =
              LoadQuantizedBlock<8>(storage.data() + anchor_offset, quant_infos, 0);
          Eigen::Array<float, 1, 8> rounded_points = point_block;
          for (int index = 0; index < 8; ++index)
          {
            rounded_points(index) = RoundToTailInputPrecision(rounded_points(index));
          }
          Eigen::Map<const Eigen::Array<float, 1, 8>> anchor_scale(
              anchor_mul[anchor].data());
          Eigen::Array<float, 1, 8> grid_offset;
          grid_offset << static_cast<float>(x) * stride_x,
              static_cast<float>(y) * stride_y,
              static_cast<float>(x) * stride_x,
              static_cast<float>(y) * stride_y,
              static_cast<float>(x) * stride_x,
              static_cast<float>(y) * stride_y,
              static_cast<float>(x) * stride_x,
              static_cast<float>(y) * stride_y;
          dst.segment(0, 8) = rounded_points * anchor_scale + grid_offset;

          for (int field = 8; field < HailortOutputFieldCount(); ++field)
          {
            dst[field] =
                RoundToTailInputPrecision(
                    LoadHailoValue(storage,
                                   anchor_offset + static_cast<size_t>(field),
                                   quant_infos, field));
          }
        }
      }
    }
    return true;
  }

  template <typename T>
  static float LoadQuantizedValue(const HailoPageVector<T>& storage, size_t index,
                                  const std::vector<hailo_quant_info_t>& quant_infos,
                                  int feature)
  {
    if (index >= storage.size())
    {
      return 0.0F;
    }
    if (quant_infos.empty())
    {
      return static_cast<float>(storage[index]);
    }
    const auto& qi =
        quant_infos.size() == 1U
            ? quant_infos.front()
            : quant_infos[static_cast<size_t>(feature) % quant_infos.size()];
    return DequantizeValue(storage[index], qi);
  }

  template <int N, typename T>
  Eigen::Array<float, 1, N> LoadQuantizedBlock(const T* source,
                                               const std::vector<hailo_quant_info_t>& quant_infos,
                                               int feature_offset) const
  {
    Eigen::Array<float, 1, N> block;
    if (quant_infos.empty())
    {
      for (int index = 0; index < N; ++index)
      {
        block(index) = static_cast<float>(source[index]);
      }
      return block;
    }

    if (quant_infos.size() == 1U)
    {
      const auto& qi = quant_infos.front();
      for (int index = 0; index < N; ++index)
      {
        block(index) = DequantizeValue(source[index], qi);
      }
      return block;
    }

    for (int index = 0; index < N; ++index)
    {
      const auto& qi =
          quant_infos[static_cast<std::size_t>(feature_offset + index) %
                      quant_infos.size()];
      block(index) = DequantizeValue(source[index], qi);
    }
    return block;
  }

  template <typename T>
  bool CopySingleInt8OutputToMatrix(const HailoPageVector<T>& storage,
                                    const std::vector<hailo_quant_info_t>& quant_infos,
                                    cv::Mat& output) const
  {
    output.create(detail::int8_candidate_count, detail::int8_output_width, CV_32F);
    if (storage.size() < static_cast<size_t>(detail::int8_output_width *
                                             detail::int8_candidate_count))
    {
      XR_LOG_ERROR("ArmorDetector int8 single-output buffer too small: got=%zu expected=%d",
                   storage.size(), detail::int8_output_width *
                                       detail::int8_candidate_count);
      output.release();
      return false;
    }

    for (int candidate = 0; candidate < detail::int8_candidate_count; ++candidate)
    {
      float* dst = output.ptr<float>(candidate);
      for (int field = 0; field < detail::int8_output_width; ++field)
      {
        const std::size_t index =
            static_cast<std::size_t>(field) *
                static_cast<std::size_t>(detail::int8_candidate_count) +
            static_cast<std::size_t>(candidate);
        if (quant_infos.empty())
        {
          dst[field] = static_cast<float>(storage[index]);
          continue;
        }
        const auto& qi = quant_infos.size() == 1U
                             ? quant_infos.front()
                             : quant_infos[static_cast<std::size_t>(field) % quant_infos.size()];
        dst[field] = DequantizeValue(storage[index], qi);
      }
    }
    return true;
  }

  template <typename T>
  bool FuseInt8HostTailOutputs(const RawOutputSlot& slot, cv::Mat& output) const
  {
    const auto read_tensor =
        [this, &slot](const char* name) -> const HailoPageVector<T>* {
          for (size_t index = 0; index < hailo_output_descriptors_.size(); ++index)
          {
            if (hailo_output_descriptors_[index].name == name)
            {
              const auto& storage = slot.buffers_[index].storage;
              if (!std::holds_alternative<HailoPageVector<T>>(storage))
              {
                return nullptr;
              }
              return &std::get<HailoPageVector<T>>(storage);
            }
          }
          return nullptr;
        };
    const auto read_quant_infos =
        [this](const char* name) -> const std::vector<hailo_quant_info_t>* {
          for (const auto& descriptor : hailo_output_descriptors_)
          {
            if (descriptor.name == name)
            {
              return &descriptor.quant_infos;
            }
          }
          return nullptr;
        };

    const auto* conv45 = read_tensor(kInt8HostTailOutputNames[0]);
    const auto* concat11 = read_tensor(kInt8HostTailOutputNames[1]);
    const auto* conv59 = read_tensor(kInt8HostTailOutputNames[2]);
    const auto* concat14 = read_tensor(kInt8HostTailOutputNames[3]);
    const auto* conv72 = read_tensor(kInt8HostTailOutputNames[4]);
    const auto* concat16 = read_tensor(kInt8HostTailOutputNames[5]);
    const auto* q45 = read_quant_infos(kInt8HostTailOutputNames[0]);
    const auto* q11 = read_quant_infos(kInt8HostTailOutputNames[1]);
    const auto* q59 = read_quant_infos(kInt8HostTailOutputNames[2]);
    const auto* q14 = read_quant_infos(kInt8HostTailOutputNames[3]);
    const auto* q72 = read_quant_infos(kInt8HostTailOutputNames[4]);
    const auto* q16_p5 = read_quant_infos(kInt8HostTailOutputNames[5]);
    if (conv45 == nullptr || concat11 == nullptr || conv59 == nullptr ||
        concat14 == nullptr || conv72 == nullptr || concat16 == nullptr ||
        q45 == nullptr || q11 == nullptr || q59 == nullptr || q14 == nullptr ||
        q72 == nullptr || q16_p5 == nullptr)
    {
      XR_LOG_ERROR("ArmorDetector missing int8 host-tail outputs");
      return false;
    }

    output.create(detail::int8_candidate_count, detail::int8_output_width, CV_32F);
    Eigen::Map<detail::RowMajorArrayXXf> output_matrix(
        output.ptr<float>(), detail::int8_candidate_count,
        detail::int8_output_width);
    constexpr int widths[3] = {80, 40, 20};
    constexpr int heights[3] = {64, 32, 16};
    constexpr int strides[3] = {8, 16, 32};

    int row = 0;
    for (int scale = 0; scale < 3; ++scale)
    {
      const HailoPageVector<T>* conv8 =
          scale == 0 ? conv45 : (scale == 1 ? conv59 : conv72);
      const HailoPageVector<T>* concat16_tensor =
          scale == 0 ? concat11 : (scale == 1 ? concat14 : concat16);
      const auto& q8 = *(scale == 0 ? q45 : (scale == 1 ? q59 : q72));
      const auto& q16 =
          *(scale == 0 ? q11 : (scale == 1 ? q14 : q16_p5));
      const int width = widths[scale];
      const int height = heights[scale];
      const int stride = strides[scale];
      for (int y = 0; y < height; ++y)
      {
        for (int x = 0; x < width; ++x)
        {
          if (row >= detail::int8_candidate_count)
          {
            XR_LOG_ERROR("ArmorDetector int8 host-tail fused too many rows");
            output.release();
            return false;
          }
          auto dst = output_matrix.row(row++);
          const size_t cell8 = (static_cast<size_t>(y) * static_cast<size_t>(width) +
                                static_cast<size_t>(x)) * 8U;
          const size_t cell16 = (static_cast<size_t>(y) * static_cast<size_t>(width) +
                                 static_cast<size_t>(x)) * 16U;
          const auto conv8_block =
              LoadQuantizedBlock<8>(conv8->data() + cell8, q8, 0);
          const auto concat16_block =
              LoadQuantizedBlock<16>(concat16_tensor->data() + cell16, q16, 0);
          dst.setZero();
          dst.segment(0, 8) = conv8_block;
          const auto activation1 =
              1.0F / (1.0F + (-concat16_block.segment(4, 12)).exp());
          dst(8) = activation1.segment(0, 4).maxCoeff();
          dst.segment(9, 12) = activation1;

          Eigen::Array<float, 1, 8> grid_offset;
          grid_offset << static_cast<float>(x * stride), static_cast<float>(y * stride),
              static_cast<float>(x * stride), static_cast<float>(y * stride),
              static_cast<float>(x * stride), static_cast<float>(y * stride),
              static_cast<float>(x * stride), static_cast<float>(y * stride);
          dst.segment(0, 8) =
              dst.segment(0, 8) * static_cast<float>(stride * 2) + grid_offset;
        }
      }
    }

    if (row != detail::int8_candidate_count)
    {
      XR_LOG_ERROR("ArmorDetector int8 host-tail fused row count mismatch: got=%d expected=%d",
                   row, detail::int8_candidate_count);
      output.release();
      return false;
    }
    return true;
  }

  static std::string SanitizeHailoDumpName(const char* name)
  {
    std::string raw = (name == nullptr) ? std::string("tensor") : std::string(name);
    const std::size_t slash = raw.find_last_of('/');
    if (slash != std::string::npos && (slash + 1U) < raw.size())
    {
      raw = raw.substr(slash + 1U);
    }
    for (char& ch : raw)
    {
      const bool keep =
          (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
          (ch >= '0' && ch <= '9') || ch == '_' || ch == '-';
      if (!keep)
      {
        ch = '_';
      }
    }
    if (raw.empty())
    {
      raw = "tensor";
    }
    return raw;
  }

  template <typename T>
  bool DumpOneHailoOutputDequantized(const std::string& path,
                                     const hailo_vstream_info_t& info,
                                     const HailoPageVector<T>& storage,
                                     const std::vector<hailo_quant_info_t>& quant_infos) const
  {
    const std::filesystem::path dump_path(path);
    if (dump_path.has_parent_path())
    {
      std::error_code error;
      std::filesystem::create_directories(dump_path.parent_path(), error);
      if (error)
      {
        XR_LOG_ERROR("ArmorDetector failed to create Hailo dump dir %s: %s",
                     dump_path.parent_path().string().c_str(),
                     error.message().c_str());
        return false;
      }
    }

    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr)
    {
      XR_LOG_ERROR("ArmorDetector failed to open Hailo dump: %s", path.c_str());
      return false;
    }

    const int32_t height = static_cast<int32_t>(info.shape.height);
    const int32_t width = static_cast<int32_t>(info.shape.width);
    const int32_t features = static_cast<int32_t>(info.shape.features);
    std::fwrite(&height, sizeof(height), 1, file);
    std::fwrite(&width, sizeof(width), 1, file);
    std::fwrite(&features, sizeof(features), 1, file);
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        const size_t cell_offset =
            (static_cast<size_t>(y) * static_cast<size_t>(width) +
             static_cast<size_t>(x)) *
            static_cast<size_t>(features);
        for (int feature = 0; feature < features; ++feature)
        {
          const float value =
              LoadHailoValue(storage, cell_offset + static_cast<size_t>(feature),
                             quant_infos, feature);
          std::fwrite(&value, sizeof(value), 1, file);
        }
      }
    }
    std::fclose(file);
    return true;
  }

  void MaybeDumpHailoHeadTensors(const RawOutputSlot& slot,
                                 uint64_t infer_index) const
  {
    const char* prefix = std::getenv("ARMOR_DETECTOR_DUMP_HAILO_HEAD_PREFIX");
    if (prefix == nullptr || prefix[0] == '\0')
    {
      return;
    }

    const char* infer_env =
        std::getenv("ARMOR_DETECTOR_DUMP_HAILO_HEAD_INFER_INDEX");
    if (infer_env != nullptr && infer_env[0] != '\0')
    {
      const unsigned long requested_infer =
          std::strtoul(infer_env, nullptr, 10);
      if (requested_infer !=
          static_cast<unsigned long>(infer_index))
      {
        return;
      }
    }

    if (hailo_dump_active_prefix_ != prefix)
    {
      hailo_dump_active_prefix_ = prefix;
      hailo_dumped_ = false;
    }
    if (hailo_dumped_)
    {
      return;
    }

    for (size_t index = 0; index < hailo_output_infos_.size(); ++index)
    {
      const auto& info = hailo_output_infos_[index];
      const auto& descriptor = hailo_output_descriptors_[index];
      const auto& buffer = slot.buffers_[index];
      const std::string path =
          hailo_dump_active_prefix_ + "_" + SanitizeHailoDumpName(info.name) +
          ".f32bin";
      bool ok = false;
      switch (descriptor.format_type)
      {
        case HAILO_FORMAT_TYPE_UINT8:
          ok = DumpOneHailoOutputDequantized(
              path, info, std::get<HailoPageVector<uint8_t>>(buffer.storage),
              descriptor.quant_infos);
          break;
        case HAILO_FORMAT_TYPE_UINT16:
          ok = DumpOneHailoOutputDequantized(
              path, info, std::get<HailoPageVector<uint16_t>>(buffer.storage),
              descriptor.quant_infos);
          break;
        case HAILO_FORMAT_TYPE_FLOAT32:
          ok = DumpOneHailoOutputDequantized(
              path, info, std::get<HailoPageVector<float>>(buffer.storage),
              descriptor.quant_infos);
          break;
        default:
          XR_LOG_ERROR("ArmorDetector unsupported Hailo output format in dump path: %d",
                       static_cast<int>(descriptor.format_type));
          ok = false;
          break;
      }
      if (!ok)
      {
        return;
      }
      XR_LOG_INFO(
          "ArmorDetector dumped Hailo head=%s shape=%ux%ux%u path=%s",
          info.name, static_cast<unsigned>(info.shape.height),
          static_cast<unsigned>(info.shape.width),
          static_cast<unsigned>(info.shape.features), path.c_str());
    }
    hailo_dumped_ = true;
  }

  template <typename DetectionT, typename DetectionBuilder, typename T>
  bool DecodeOneHailoOutput(const hailo_vstream_info_t& info,
                            const HailoPageVector<T>& storage,
                            const std::vector<hailo_quant_info_t>& quant_infos,
                            DetectionBuilder&& build_detection,
                            std::vector<DetectionT>& detections,
                            const std::array<std::array<float, 8>, 3>& anchor_mul) const
  {
    const int height = static_cast<int>(info.shape.height);
    const int width = static_cast<int>(info.shape.width);
    const int features = static_cast<int>(info.shape.features);
    const int anchors = features / HailortOutputFieldCount();
    const float stride_x =
        static_cast<float>(model_input_width) / static_cast<float>(width);
    const float stride_y =
        static_cast<float>(model_input_height) / static_cast<float>(height);

    for (int anchor = 0; anchor < anchors; ++anchor)
    {
      for (int y = 0; y < height; ++y)
      {
        for (int x = 0; x < width; ++x)
        {
          const size_t cell_offset =
              (static_cast<size_t>(y) * static_cast<size_t>(width) +
               static_cast<size_t>(x)) *
              static_cast<size_t>(features);
          const size_t anchor_offset =
              cell_offset +
              static_cast<size_t>(anchor * HailortOutputFieldCount());
          Eigen::Array<float, 1, detail::int16_output_width> fields;
          fields.setZero();
          const auto point_block =
              LoadQuantizedBlock<8>(storage.data() + anchor_offset, quant_infos, 0);
          Eigen::Array<float, 1, 8> rounded_points = point_block;
          for (int index = 0; index < 8; ++index)
          {
            rounded_points(index) = RoundToTailInputPrecision(rounded_points(index));
          }
          Eigen::Map<const Eigen::Array<float, 1, 8>> anchor_scale(
              anchor_mul[anchor].data());
          Eigen::Array<float, 1, 8> grid_offset;
          grid_offset << static_cast<float>(x) * stride_x,
              static_cast<float>(y) * stride_y,
              static_cast<float>(x) * stride_x,
              static_cast<float>(y) * stride_y,
              static_cast<float>(x) * stride_x,
              static_cast<float>(y) * stride_y,
              static_cast<float>(x) * stride_x,
              static_cast<float>(y) * stride_y;
          fields.segment(0, 8) = rounded_points * anchor_scale + grid_offset;
          for (int field = 8; field < HailortOutputFieldCount(); ++field)
          {
            fields(field) =
                RoundToTailInputPrecision(
                    LoadHailoValue(storage,
                                   anchor_offset + static_cast<size_t>(field),
                                   quant_infos, field));
          }

          auto detection = build_detection(
              [&fields](int field)
              {
                return fields(field);
              });
          if (detection.has_value())
          {
            detections.emplace_back(std::move(detection.value()));
          }
        }
      }
    }
    return true;
  }

  template <typename DetectionT, typename DetectionBuilder>
  std::optional<std::vector<DetectionT>> DecodeHailoDetections(
      const RawOutputSlot& slot, DetectionBuilder&& build_detection) const
  {
    std::vector<DetectionT> detections;
    detections.reserve(128);

    for (size_t index = 0; index < hailo_output_infos_.size(); ++index)
    {
      const auto& info = hailo_output_infos_[index];
      const auto& descriptor = hailo_output_descriptors_[index];
      const auto& buffer = slot.buffers_[index];
      const auto& anchor_mul = AnchorMulForShape(info);
      bool ok = false;
      switch (descriptor.format_type)
      {
        case HAILO_FORMAT_TYPE_UINT8:
        {
          ok = DecodeOneHailoOutput(
              info, std::get<HailoPageVector<uint8_t>>(buffer.storage),
              descriptor.quant_infos, std::forward<DetectionBuilder>(build_detection),
              detections, anchor_mul);
          break;
        }
        case HAILO_FORMAT_TYPE_UINT16:
        {
          ok = DecodeOneHailoOutput(
              info, std::get<HailoPageVector<uint16_t>>(buffer.storage),
              descriptor.quant_infos, std::forward<DetectionBuilder>(build_detection),
              detections, anchor_mul);
          break;
        }
        case HAILO_FORMAT_TYPE_FLOAT32:
        {
          ok = DecodeOneHailoOutput(
              info, std::get<HailoPageVector<float>>(buffer.storage),
              descriptor.quant_infos, std::forward<DetectionBuilder>(build_detection),
              detections, anchor_mul);
          break;
        }
        default:
          XR_LOG_ERROR("ArmorDetector unsupported Hailo output format in decode path: %d",
                       static_cast<int>(descriptor.format_type));
          ok = false;
          break;
      }
      if (!ok)
      {
        return std::nullopt;
      }
    }

    return detections;
  }

  bool InferHailoRt(const cv::Mat& input, cv::Mat& output)
  {
    cv::Mat continuous_input = input;
    if (!continuous_input.isContinuous())
    {
      continuous_input = input.clone();
    }

    HailoRawTimingSnapshot infer_timing{};
    if (!InferRaw(continuous_input, compatibility_raw_output_slot_, infer_timing))
    {
      last_hailo_timing_ = {};
      return false;
    }

    HailoDecodeTimingSnapshot decode_timing{};
    if (!DecodeRaw(compatibility_raw_output_slot_, output, decode_timing))
    {
      last_hailo_timing_ = {};
      return false;
    }

    ++hailo_timing_count_;
    last_hailo_timing_ = {
        true,
        infer_timing.infer_ms,
        decode_timing.tail_ms,
        infer_timing.infer_ms + decode_timing.tail_ms,
    };
    if (hailo_timing_count_ <= 5U || (hailo_timing_count_ % 30U) == 0U)
    {
      XR_LOG_INFO(
          "ArmorDetector Hailo timing count=%u infer_ms=%.3f tail_ms=%.3f total_ms=%.3f",
          hailo_timing_count_, last_hailo_timing_.infer_ms,
          last_hailo_timing_.tail_ms, last_hailo_timing_.total_ms);
    }
    return true;
  }

  bool BuildInt8HailoOutput(const RawOutputSlot& slot, cv::Mat& output) const
  {
    if (hailo_output_infos_.size() == 1U)
    {
      const auto& descriptor = hailo_output_descriptors_.front();
      const auto& buffer = slot.buffers_.front();
      switch (descriptor.format_type)
      {
        case HAILO_FORMAT_TYPE_UINT8:
          return CopySingleInt8OutputToMatrix(
              std::get<HailoPageVector<uint8_t>>(buffer.storage),
              descriptor.quant_infos, output);
        case HAILO_FORMAT_TYPE_UINT16:
          return CopySingleInt8OutputToMatrix(
              std::get<HailoPageVector<uint16_t>>(buffer.storage),
              descriptor.quant_infos, output);
        case HAILO_FORMAT_TYPE_FLOAT32:
          return CopySingleInt8OutputToMatrix(
              std::get<HailoPageVector<float>>(buffer.storage),
              descriptor.quant_infos, output);
        default:
          XR_LOG_ERROR("ArmorDetector unsupported int8 Hailo output format in single-output path: %d",
                       static_cast<int>(descriptor.format_type));
          return false;
      }
    }

    if (hailo_output_infos_.size() == 6U)
    {
      switch (hailo_output_descriptors_.front().format_type)
      {
        case HAILO_FORMAT_TYPE_UINT8:
          return FuseInt8HostTailOutputs<uint8_t>(slot, output);
        case HAILO_FORMAT_TYPE_UINT16:
          return FuseInt8HostTailOutputs<uint16_t>(slot, output);
        case HAILO_FORMAT_TYPE_FLOAT32:
          return FuseInt8HostTailOutputs<float>(slot, output);
        default:
          XR_LOG_ERROR("ArmorDetector unsupported int8 host-tail format: %d",
                       static_cast<int>(hailo_output_descriptors_.front().format_type));
          return false;
      }
    }

    XR_LOG_ERROR("ArmorDetector unsupported int8 Hailo output topology: outputs=%zu",
                 hailo_output_infos_.size());
    return false;
  }

  bool BuildInt16HailoOutput(const RawOutputSlot& slot, cv::Mat& output) const
  {
    output.create(CandidateCount(), OutputWidth(), CV_32F);
    int row = 0;
    ++hailo_tail_call_count_;
    if (hailo_tail_call_count_ <= 5U || (hailo_tail_call_count_ % 30U) == 1U)
    {
      XR_LOG_INFO("ArmorDetector Hailo native-tail call=%llu begin",
                  static_cast<unsigned long long>(hailo_tail_call_count_));
    }

    for (size_t index = 0; index < hailo_output_infos_.size(); ++index)
    {
      const auto& info = hailo_output_infos_[index];
      const auto& descriptor = hailo_output_descriptors_[index];
      const auto& buffer = slot.buffers_[index];
      const int height = static_cast<int>(info.shape.height);
      const int width = static_cast<int>(info.shape.width);
      const int features = static_cast<int>(info.shape.features);
      const auto& anchor_mul = AnchorMulForShape(info);
      (void)height;
      (void)width;
      (void)features;

      bool ok = false;
      switch (descriptor.format_type)
      {
        case HAILO_FORMAT_TYPE_UINT8:
        {
          ok = FuseOneHailoOutput(info, std::get<HailoPageVector<uint8_t>>(buffer.storage),
                                  descriptor.quant_infos, output, row, anchor_mul);
          break;
        }
        case HAILO_FORMAT_TYPE_UINT16:
        {
          ok = FuseOneHailoOutput(info, std::get<HailoPageVector<uint16_t>>(buffer.storage),
                                  descriptor.quant_infos, output, row, anchor_mul);
          break;
        }
        case HAILO_FORMAT_TYPE_FLOAT32:
        {
          ok = FuseOneHailoOutput(info, std::get<HailoPageVector<float>>(buffer.storage),
                                  descriptor.quant_infos, output, row, anchor_mul);
          break;
        }
        default:
          XR_LOG_ERROR("ArmorDetector unsupported Hailo output format in fuse path: %d",
                       static_cast<int>(descriptor.format_type));
          ok = false;
          break;
      }
      if (!ok)
      {
        return false;
      }
    }

    if (row != CandidateCount())
    {
      XR_LOG_ERROR("ArmorDetector Hailo fused row count mismatch: got=%d expected=%d",
                   row, CandidateCount());
      output.release();
      return false;
    }
    if (hailo_tail_call_count_ <= 5U || (hailo_tail_call_count_ % 30U) == 1U)
    {
      XR_LOG_INFO("ArmorDetector Hailo native-tail call=%llu done rows=%d cols=%d",
                  static_cast<unsigned long long>(hailo_tail_call_count_), output.rows,
                  output.cols);
    }
    LogHailoFuseSnapshot(output);
    return true;
  }

  void LogHailoFuseSnapshot(const cv::Mat& output) const
  {
    if (hailo_fuse_snapshot_count_ >= 3U || output.empty() || output.rows <= 0)
    {
      return;
    }
    ++hailo_fuse_snapshot_count_;

    const int preview_rows = std::min(output.rows, 3);
    for (int row = 0; row < preview_rows; ++row)
    {
      const float* data = output.ptr<float>(row);
      const detail::ModelOutputView output_view(output, detail::int16_candidate_count,
                                                detail::int16_output_width);
      const int raw_color_id =
          detail::ArgMaxOutputRange(output_view, row, 9, 13);
      const int raw_class_id =
          detail::ArgMaxOutputRange(output_view, row, 13, 22);
      XR_LOG_INFO(
          "ArmorDetector Hailo fused row=%d xy0=(%.3f,%.3f) xy1=(%.3f,%.3f) obj_logit=%.3f color_id=%d class_id=%d",
          row, data[0], data[1], data[2], data[3], data[8], raw_color_id,
          raw_class_id);
    }
  }
#endif

  DetectorBackendKind backend_kind_{DetectorBackendKind::NONE};
  infer::ModelLine model_line_{infer::ModelLine::INT16};
  std::string backend_name_{"NONE"};
  std::string hailort_hef_path_{};
  bool model_ready_{false};
  NetworkInputShape input_shape_{};
  HailoTimingSnapshot last_hailo_timing_{};

#if defined(ARMOR_DETECTOR_HAVE_HAILORT)
  std::unique_ptr<hailort::VDevice> hailo_vdevice_{};
  std::shared_ptr<hailort::InferModel> hailo_infer_model_{};
  std::unique_ptr<hailort::ConfiguredInferModel> hailo_configured_infer_model_{};
  std::vector<hailo_vstream_info_t> hailo_output_infos_{};
  std::string hailo_input_name_{};
  size_t hailo_input_frame_size_{0U};
  std::vector<HailoOutputDescriptor> hailo_output_descriptors_{};
  RawOutputSlot compatibility_raw_output_slot_{};
  std::size_t hailo_async_queue_size_{0U};
  uint64_t descriptor_generation_{0U};
  uint32_t hailo_timing_count_{0};
  uint64_t hailo_infer_call_count_{0};
  mutable std::string hailo_dump_active_prefix_{};
  mutable bool hailo_dumped_{false};
  mutable uint64_t hailo_tail_call_count_{0};
  mutable uint32_t hailo_fuse_snapshot_count_{0};
#endif
};

}  // namespace armor_detector_detail
