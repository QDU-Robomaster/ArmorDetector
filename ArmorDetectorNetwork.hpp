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
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

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
    backend_kind_ = DetectorBackendKind::HAILORT;
    return ConfigureHailoRt(hailort_hef_path_.c_str());
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

  [[nodiscard]] HailoTimingSnapshot LastHailoTiming() const
  {
    return last_hailo_timing_;
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
    return InferHailoRt(input, output);
  }

  template <typename DetectionT, typename DetectionBuilder>
  std::optional<std::vector<DetectionT>> InferHailoDetections(
      const cv::Mat& input, DetectionBuilder&& build_detection)
  {
#if defined(ARMOR_DETECTOR_HAVE_HAILORT)
    if (!model_ready_ || backend_kind_ != DetectorBackendKind::HAILORT ||
        input.empty())
    {
      return std::nullopt;
    }
    if (input.type() != CV_8UC3)
    {
      XR_LOG_ERROR("ArmorDetector Hailo input type invalid: %d", input.type());
      return std::nullopt;
    }

    cv::Mat continuous_input = input;
    if (!continuous_input.isContinuous())
    {
      continuous_input = input.clone();
    }

    std::map<std::string, hailort::MemoryView> input_views;
    std::map<std::string, hailort::MemoryView> output_views;
    const auto input_name = hailo_input_params_.begin()->first;
    input_views.emplace(input_name,
                        hailort::MemoryView(continuous_input.data,
                                            continuous_input.total() *
                                                continuous_input.elemSize()));

    for (auto& [name, buffer] : hailo_output_buffers_)
    {
      switch (buffer.format_type)
      {
        case HAILO_FORMAT_TYPE_UINT8:
        {
          auto& storage = std::get<std::vector<uint8_t>>(buffer.storage);
          output_views.emplace(name, hailort::MemoryView(storage.data(), storage.size()));
          break;
        }
        case HAILO_FORMAT_TYPE_UINT16:
        {
          auto& storage = std::get<std::vector<uint16_t>>(buffer.storage);
          output_views.emplace(
              name, hailort::MemoryView(storage.data(), storage.size() * sizeof(uint16_t)));
          break;
        }
        case HAILO_FORMAT_TYPE_FLOAT32:
        {
          auto& storage = std::get<std::vector<float>>(buffer.storage);
          output_views.emplace(
              name, hailort::MemoryView(storage.data(), storage.size() * sizeof(float)));
          break;
        }
        default:
          XR_LOG_ERROR("ArmorDetector unsupported Hailo output format in infer path: %d",
                       static_cast<int>(buffer.format_type));
          return std::nullopt;
      }
    }

    try
    {
      const auto infer_begin = std::chrono::steady_clock::now();
      const auto status = hailo_infer_pipeline_->infer(input_views, output_views, 1);
      const auto infer_end = std::chrono::steady_clock::now();
      if (HAILO_SUCCESS != status)
      {
        XR_LOG_ERROR("ArmorDetector Hailo inference failed status=%d",
                     static_cast<int>(status));
        return std::nullopt;
      }

      const auto tail_begin = std::chrono::steady_clock::now();
      auto detections = DecodeHailoDetections<DetectionT>(
          std::forward<DetectionBuilder>(build_detection));
      const auto tail_end = std::chrono::steady_clock::now();
      if (detections.has_value())
      {
        ++hailo_timing_count_;
        const double infer_ms =
            std::chrono::duration<double, std::milli>(infer_end - infer_begin).count();
        const double tail_ms =
            std::chrono::duration<double, std::milli>(tail_end - tail_begin).count();
        if (hailo_timing_count_ <= 5U || (hailo_timing_count_ % 30U) == 0U)
        {
          XR_LOG_INFO(
              "ArmorDetector Hailo direct timing count=%u infer_ms=%.3f tail_ms=%.3f total_ms=%.3f",
              hailo_timing_count_, infer_ms, tail_ms, infer_ms + tail_ms);
        }
      }
      return detections;
    }
    catch (const std::exception& exception)
    {
      XR_LOG_ERROR("ArmorDetector Hailo inference exception: %s",
                   exception.what());
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
    hailo_infer_call_count_ = 0;

#if defined(ARMOR_DETECTOR_HAVE_HAILORT)
    hailo_output_infos_.clear();
    hailo_output_buffers_.clear();
    hailo_output_bytes_.clear();
    hailo_network_group_.reset();
    hailo_activation_.reset();
    hailo_infer_pipeline_.reset();
    hailo_vdevice_.reset();
    hailo_hef_.reset();
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
  struct HailoOutputBuffer
  {
    hailo_format_type_t format_type{HAILO_FORMAT_TYPE_FLOAT32};
    std::variant<std::vector<uint8_t>, std::vector<uint16_t>, std::vector<float>>
        storage{};
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
      auto hef_expected = hailort::Hef::create(std::string(hef_path));
      if (!hef_expected.has_value())
      {
        XR_LOG_ERROR("ArmorDetector failed to load HEF %s status=%d",
                     hef_path, static_cast<int>(hef_expected.status()));
        return false;
      }
      hailo_hef_ = std::make_unique<hailort::Hef>(hef_expected.release());

      auto vdevice_expected = hailort::VDevice::create();
      if (!vdevice_expected.has_value())
      {
        XR_LOG_ERROR("ArmorDetector failed to create Hailo VDevice status=%d",
                     static_cast<int>(vdevice_expected.status()));
        return false;
      }
      hailo_vdevice_ = vdevice_expected.release();

      auto configure_params_expected =
          hailo_vdevice_->create_configure_params(*hailo_hef_);
      if (!configure_params_expected.has_value())
      {
        XR_LOG_ERROR("ArmorDetector failed to create Hailo configure params status=%d",
                     static_cast<int>(configure_params_expected.status()));
        return false;
      }

      auto network_groups_expected =
          hailo_vdevice_->configure(*hailo_hef_,
                                    configure_params_expected.release());
      if (!network_groups_expected.has_value() ||
          network_groups_expected.value().empty())
      {
        XR_LOG_ERROR("ArmorDetector failed to configure Hailo network group status=%d",
                     static_cast<int>(network_groups_expected.status()));
        return false;
      }
      auto network_groups = network_groups_expected.release();
      hailo_network_group_ = network_groups.front();

      auto input_infos_expected = hailo_network_group_->get_input_vstream_infos();
      auto output_infos_expected =
          hailo_network_group_->get_output_vstream_infos();
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

      auto input_params_expected =
          hailo_network_group_->make_input_vstream_params(
              false, HAILO_FORMAT_TYPE_AUTO,
              HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
              HAILO_DEFAULT_VSTREAM_QUEUE_SIZE, "");
      auto output_params_expected =
          hailo_network_group_->make_output_vstream_params(
              false, HAILO_FORMAT_TYPE_AUTO,
              HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
              HAILO_DEFAULT_VSTREAM_QUEUE_SIZE, "");
      if (!input_params_expected.has_value() || !output_params_expected.has_value())
      {
        XR_LOG_ERROR("ArmorDetector failed to create Hailo vstream params");
        return false;
      }
      hailo_input_params_ = input_params_expected.release();
      hailo_output_params_ = output_params_expected.release();

      if (!hailo_network_group_->is_scheduled())
      {
        auto activation_expected = hailo_network_group_->activate();
        if (!activation_expected.has_value())
        {
          XR_LOG_ERROR("ArmorDetector failed to activate Hailo network group status=%d",
                       static_cast<int>(activation_expected.status()));
          return false;
        }
        hailo_activation_ = activation_expected.release();
      }

      auto infer_pipeline_expected =
          hailort::InferVStreams::create(*hailo_network_group_,
                                         hailo_input_params_, hailo_output_params_);
      if (!infer_pipeline_expected.has_value())
      {
        XR_LOG_ERROR("ArmorDetector failed to create Hailo infer pipeline status=%d",
                     static_cast<int>(infer_pipeline_expected.status()));
        return false;
      }
      hailo_infer_pipeline_ =
          std::make_unique<hailort::InferVStreams>(
              infer_pipeline_expected.release());

      if (!PrepareHailoOutputBuffers())
      {
        return false;
      }

      backend_name_ = "HAILORT";
      backend_kind_ = DetectorBackendKind::HAILORT;
      model_ready_ = true;
      XR_LOG_PASS(
          "ArmorDetector loaded HailoRT HEF=%s input=%dx%d outputs=%zu",
          hef_path, input_shape_.width, input_shape_.height,
          hailo_output_infos_.size());
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

  bool PrepareHailoOutputBuffers()
  {
    hailo_output_buffers_.clear();
    hailo_output_bytes_.clear();
    for (const auto& info : hailo_output_infos_)
    {
      auto output_vstream_expected =
          hailo_infer_pipeline_->get_output_by_name(info.name);
      if (!output_vstream_expected.has_value())
      {
        XR_LOG_ERROR("ArmorDetector cannot find Hailo output vstream %s",
                     info.name);
        return false;
      }
      const auto& output_vstream = output_vstream_expected.value().get();
      const size_t frame_size = output_vstream.get_frame_size();
      const auto format = output_vstream.get_user_buffer_format();

      HailoOutputBuffer buffer{};
      buffer.format_type = format.type;
      buffer.quant_infos = output_vstream.get_quant_infos();

      switch (format.type)
      {
        case HAILO_FORMAT_TYPE_UINT8:
          buffer.storage = std::vector<uint8_t>(frame_size, 0U);
          break;
        case HAILO_FORMAT_TYPE_UINT16:
          if ((frame_size % sizeof(uint16_t)) != 0U)
          {
            XR_LOG_ERROR("ArmorDetector Hailo output frame size invalid for UINT16: %s size=%zu",
                         info.name, frame_size);
            return false;
          }
          buffer.storage =
              std::vector<uint16_t>(frame_size / sizeof(uint16_t), 0U);
          break;
        case HAILO_FORMAT_TYPE_FLOAT32:
          if ((frame_size % sizeof(float)) != 0U)
          {
            XR_LOG_ERROR("ArmorDetector Hailo output frame size invalid for FLOAT32: %s size=%zu",
                         info.name, frame_size);
            return false;
          }
          buffer.storage =
              std::vector<float>(frame_size / sizeof(float), 0.0F);
          break;
        default:
          XR_LOG_ERROR("ArmorDetector unsupported Hailo output format type=%d for %s",
                       static_cast<int>(format.type), info.name);
          return false;
      }

      hailo_output_bytes_.emplace(std::string(info.name), frame_size);
      hailo_output_buffers_.emplace(std::string(info.name), std::move(buffer));
    }
    return true;
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
  float LoadHailoValue(const std::vector<T>& storage, size_t index,
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
                          const std::vector<T>& storage,
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
  static float LoadQuantizedValue(const std::vector<T>& storage, size_t index,
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
  bool CopySingleInt8OutputToMatrix(const std::vector<T>& storage,
                                    const std::vector<hailo_quant_info_t>& quant_infos,
                                    cv::Mat& output) const
  {
    output.create(detail::int8_output_width, detail::int8_candidate_count, CV_32F);
    if (storage.size() < static_cast<size_t>(detail::int8_output_width *
                                             detail::int8_candidate_count))
    {
      XR_LOG_ERROR("ArmorDetector int8 single-output buffer too small: got=%zu expected=%d",
                   storage.size(), detail::int8_output_width *
                                       detail::int8_candidate_count);
      output.release();
      return false;
    }

    Eigen::Map<detail::RowMajorArrayXXf> output_matrix(
        output.ptr<float>(), detail::int8_output_width,
        detail::int8_candidate_count);
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        source_matrix(storage.data(), detail::int8_output_width,
                      detail::int8_candidate_count);

    if (quant_infos.empty())
    {
      output_matrix = source_matrix.template cast<float>().array();
      return true;
    }

    if (quant_infos.size() == 1U)
    {
      const auto& qi = quant_infos.front();
      output_matrix =
          (source_matrix.template cast<float>().array() - qi.qp_zp) * qi.qp_scale;
      return true;
    }

    for (int row = 0; row < detail::int8_output_width; ++row)
    {
      const auto& qi = quant_infos[static_cast<std::size_t>(row) % quant_infos.size()];
      output_matrix.row(row) =
          (source_matrix.row(row).template cast<float>().array() - qi.qp_zp) *
          qi.qp_scale;
    }
    return true;
  }

  template <typename T>
  bool FuseInt8HostTailOutputs(
      const std::unordered_map<std::string, HailoOutputBuffer>& buffers,
      cv::Mat& output) const
  {
    const auto read_tensor =
        [&buffers](const char* name) -> const std::vector<T>* {
          const auto it = buffers.find(name);
          if (it == buffers.end())
          {
            return nullptr;
          }
          return &std::get<std::vector<T>>(it->second.storage);
        };

    const auto* conv45 = read_tensor(kInt8HostTailOutputNames[0]);
    const auto* concat11 = read_tensor(kInt8HostTailOutputNames[1]);
    const auto* conv59 = read_tensor(kInt8HostTailOutputNames[2]);
    const auto* concat14 = read_tensor(kInt8HostTailOutputNames[3]);
    const auto* conv72 = read_tensor(kInt8HostTailOutputNames[4]);
    const auto* concat16 = read_tensor(kInt8HostTailOutputNames[5]);
    if (conv45 == nullptr || concat11 == nullptr || conv59 == nullptr ||
        concat14 == nullptr || conv72 == nullptr || concat16 == nullptr)
    {
      XR_LOG_ERROR("ArmorDetector missing int8 host-tail outputs");
      return false;
    }

    const auto& q45 = buffers.at(kInt8HostTailOutputNames[0]).quant_infos;
    const auto& q11 = buffers.at(kInt8HostTailOutputNames[1]).quant_infos;
    const auto& q59 = buffers.at(kInt8HostTailOutputNames[2]).quant_infos;
    const auto& q14 = buffers.at(kInt8HostTailOutputNames[3]).quant_infos;
    const auto& q72 = buffers.at(kInt8HostTailOutputNames[4]).quant_infos;
    const auto& q16_p5 = buffers.at(kInt8HostTailOutputNames[5]).quant_infos;

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
      const std::vector<T>* conv8 = scale == 0 ? conv45 : (scale == 1 ? conv59 : conv72);
      const std::vector<T>* concat16_tensor =
          scale == 0 ? concat11 : (scale == 1 ? concat14 : concat16);
      const auto& q8 = scale == 0 ? q45 : (scale == 1 ? q59 : q72);
      const auto& q16 =
          scale == 0 ? q11 : (scale == 1 ? q14 : q16_p5);
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
          dst(8) = concat16_block(0);
          dst(17) = concat16_block(1);
          dst(18) = concat16_block(2);
          dst.segment(9, 8) = concat16_block.segment(4, 8);
          dst.segment(19, 2) = concat16_block.segment(12, 2);

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
                                     const std::vector<T>& storage,
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

  void MaybeDumpHailoHeadTensors(uint64_t infer_index) const
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

    static std::string active_prefix;
    static bool dumped = false;
    if (active_prefix != prefix)
    {
      active_prefix = prefix;
      dumped = false;
    }
    if (dumped)
    {
      return;
    }

    for (const auto& info : hailo_output_infos_)
    {
      const auto buffer_it = hailo_output_buffers_.find(info.name);
      if (buffer_it == hailo_output_buffers_.end())
      {
        XR_LOG_ERROR("ArmorDetector missing Hailo output buffer for dump: %s",
                     info.name);
        return;
      }

      const auto& buffer = buffer_it->second;
      const std::string path =
          active_prefix + "_" + SanitizeHailoDumpName(info.name) + ".f32bin";
      bool ok = false;
      switch (buffer.format_type)
      {
        case HAILO_FORMAT_TYPE_UINT8:
          ok = DumpOneHailoOutputDequantized(
              path, info, std::get<std::vector<uint8_t>>(buffer.storage),
              buffer.quant_infos);
          break;
        case HAILO_FORMAT_TYPE_UINT16:
          ok = DumpOneHailoOutputDequantized(
              path, info, std::get<std::vector<uint16_t>>(buffer.storage),
              buffer.quant_infos);
          break;
        case HAILO_FORMAT_TYPE_FLOAT32:
          ok = DumpOneHailoOutputDequantized(
              path, info, std::get<std::vector<float>>(buffer.storage),
              buffer.quant_infos);
          break;
        default:
          XR_LOG_ERROR("ArmorDetector unsupported Hailo output format in dump path: %d",
                       static_cast<int>(buffer.format_type));
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
    dumped = true;
  }

  template <typename DetectionT, typename DetectionBuilder, typename T>
  bool DecodeOneHailoOutput(const hailo_vstream_info_t& info,
                            const std::vector<T>& storage,
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
      DetectionBuilder&& build_detection) const
  {
    std::vector<DetectionT> detections;
    detections.reserve(128);

    for (const auto& info : hailo_output_infos_)
    {
      const auto buffer_it = hailo_output_buffers_.find(info.name);
      if (buffer_it == hailo_output_buffers_.end())
      {
        XR_LOG_ERROR("ArmorDetector missing Hailo output buffer for %s",
                     info.name);
        return std::nullopt;
      }

      const auto& buffer = buffer_it->second;
      const auto& anchor_mul = AnchorMulForShape(info);
      bool ok = false;
      switch (buffer.format_type)
      {
        case HAILO_FORMAT_TYPE_UINT8:
        {
          ok = DecodeOneHailoOutput(
              info, std::get<std::vector<uint8_t>>(buffer.storage),
              buffer.quant_infos, std::forward<DetectionBuilder>(build_detection),
              detections, anchor_mul);
          break;
        }
        case HAILO_FORMAT_TYPE_UINT16:
        {
          ok = DecodeOneHailoOutput(
              info, std::get<std::vector<uint16_t>>(buffer.storage),
              buffer.quant_infos, std::forward<DetectionBuilder>(build_detection),
              detections, anchor_mul);
          break;
        }
        case HAILO_FORMAT_TYPE_FLOAT32:
        {
          ok = DecodeOneHailoOutput(
              info, std::get<std::vector<float>>(buffer.storage),
              buffer.quant_infos, std::forward<DetectionBuilder>(build_detection),
              detections, anchor_mul);
          break;
        }
        default:
          XR_LOG_ERROR("ArmorDetector unsupported Hailo output format in decode path: %d",
                       static_cast<int>(buffer.format_type));
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
    if (input.type() != CV_8UC3)
    {
      XR_LOG_ERROR("ArmorDetector Hailo input type invalid: %d", input.type());
      return false;
    }

    cv::Mat continuous_input = input;
    if (!continuous_input.isContinuous())
    {
      continuous_input = input.clone();
    }

    std::map<std::string, hailort::MemoryView> input_views;
    std::map<std::string, hailort::MemoryView> output_views;
    const auto input_name = hailo_input_params_.begin()->first;
    input_views.emplace(input_name,
                        hailort::MemoryView(continuous_input.data,
                                            continuous_input.total() *
                                                continuous_input.elemSize()));

    for (auto& [name, buffer] : hailo_output_buffers_)
    {
      switch (buffer.format_type)
      {
        case HAILO_FORMAT_TYPE_UINT8:
        {
          auto& storage = std::get<std::vector<uint8_t>>(buffer.storage);
          output_views.emplace(name, hailort::MemoryView(storage.data(), storage.size()));
          break;
        }
        case HAILO_FORMAT_TYPE_UINT16:
        {
          auto& storage = std::get<std::vector<uint16_t>>(buffer.storage);
          output_views.emplace(
              name, hailort::MemoryView(storage.data(), storage.size() * sizeof(uint16_t)));
          break;
        }
        case HAILO_FORMAT_TYPE_FLOAT32:
        {
          auto& storage = std::get<std::vector<float>>(buffer.storage);
          output_views.emplace(
              name, hailort::MemoryView(storage.data(), storage.size() * sizeof(float)));
          break;
        }
        default:
          XR_LOG_ERROR("ArmorDetector unsupported Hailo output format in infer path: %d",
                       static_cast<int>(buffer.format_type));
          return false;
      }
    }

    try
    {
      ++hailo_infer_call_count_;
      const auto infer_begin = std::chrono::steady_clock::now();
      const auto status = hailo_infer_pipeline_->infer(input_views, output_views, 1);
      const auto infer_end = std::chrono::steady_clock::now();
      if (HAILO_SUCCESS != status)
      {
        XR_LOG_ERROR("ArmorDetector Hailo inference failed status=%d",
                     static_cast<int>(status));
        return false;
      }
      MaybeDumpHailoHeadTensors(hailo_infer_call_count_);
      const auto tail_begin = std::chrono::steady_clock::now();
      const bool ok =
          (model_line_ == infer::ModelLine::INT8) ? BuildInt8HailoOutput(output)
                                                  : BuildInt16HailoOutput(output);
      const auto tail_end = std::chrono::steady_clock::now();
      if (ok)
      {
        ++hailo_timing_count_;
        const double infer_ms =
            std::chrono::duration<double, std::milli>(infer_end - infer_begin).count();
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
              "ArmorDetector Hailo timing count=%u infer_ms=%.3f tail_ms=%.3f total_ms=%.3f",
              hailo_timing_count_, infer_ms, tail_ms, infer_ms + tail_ms);
        }
      }
      else
      {
        last_hailo_timing_ = {};
      }
      return ok;
    }
    catch (const std::exception& exception)
    {
      XR_LOG_ERROR("ArmorDetector Hailo inference exception: %s",
                   exception.what());
      return false;
    }
  }

  bool BuildInt8HailoOutput(cv::Mat& output)
  {
    if (hailo_output_infos_.size() == 1U)
    {
      const auto& info = hailo_output_infos_.front();
      const auto buffer_it = hailo_output_buffers_.find(info.name);
      if (buffer_it == hailo_output_buffers_.end())
      {
        XR_LOG_ERROR("ArmorDetector missing int8 Hailo output buffer for %s",
                     info.name);
        return false;
      }
      const auto& buffer = buffer_it->second;
      switch (buffer.format_type)
      {
        case HAILO_FORMAT_TYPE_UINT8:
          return CopySingleInt8OutputToMatrix(
              std::get<std::vector<uint8_t>>(buffer.storage),
              buffer.quant_infos, output);
        case HAILO_FORMAT_TYPE_UINT16:
          return CopySingleInt8OutputToMatrix(
              std::get<std::vector<uint16_t>>(buffer.storage),
              buffer.quant_infos, output);
        case HAILO_FORMAT_TYPE_FLOAT32:
          return CopySingleInt8OutputToMatrix(
              std::get<std::vector<float>>(buffer.storage),
              buffer.quant_infos, output);
        default:
          XR_LOG_ERROR("ArmorDetector unsupported int8 Hailo output format in single-output path: %d",
                       static_cast<int>(buffer.format_type));
          return false;
      }
    }

    if (hailo_output_infos_.size() == 6U)
    {
      const auto first_it = hailo_output_buffers_.begin();
      if (first_it == hailo_output_buffers_.end())
      {
        XR_LOG_ERROR("ArmorDetector missing int8 host-tail buffers");
        return false;
      }
      switch (first_it->second.format_type)
      {
        case HAILO_FORMAT_TYPE_UINT8:
          return FuseInt8HostTailOutputs<uint8_t>(hailo_output_buffers_, output);
        case HAILO_FORMAT_TYPE_UINT16:
          return FuseInt8HostTailOutputs<uint16_t>(hailo_output_buffers_, output);
        case HAILO_FORMAT_TYPE_FLOAT32:
          return FuseInt8HostTailOutputs<float>(hailo_output_buffers_, output);
        default:
          XR_LOG_ERROR("ArmorDetector unsupported int8 host-tail format: %d",
                       static_cast<int>(first_it->second.format_type));
          return false;
      }
    }

    XR_LOG_ERROR("ArmorDetector unsupported int8 Hailo output topology: outputs=%zu",
                 hailo_output_infos_.size());
    return false;
  }

  bool BuildInt16HailoOutput(cv::Mat& output)
  {
    output.create(CandidateCount(), OutputWidth(), CV_32F);
    int row = 0;
    static uint64_t tail_call_count = 0;
    ++tail_call_count;
    if (tail_call_count <= 5U || (tail_call_count % 30U) == 1U)
    {
      XR_LOG_INFO("ArmorDetector Hailo native-tail call=%llu begin",
                  static_cast<unsigned long long>(tail_call_count));
    }

    for (const auto& info : hailo_output_infos_)
    {
      const auto buffer_it = hailo_output_buffers_.find(info.name);
      if (buffer_it == hailo_output_buffers_.end())
      {
        XR_LOG_ERROR("ArmorDetector missing Hailo output buffer for %s",
                     info.name);
        output.release();
        return false;
      }

      const auto& buffer = buffer_it->second;
      const int height = static_cast<int>(info.shape.height);
      const int width = static_cast<int>(info.shape.width);
      const int features = static_cast<int>(info.shape.features);
      const auto& anchor_mul = AnchorMulForShape(info);
      (void)height;
      (void)width;
      (void)features;

      bool ok = false;
      switch (buffer.format_type)
      {
        case HAILO_FORMAT_TYPE_UINT8:
        {
          ok = FuseOneHailoOutput(info, std::get<std::vector<uint8_t>>(buffer.storage),
                                  buffer.quant_infos, output, row, anchor_mul);
          break;
        }
        case HAILO_FORMAT_TYPE_UINT16:
        {
          ok = FuseOneHailoOutput(info, std::get<std::vector<uint16_t>>(buffer.storage),
                                  buffer.quant_infos, output, row, anchor_mul);
          break;
        }
        case HAILO_FORMAT_TYPE_FLOAT32:
        {
          ok = FuseOneHailoOutput(info, std::get<std::vector<float>>(buffer.storage),
                                  buffer.quant_infos, output, row, anchor_mul);
          break;
        }
        default:
          XR_LOG_ERROR("ArmorDetector unsupported Hailo output format in fuse path: %d",
                       static_cast<int>(buffer.format_type));
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
    if (tail_call_count <= 5U || (tail_call_count % 30U) == 1U)
    {
      XR_LOG_INFO("ArmorDetector Hailo native-tail call=%llu done rows=%d cols=%d",
                  static_cast<unsigned long long>(tail_call_count), output.rows,
                  output.cols);
    }
    LogHailoFuseSnapshot(output);
    return true;
  }

  void LogHailoFuseSnapshot(const cv::Mat& output)
  {
    static uint32_t snapshot_count = 0;
    if (snapshot_count >= 3U || output.empty() || output.rows <= 0)
    {
      return;
    }
    ++snapshot_count;

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
  std::unique_ptr<hailort::Hef> hailo_hef_{};
  std::unique_ptr<hailort::VDevice> hailo_vdevice_{};
  std::shared_ptr<hailort::ConfiguredNetworkGroup> hailo_network_group_{};
  std::unique_ptr<hailort::ActivatedNetworkGroup> hailo_activation_{};
  std::unique_ptr<hailort::InferVStreams> hailo_infer_pipeline_{};
  std::map<std::string, hailo_vstream_params_t> hailo_input_params_{};
  std::map<std::string, hailo_vstream_params_t> hailo_output_params_{};
  std::vector<hailo_vstream_info_t> hailo_output_infos_{};
  std::unordered_map<std::string, size_t> hailo_output_bytes_{};
  std::unordered_map<std::string, HailoOutputBuffer> hailo_output_buffers_{};
  uint32_t hailo_timing_count_{0};
  uint64_t hailo_infer_call_count_{0};
#endif
};

}  // namespace armor_detector_detail
