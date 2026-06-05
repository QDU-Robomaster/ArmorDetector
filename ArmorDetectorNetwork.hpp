#pragma once

/**
 * @file ArmorDetectorNetwork.hpp
 * @brief ArmorDetector 的推理后端封装，支持 OpenVINO 与 HailoRT。
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

#include <opencv2/core.hpp>

#if defined(ARMOR_DETECTOR_HAVE_OPENVINO)
#include <openvino/openvino.hpp>
#endif

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
  OPENVINO,
  HAILORT,
};

/**
 * @brief ArmorDetector 推理后端封装。
 *
 * 对外保持单一接口：输入为 resize 后的 RGB8 图像，输出为当前 detector family
 * 对应的 `CV_32F` 候选矩阵。SZU 使用 `20160x22`；SKD 使用 `6720x21`
 * direct-keypoint surface 或其 host-tail 变体。
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
   * @brief 初始化推理后端。
   * @param model_family detector 模型族：SZU / SKD。
   * @param backend_name 后端选择：AUTO_DETECT / OPENVINO / HAILORT。
   * @param model_path OpenVINO ONNX 模型路径。
   * @param device_name OpenVINO 设备名，例如 CPU/GPU/NPU/AUTO:GPU,NPU。
   * @param performance_mode OpenVINO performance hint。
   * @param hailort_hef_path HailoRT HEF 文件路径。
   * @return 后端可用且张量约定检查通过时返回 true。
   */
  bool Configure(const char* model_family, const char* backend_name,
                 const char* model_path,
                 const char* device_name, const char* performance_mode,
                 const char* hailort_hef_path,
                 const char*)
  {
    Reset();

    model_family_ = model_family_from_name(model_family);
    requested_backend_name_ = NormalizeBackendName(backend_name);
    requested_device_name_ = NormalizeDeviceName(device_name);
    performance_mode_name_ = NormalizePerformanceModeName(performance_mode);
    hailort_hef_path_ = NormalizeOptionalPath(hailort_hef_path);
    openvino_model_path_ = NormalizeOptionalPath(model_path);
    backend_kind_ = ResolveBackendKind(requested_backend_name_, hailort_hef_path_);

    switch (backend_kind_)
    {
      case DetectorBackendKind::OPENVINO:
#if defined(ARMOR_DETECTOR_HAVE_OPENVINO)
        return ConfigureOpenVino(openvino_model_path_.c_str(),
                                 requested_device_name_.c_str(),
                                 performance_mode_name_.c_str());
#else
        XR_LOG_ERROR("ArmorDetector OpenVINO backend is unavailable in this build");
        return false;
#endif
      case DetectorBackendKind::HAILORT:
        return ConfigureHailoRt(hailort_hef_path_.c_str());
      case DetectorBackendKind::NONE:
      default:
        XR_LOG_ERROR("ArmorDetector backend is unavailable (requested=%s)",
                     requested_backend_name_.c_str());
        return false;
    }
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
   * @param output 输出候选矩阵，格式与原 OpenVINO decoder 保持一致。
   * @return 成功时返回 true。
   */
  bool Infer(const cv::Mat& input, cv::Mat& output)
  {
    output.release();
    if (!model_ready_ || input.empty())
    {
      return false;
    }

    switch (backend_kind_)
    {
      case DetectorBackendKind::OPENVINO:
#if defined(ARMOR_DETECTOR_HAVE_OPENVINO)
        return InferOpenVino(input, output);
#else
        XR_LOG_ERROR("ArmorDetector OpenVINO inference path is unavailable in this build");
        return false;
#endif
      case DetectorBackendKind::HAILORT:
        return InferHailoRt(input, output);
      case DetectorBackendKind::NONE:
      default:
        XR_LOG_ERROR("ArmorDetector backend is not configured");
        return false;
    }
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
    return model_family_ == ModelFamily::SKD ? skd_candidate_count
                                             : model_candidate_count;
  }

  int OutputWidth() const
  {
    return model_family_ == ModelFamily::SKD ? skd_output_width
                                             : model_output_width;
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
    requested_backend_name_ = "AUTO_DETECT";
    requested_device_name_ = "AUTO_DETECT";
    performance_mode_name_ = "LATENCY";
    openvino_model_path_.clear();
    hailort_hef_path_.clear();
    last_hailo_timing_ = {};
    hailo_infer_call_count_ = 0;

#if defined(ARMOR_DETECTOR_HAVE_OPENVINO)
    available_device_names_.clear();
    device_name_ = "CPU";
    compiled_model_ = ov::CompiledModel();
    infer_request_ = ov::InferRequest();
    output_tensor_ = ov::Tensor();
#endif

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

  static std::string NormalizeBackendName(const char* backend_name)
  {
    if (backend_name == nullptr || backend_name[0] == '\0')
    {
      return "AUTO_DETECT";
    }
    return backend_name;
  }

  static std::string NormalizeDeviceName(const char* device_name)
  {
    if (device_name == nullptr || device_name[0] == '\0')
    {
      return "AUTO_DETECT";
    }
    return device_name;
  }

  static std::string NormalizePerformanceModeName(const char* performance_mode)
  {
    if (performance_mode == nullptr || performance_mode[0] == '\0')
    {
      return "LATENCY";
    }
    return performance_mode;
  }

  [[nodiscard]] bool OutputShapeMatches(const cv::Mat& output) const
  {
    return (output.rows == CandidateCount() &&
            output.cols == OutputWidth()) ||
           (output.rows == OutputWidth() &&
            output.cols == CandidateCount());
  }

  DetectorBackendKind ResolveBackendKind(const std::string& requested_backend,
                                         const std::string& hailort_hef_path) const
  {
    if (requested_backend == "OPENVINO")
    {
#if defined(ARMOR_DETECTOR_HAVE_OPENVINO)
      return DetectorBackendKind::OPENVINO;
#else
      XR_LOG_ERROR("ArmorDetector requested OPENVINO but OpenVINO support is not compiled");
      return DetectorBackendKind::NONE;
#endif
    }

    if (requested_backend == "HAILORT")
    {
#if defined(ARMOR_DETECTOR_HAVE_HAILORT)
      if (hailort_hef_path.empty())
      {
        XR_LOG_ERROR("ArmorDetector requested HAILORT but HEF path is empty");
        return DetectorBackendKind::NONE;
      }
      return DetectorBackendKind::HAILORT;
#else
      XR_LOG_ERROR("ArmorDetector requested HAILORT but HailoRT support is not compiled");
      return DetectorBackendKind::NONE;
#endif
    }

    if (requested_backend == "AUTO_DETECT")
    {
#if defined(ARMOR_DETECTOR_HAVE_HAILORT)
      if (!hailort_hef_path.empty())
      {
        return DetectorBackendKind::HAILORT;
      }
#endif
#if defined(ARMOR_DETECTOR_HAVE_OPENVINO)
      return DetectorBackendKind::OPENVINO;
#endif
#if defined(ARMOR_DETECTOR_HAVE_HAILORT)
      return DetectorBackendKind::HAILORT;
#endif
    }

    XR_LOG_ERROR("ArmorDetector unknown backend %s", requested_backend.c_str());
    return DetectorBackendKind::NONE;
  }

#if defined(ARMOR_DETECTOR_HAVE_OPENVINO)
  static NetworkInputShape ModelInputShape(
      const std::shared_ptr<ov::Model>& model)
  {
    const auto shape = model->input().get_partial_shape();
    if (shape.rank().is_dynamic() || shape.size() != 3U ||
        shape[0].is_dynamic() || shape[1].is_dynamic() ||
        shape[2].is_dynamic() || shape[2].get_length() != 3)
    {
      return {};
    }
    return {static_cast<int>(shape[1].get_length()),
            static_cast<int>(shape[0].get_length())};
  }

  bool ModelOutputShapeSupported(const std::shared_ptr<ov::Model>& model) const
  {
    const auto shape = model->output().get_partial_shape();
    const int candidate_count = CandidateCount();
    const int output_width = OutputWidth();
    if (shape.rank().is_dynamic())
    {
      return false;
    }
    if (shape.size() == 3U && !shape[1].is_dynamic() &&
        !shape[2].is_dynamic())
    {
      return shape[1].get_length() == candidate_count &&
             shape[2].get_length() == output_width;
    }
    if (shape.size() == 2U && !shape[0].is_dynamic() &&
        !shape[1].is_dynamic())
    {
      return (shape[0].get_length() == candidate_count &&
              shape[1].get_length() == output_width) ||
             (shape[0].get_length() == output_width &&
              shape[1].get_length() == candidate_count);
    }
    return false;
  }

  static bool DeviceMatches(const std::string& device, const char* base)
  {
    const std::string base_name(base);
    return device == base_name || device.rfind(base_name + ".", 0) == 0;
  }

  std::string ResolveDeviceName(const std::string& requested_device)
  {
    if (requested_device != "AUTO_DETECT")
    {
      return requested_device;
    }

    for (const char* preferred : {"NPU", "GPU", "CPU"})
    {
      for (const auto& device : available_device_names_)
      {
        if (DeviceMatches(device, preferred))
        {
          return device;
        }
      }
    }

    XR_LOG_WARN(
        "ArmorDetector OpenVINO AUTO_DETECT found no devices, fallback CPU");
    return "CPU";
  }

  std::vector<std::string> QueryAvailableDeviceNames()
  {
    try
    {
      return ov_core_.get_available_devices();
    }
    catch (const std::exception& exception)
    {
      XR_LOG_WARN("ArmorDetector failed to query OpenVINO devices: %s",
                  exception.what());
      return {};
    }
  }

  [[nodiscard]] std::string AvailableDevicesText() const
  {
    if (available_device_names_.empty())
    {
      return "none";
    }

    std::string text;
    for (const auto& device : available_device_names_)
    {
      if (!text.empty())
      {
        text += ",";
      }
      text += device;
    }
    return text;
  }

  [[nodiscard]] bool HasAcceleratorDevice() const
  {
    for (const auto& device : available_device_names_)
    {
      if (!DeviceMatches(device, "CPU"))
      {
        return true;
      }
    }
    return false;
  }

  void LogDeviceSelection() const
  {
    const auto available_devices_text = AvailableDevicesText();
    if (!HasAcceleratorDevice())
    {
      XR_LOG_WARN(
          "ArmorDetector OpenVINO CPU-only runtime; available=%s requested=%s resolved=%s mode=%s",
          available_devices_text.c_str(), requested_device_name_.c_str(),
          device_name_.c_str(), performance_mode_name_.c_str());
      return;
    }

    XR_LOG_INFO(
        "ArmorDetector OpenVINO devices=%s requested=%s resolved=%s mode=%s",
        available_devices_text.c_str(), requested_device_name_.c_str(),
        device_name_.c_str(), performance_mode_name_.c_str());
  }

  static ov::hint::PerformanceMode ParsePerformanceMode(
      const std::string& performance_mode)
  {
    if (performance_mode == "LATENCY")
    {
      return ov::hint::PerformanceMode::LATENCY;
    }
    if (performance_mode == "THROUGHPUT")
    {
      return ov::hint::PerformanceMode::THROUGHPUT;
    }
    if (performance_mode == "CUMULATIVE_THROUGHPUT")
    {
      return ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT;
    }

    XR_LOG_WARN(
        "ArmorDetector unknown OpenVINO performance mode %s, fallback LATENCY",
        performance_mode.c_str());
    return ov::hint::PerformanceMode::LATENCY;
  }

  bool ConfigureOpenVino(const char* model_path, const char* device_name,
                         const char* performance_mode)
  {
    if (model_path == nullptr || model_path[0] == '\0')
    {
      XR_LOG_ERROR("ArmorDetector OpenVINO model path is empty");
      return false;
    }

    available_device_names_ = QueryAvailableDeviceNames();
    device_name_ = ResolveDeviceName(requested_device_name_);
    performance_mode_name_ = NormalizePerformanceModeName(performance_mode);

    try
    {
      LogDeviceSelection();
      auto model = ov_core_.read_model(model_path);
      input_shape_ = ModelInputShape(model);
      if (!IsValidNetworkInputShape(input_shape_))
      {
        XR_LOG_ERROR("ArmorDetector cannot resolve OpenVINO model input shape");
        return false;
      }
      if (!ModelOutputShapeSupported(model))
      {
        XR_LOG_ERROR("ArmorDetector OpenVINO model output shape is unsupported");
        return false;
      }

      compiled_model_ = ov_core_.compile_model(
          model, device_name_,
          ov::hint::performance_mode(
              ParsePerformanceMode(performance_mode_name_)));
      infer_request_ = compiled_model_.create_infer_request();
      backend_name_ = "OPENVINO";
      backend_kind_ = DetectorBackendKind::OPENVINO;
      model_ready_ = true;
      XR_LOG_PASS(
          "ArmorDetector loaded OpenVINO model family=%s output=%dx%d input=%dx%d on device %s requested %s mode %s",
          model_family_ == ModelFamily::SKD ? "SKD" : "SZU",
          CandidateCount(), OutputWidth(), input_shape_.width,
          input_shape_.height, device_name_.c_str(),
          requested_device_name_.c_str(), performance_mode_name_.c_str());
      return true;
    }
    catch (const std::exception& exception)
    {
      XR_LOG_ERROR("ArmorDetector failed to load OpenVINO model on device %s mode %s: %s",
                   device_name_.c_str(), performance_mode_name_.c_str(),
                   exception.what());
      return false;
    }
  }

  bool InferOpenVino(const cv::Mat& input, cv::Mat& output)
  {
    try
    {
      const ov::Shape input_tensor_shape{
          static_cast<size_t>(input_shape_.height),
          static_cast<size_t>(input_shape_.width), 3};
      ov::Tensor input_tensor(ov::element::u8, input_tensor_shape, input.data);
      infer_request_.set_input_tensor(input_tensor);
      infer_request_.infer();

      output_tensor_ = infer_request_.get_output_tensor();
      const auto output_shape = output_tensor_.get_shape();
      if (output_shape.size() != 2U && output_shape.size() != 3U)
      {
        XR_LOG_ERROR("ArmorDetector OpenVINO output rank invalid: %u",
                     static_cast<unsigned>(output_shape.size()));
        return false;
      }

      const int rows = static_cast<int>(
          output_shape.size() == 3U ? output_shape[1] : output_shape[0]);
      const int cols = static_cast<int>(
          output_shape.size() == 3U ? output_shape[2] : output_shape[1]);
      output = cv::Mat(rows, cols, CV_32F, output_tensor_.data<float>());
      if (!OutputShapeMatches(output))
      {
        XR_LOG_ERROR(
            "ArmorDetector OpenVINO output shape invalid: rows=%d cols=%d expected=%dx%d",
            output.rows, output.cols, CandidateCount(), OutputWidth());
        output.release();
        return false;
      }
      return !output.empty();
    }
    catch (const std::exception& exception)
    {
      XR_LOG_ERROR("ArmorDetector OpenVINO inference failed: %s",
                   exception.what());
      return false;
    }
  }
#endif

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
    if (model_family_ == ModelFamily::SKD)
    {
      if (hailo_output_infos_.size() == 1U)
      {
        const auto& info = hailo_output_infos_.front();
        const size_t total_values =
            static_cast<size_t>(info.shape.height) *
            static_cast<size_t>(info.shape.width) *
            static_cast<size_t>(info.shape.features);
        return total_values ==
               static_cast<size_t>(skd_candidate_count * skd_output_width);
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

  static constexpr std::array<const char*, 6> kSkdHostTailOutputNames{{
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
          float* dst = output.ptr<float>(row++);
          const size_t anchor_offset =
              cell_offset +
              static_cast<size_t>(anchor * HailortOutputFieldCount());
          for (int point_index = 0; point_index < 4; ++point_index)
          {
            const int x_index = point_index * 2;
            const int y_index = x_index + 1;
            const float src_x =
                RoundToTailInputPrecision(
                    LoadHailoValue(storage,
                                   anchor_offset + static_cast<size_t>(x_index),
                                   quant_infos, x_index));
            const float src_y =
                RoundToTailInputPrecision(
                    LoadHailoValue(storage,
                                   anchor_offset + static_cast<size_t>(y_index),
                                   quant_infos, y_index));
            dst[x_index] =
                src_x * anchor_mul[anchor][x_index] +
                static_cast<float>(x) * stride_x;
            dst[y_index] =
                src_y * anchor_mul[anchor][y_index] +
                static_cast<float>(y) * stride_y;
          }
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

  template <typename T>
  bool CopySingleSkdOutputToMatrix(const std::vector<T>& storage,
                                   const std::vector<hailo_quant_info_t>& quant_infos,
                                   cv::Mat& output) const
  {
    output.create(skd_output_width, skd_candidate_count, CV_32F);
    if (storage.size() < static_cast<size_t>(skd_output_width * skd_candidate_count))
    {
      XR_LOG_ERROR("ArmorDetector SKD single-output buffer too small: got=%zu expected=%d",
                   storage.size(), skd_output_width * skd_candidate_count);
      output.release();
      return false;
    }

    size_t index = 0;
    for (int row = 0; row < skd_output_width; ++row)
    {
      float* dst = output.ptr<float>(row);
      for (int col = 0; col < skd_candidate_count; ++col, ++index)
      {
        dst[col] = LoadQuantizedValue(storage, index, quant_infos, row);
      }
    }
    return true;
  }

  template <typename T>
  bool FuseSkdHostTailOutputs(const std::unordered_map<std::string, HailoOutputBuffer>& buffers,
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

    const auto* conv45 = read_tensor(kSkdHostTailOutputNames[0]);
    const auto* concat11 = read_tensor(kSkdHostTailOutputNames[1]);
    const auto* conv59 = read_tensor(kSkdHostTailOutputNames[2]);
    const auto* concat14 = read_tensor(kSkdHostTailOutputNames[3]);
    const auto* conv72 = read_tensor(kSkdHostTailOutputNames[4]);
    const auto* concat16 = read_tensor(kSkdHostTailOutputNames[5]);
    if (conv45 == nullptr || concat11 == nullptr || conv59 == nullptr ||
        concat14 == nullptr || conv72 == nullptr || concat16 == nullptr)
    {
      XR_LOG_ERROR("ArmorDetector missing SKD host-tail outputs");
      return false;
    }

    const auto& q45 = buffers.at(kSkdHostTailOutputNames[0]).quant_infos;
    const auto& q11 = buffers.at(kSkdHostTailOutputNames[1]).quant_infos;
    const auto& q59 = buffers.at(kSkdHostTailOutputNames[2]).quant_infos;
    const auto& q14 = buffers.at(kSkdHostTailOutputNames[3]).quant_infos;
    const auto& q72 = buffers.at(kSkdHostTailOutputNames[4]).quant_infos;
    const auto& q16_p5 = buffers.at(kSkdHostTailOutputNames[5]).quant_infos;

    output.create(skd_candidate_count, skd_output_width, CV_32F);
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
          if (row >= skd_candidate_count)
          {
            XR_LOG_ERROR("ArmorDetector SKD host-tail fused too many rows");
            output.release();
            return false;
          }
          float* dst = output.ptr<float>(row++);
          const size_t cell8 = (static_cast<size_t>(y) * static_cast<size_t>(width) +
                                static_cast<size_t>(x)) * 8U;
          const size_t cell16 = (static_cast<size_t>(y) * static_cast<size_t>(width) +
                                 static_cast<size_t>(x)) * 16U;
          for (int field = 0; field < 8; ++field)
          {
            dst[field] = LoadQuantizedValue(*conv8, cell8 + static_cast<size_t>(field), q8, field);
          }
          dst[8] = LoadQuantizedValue(*concat16_tensor, cell16 + 0U, q16, 0);
          dst[17] = LoadQuantizedValue(*concat16_tensor, cell16 + 1U, q16, 1);
          dst[18] = LoadQuantizedValue(*concat16_tensor, cell16 + 2U, q16, 2);
          dst[9] = LoadQuantizedValue(*concat16_tensor, cell16 + 4U, q16, 4);
          dst[10] = LoadQuantizedValue(*concat16_tensor, cell16 + 5U, q16, 5);
          dst[11] = LoadQuantizedValue(*concat16_tensor, cell16 + 6U, q16, 6);
          dst[12] = LoadQuantizedValue(*concat16_tensor, cell16 + 7U, q16, 7);
          dst[13] = LoadQuantizedValue(*concat16_tensor, cell16 + 8U, q16, 8);
          dst[14] = LoadQuantizedValue(*concat16_tensor, cell16 + 9U, q16, 9);
          dst[15] = LoadQuantizedValue(*concat16_tensor, cell16 + 10U, q16, 10);
          dst[16] = LoadQuantizedValue(*concat16_tensor, cell16 + 11U, q16, 11);
          dst[19] = LoadQuantizedValue(*concat16_tensor, cell16 + 12U, q16, 12);
          dst[20] = LoadQuantizedValue(*concat16_tensor, cell16 + 13U, q16, 13);

          for (int point = 0; point < 4; ++point)
          {
            dst[point * 2] =
                dst[point * 2] * static_cast<float>(stride * 2) +
                static_cast<float>(x * stride);
            dst[point * 2 + 1] =
                dst[point * 2 + 1] * static_cast<float>(stride * 2) +
                static_cast<float>(y * stride);
          }
        }
      }
    }

    if (row != skd_candidate_count)
    {
      XR_LOG_ERROR("ArmorDetector SKD host-tail fused row count mismatch: got=%d expected=%d",
                   row, skd_candidate_count);
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
          std::array<float, model_output_width> fields{};
          for (int point_index = 0; point_index < 4; ++point_index)
          {
            const int x_index = point_index * 2;
            const int y_index = x_index + 1;
            const float src_x =
                RoundToTailInputPrecision(
                    LoadHailoValue(storage,
                                   anchor_offset + static_cast<size_t>(x_index),
                                   quant_infos, x_index));
            const float src_y =
                RoundToTailInputPrecision(
                    LoadHailoValue(storage,
                                   anchor_offset + static_cast<size_t>(y_index),
                                   quant_infos, y_index));
            fields[static_cast<size_t>(x_index)] =
                src_x * anchor_mul[anchor][x_index] +
                static_cast<float>(x) * stride_x;
            fields[static_cast<size_t>(y_index)] =
                src_y * anchor_mul[anchor][y_index] +
                static_cast<float>(y) * stride_y;
          }
          for (int field = 8; field < HailortOutputFieldCount(); ++field)
          {
            fields[static_cast<size_t>(field)] =
                RoundToTailInputPrecision(
                    LoadHailoValue(storage,
                                   anchor_offset + static_cast<size_t>(field),
                                   quant_infos, field));
          }

          auto detection = build_detection(
              [&fields](int field)
              {
                return fields[static_cast<size_t>(field)];
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
          (model_family_ == ModelFamily::SKD) ? BuildSkdHailoOutput(output)
                                              : FuseHailoOutputs(output);
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

  bool BuildSkdHailoOutput(cv::Mat& output)
  {
    if (hailo_output_infos_.size() == 1U)
    {
      const auto& info = hailo_output_infos_.front();
      const auto buffer_it = hailo_output_buffers_.find(info.name);
      if (buffer_it == hailo_output_buffers_.end())
      {
        XR_LOG_ERROR("ArmorDetector missing SKD Hailo output buffer for %s",
                     info.name);
        return false;
      }
      const auto& buffer = buffer_it->second;
      switch (buffer.format_type)
      {
        case HAILO_FORMAT_TYPE_UINT8:
          return CopySingleSkdOutputToMatrix(
              std::get<std::vector<uint8_t>>(buffer.storage),
              buffer.quant_infos, output);
        case HAILO_FORMAT_TYPE_UINT16:
          return CopySingleSkdOutputToMatrix(
              std::get<std::vector<uint16_t>>(buffer.storage),
              buffer.quant_infos, output);
        case HAILO_FORMAT_TYPE_FLOAT32:
          return CopySingleSkdOutputToMatrix(
              std::get<std::vector<float>>(buffer.storage),
              buffer.quant_infos, output);
        default:
          XR_LOG_ERROR("ArmorDetector unsupported SKD Hailo output format in single-output path: %d",
                       static_cast<int>(buffer.format_type));
          return false;
      }
    }

    if (hailo_output_infos_.size() == 6U)
    {
      const auto first_it = hailo_output_buffers_.begin();
      if (first_it == hailo_output_buffers_.end())
      {
        XR_LOG_ERROR("ArmorDetector missing SKD host-tail buffers");
        return false;
      }
      switch (first_it->second.format_type)
      {
        case HAILO_FORMAT_TYPE_UINT8:
          return FuseSkdHostTailOutputs<uint8_t>(hailo_output_buffers_, output);
        case HAILO_FORMAT_TYPE_UINT16:
          return FuseSkdHostTailOutputs<uint16_t>(hailo_output_buffers_, output);
        case HAILO_FORMAT_TYPE_FLOAT32:
          return FuseSkdHostTailOutputs<float>(hailo_output_buffers_, output);
        default:
          XR_LOG_ERROR("ArmorDetector unsupported SKD host-tail format: %d",
                       static_cast<int>(first_it->second.format_type));
          return false;
      }
    }

    XR_LOG_ERROR("ArmorDetector unsupported SKD Hailo output topology: outputs=%zu",
                 hailo_output_infos_.size());
    return false;
  }

  bool FuseHailoOutputs(cv::Mat& output)
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
      const detail::ModelOutputView output_view(output, model_candidate_count,
                                                model_output_width);
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
  ModelFamily model_family_{ModelFamily::SZU};
  std::string backend_name_{"NONE"};
  std::string requested_backend_name_{"AUTO_DETECT"};
  std::string requested_device_name_{"AUTO_DETECT"};
  std::string performance_mode_name_{"LATENCY"};
  std::string openvino_model_path_{};
  std::string hailort_hef_path_{};
  std::string hailort_tail_onnx_path_{};
  bool model_ready_{false};
  NetworkInputShape input_shape_{};
  HailoTimingSnapshot last_hailo_timing_{};
  bool hailort_tail_ready_{false};
  cv::dnn::Net hailort_tail_net_{};

#if defined(ARMOR_DETECTOR_HAVE_OPENVINO)
  std::vector<std::string> available_device_names_{};
  std::string device_name_{"CPU"};
  ov::Core ov_core_{};
  ov::CompiledModel compiled_model_{};
  ov::InferRequest infer_request_{};
  ov::Tensor output_tensor_{};
#endif

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
