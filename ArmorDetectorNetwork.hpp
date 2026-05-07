#pragma once

/**
 * @file ArmorDetectorNetwork.hpp
 * @brief ArmorDetector 的 OpenVINO 模型加载和单帧推理封装。
 */

#include <cmath>
#include <exception>
#include <initializer_list>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <openvino/openvino.hpp>

namespace armor_detector_detail
{

/**
 * @brief OpenVINO armor keypoint network wrapper.
 *
 * 该类只负责模型生命周期、OpenVINO 预处理描述和同步推理。输出 tensor
 * 会被包装成 cv::Mat 视图交给 detector decoder，视图有效期由本对象保存的
 * output_tensor_ 保证到下一次 Infer() 调用前。
 */
class OpenVinoArmorNetwork
{
 public:
  /**
   * @brief 加载并编译 OpenVINO dense-grid keypoint 模型。
   * @param model_path 模型文件路径。
   * @param device_name OpenVINO 设备名，例如 CPU/GPU/NPU/AUTO:GPU,NPU。
   * @param performance_mode OpenVINO performance hint 名称。
   * @param source_width 源图像宽度。
   * @param source_height 源图像高度。
   * @return 模型加载、预处理构建和指定设备编译全部成功时返回 true。
   */
  bool Configure(const char* model_path, const char* device_name,
                 const char* performance_mode, int source_width,
                 int source_height)
  {
    requested_device_name_ = NormalizeDeviceName(device_name);
    available_device_names_ = QueryAvailableDeviceNames();
    device_name_ = ResolveDeviceName(requested_device_name_);
    performance_mode_name_ = NormalizePerformanceModeName(performance_mode);
    source_shape_.width =
        (source_width > 0) ? source_width : input_shape_.width;
    source_shape_.height =
        (source_height > 0) ? source_height : input_shape_.height;
    model_ready_ = false;
    compiled_model_ = ov::CompiledModel();
    infer_request_ = ov::InferRequest();
    output_tensor_ = ov::Tensor();

    try
    {
      LogDeviceSelection();
      auto model = ov_core_.read_model(model_path);
      ov::preprocess::PrePostProcessor post_processor(model);
      auto& input = post_processor.input();

      input.tensor()
          .set_element_type(ov::element::u8)
          .set_layout("NHWC")
          .set_color_format(ov::preprocess::ColorFormat::BGR)
          .set_spatial_static_shape(static_cast<size_t>(source_shape_.height),
                                    static_cast<size_t>(source_shape_.width));
      input.model().set_layout("NCHW");
      auto& preprocess = input.preprocess();
      preprocess.resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
      preprocess.convert_element_type(ov::element::f32);
      preprocess.scale(255.0);

      model = post_processor.build();
      compiled_model_ = ov_core_.compile_model(
          model, device_name_,
          ov::hint::performance_mode(
              ParsePerformanceMode(performance_mode_name_)));
      infer_request_ = compiled_model_.create_infer_request();
      model_ready_ = true;
      XR_LOG_PASS(
          "ArmorDetector loaded %s model on OpenVINO device %s requested %s "
          "mode %s source=%dx%d",
          detector_model_name, device_name_.c_str(),
          requested_device_name_.c_str(), performance_mode_name_.c_str(),
          source_shape_.width, source_shape_.height);
      return true;
    }
    catch (const std::exception& exception)
    {
      XR_LOG_ERROR(
          "ArmorDetector failed to load %s model on device %s mode %s: %s",
          detector_model_name, device_name_.c_str(),
          performance_mode_name_.c_str(), exception.what());
      return false;
    }
  }

  /**
   * @brief 当前模型是否已经成功加载并可推理。
   * @return 模型 ready 时返回 true。
   */
  [[nodiscard]] bool Ready() const { return model_ready_; }

  /**
   * @brief 当前请求的 OpenVINO 设备名。
   * @return 设备名字符串。
   */
  [[nodiscard]] const std::string& DeviceName() const { return device_name_; }

  /**
   * @brief 外部配置请求的 OpenVINO 设备名。
   * @return 请求设备名；自动探测时为 AUTO_DETECT。
   */
  [[nodiscard]] const std::string& RequestedDeviceName() const
  {
    return requested_device_name_;
  }

  /**
   * @brief 当前请求的 OpenVINO performance hint 名称。
   * @return performance hint 名称。
   */
  [[nodiscard]] const std::string& PerformanceModeName() const
  {
    return performance_mode_name_;
  }

  /**
   * @brief 当前模型输入尺寸。
   * @return 输入宽高。
   */
  [[nodiscard]] NetworkInputShape InputShape() const { return input_shape_; }

  /**
   * @brief 对一帧原始 BGR 图像执行推理。
   *
   * OpenVINO 预处理图负责把输入图像缩放到模型固定的 640x512 平面。
   *
   * @param input NHWC BGR8 输入图像。
   * @param output 输出矩阵视图，行是候选，列是 keypoint/score/class 字段。
   * @return 推理成功且输出非空时返回 true。
   */
  bool Infer(const cv::Mat& input, cv::Mat& output)
  {
    output.release();
    if (!model_ready_ || input.empty())
    {
      return false;
    }
    if (input.cols != source_shape_.width || input.rows != source_shape_.height)
    {
      XR_LOG_ERROR(
          "ArmorDetector network input shape invalid: got=%dx%d expected=%dx%d",
          input.cols, input.rows, source_shape_.width, source_shape_.height);
      return false;
    }
    if (input.type() != CV_8UC3)
    {
      XR_LOG_ERROR("ArmorDetector network input type invalid: type=%d expected=CV_8UC3",
                   input.type());
      return false;
    }

    try
    {
      const ov::Shape input_tensor_shape{
          1, static_cast<size_t>(input.rows), static_cast<size_t>(input.cols),
          3};
      ov::Tensor input_tensor;
      if (input.isContinuous())
      {
        input_tensor =
            ov::Tensor(ov::element::u8, input_tensor_shape, input.data);
      }
      else
      {
        input_tensor = ov::Tensor(
            ov::element::u8, input_tensor_shape, input.data,
            ov::Strides{static_cast<size_t>(input.rows) * input.step[0],
                        input.step[0], input.elemSize(), input.elemSize1()});
      }
      infer_request_.set_input_tensor(input_tensor);
      infer_request_.infer();

      output_tensor_ = infer_request_.get_output_tensor();
      const auto output_shape = output_tensor_.get_shape();
      if (output_shape.size() < 3U)
      {
        XR_LOG_ERROR("ArmorDetector network output rank invalid: %u",
                     static_cast<unsigned>(output_shape.size()));
        return false;
      }

      output = cv::Mat(static_cast<int>(output_shape[1]),
                       static_cast<int>(output_shape[2]), CV_32F,
                       output_tensor_.data<float>());
      return !output.empty();
    }
    catch (const std::exception& exception)
    {
      XR_LOG_ERROR("ArmorDetector network inference failed: %s",
                   exception.what());
      return false;
    }
  }

 private:
  /**
   * @brief 归一化外部传入的 OpenVINO 设备名。
   * @param device_name 可能为空的设备名。
   * @return 非空设备名；空值使用 AUTO_DETECT。
   */
  static std::string NormalizeDeviceName(const char* device_name)
  {
    if (device_name == nullptr || device_name[0] == '\0')
    {
      return "AUTO_DETECT";
    }
    return device_name;
  }

  /**
   * @brief 判断 OpenVINO 设备名是否匹配基础设备类型。
   * @param device OpenVINO 返回的设备名，例如 GPU 或 GPU.0。
   * @param base 基础设备名，例如 GPU。
   * @return 完全匹配或前缀匹配时返回 true。
   */
  static bool DeviceMatches(const std::string& device, const char* base)
  {
    const std::string base_name(base);
    return device == base_name || device.rfind(base_name + ".", 0) == 0;
  }

  /**
   * @brief 自动探测并解析最终传给 OpenVINO 的设备名。
   *
   * AUTO_DETECT 是本模块的部署策略：优先 NPU，其次 GPU，最后 CPU。这样
   * Core Ultra 类机器会用 NPU，11/12 代 NUC 会用 iGPU，CI 或无加速设备
   * 的机器会稳定回退 CPU。
   *
   * @param requested_device 外部配置设备名。
   * @return 实际 OpenVINO 设备名。
   */
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

  /**
   * @brief 查询当前 OpenVINO runtime 可见设备。
   * @return 设备名列表，查询失败时返回空列表。
   */
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

  /**
   * @brief 将可见 OpenVINO 设备拼接成日志字符串。
   * @return 逗号分隔设备名；空列表返回 none。
   */
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

  /**
   * @brief 当前 OpenVINO 可见设备中是否存在 GPU/NPU 等加速设备。
   * @return 只要存在非 CPU 设备就返回 true。
   */
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

  /**
   * @brief 打印 OpenVINO 设备选择日志。
   *
   * 可见设备只有 CPU 时使用 warning，因为这通常意味着目标机缺少可用
   * GPU/NPU 加速；否则打印普通初始化信息。
   */
  void LogDeviceSelection() const
  {
    const auto available_devices_text = AvailableDevicesText();
    if (!HasAcceleratorDevice())
    {
      XR_LOG_WARN(
          "ArmorDetector OpenVINO CPU-only runtime; available=%s requested=%s "
          "resolved=%s mode=%s",
          available_devices_text.c_str(), requested_device_name_.c_str(),
          device_name_.c_str(), performance_mode_name_.c_str());
      return;
    }

    XR_LOG_INFO(
        "ArmorDetector OpenVINO devices=%s requested=%s resolved=%s mode=%s",
        available_devices_text.c_str(), requested_device_name_.c_str(),
        device_name_.c_str(), performance_mode_name_.c_str());
  }

  /**
   * @brief 归一化外部传入的 OpenVINO performance hint。
   * @param performance_mode 可能为空的 hint 名称。
   * @return 非空 hint 名称；空值使用 LATENCY。
   */
  static std::string NormalizePerformanceModeName(const char* performance_mode)
  {
    if (performance_mode == nullptr || performance_mode[0] == '\0')
    {
      return "LATENCY";
    }
    return performance_mode;
  }

  /**
   * @brief 将配置字符串转换成 OpenVINO performance mode 枚举。
   * @param performance_mode 已归一化的 hint 名称。
   * @return OpenVINO performance mode；未知值回退到 LATENCY。
   */
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

  std::vector<std::string> available_device_names_{};                ///< OpenVINO 可见设备。
  std::string requested_device_name_{"AUTO_DETECT"};                 ///< 配置请求设备。
  std::string device_name_{"CPU"};                                   ///< OpenVINO 编译设备。
  std::string performance_mode_name_{"LATENCY"};                     ///< OpenVINO 性能模式。
  bool model_ready_{false};                                         ///< 模型是否可用。
  NetworkInputShape input_shape_{};                                 ///< 当前模型输入宽高。
  NetworkInputShape source_shape_{};                                ///< OpenVINO 输入 tensor 的源图像宽高。
  ov::Core ov_core_{};                                               ///< OpenVINO runtime core。
  ov::CompiledModel compiled_model_{};                               ///< 已编译的 OpenVINO 模型。
  ov::InferRequest infer_request_{};                                 ///< 复用的同步推理请求。
  ov::Tensor output_tensor_{};                                       ///< 保存输出视图生命周期的 tensor。
};

}  // namespace armor_detector_detail
