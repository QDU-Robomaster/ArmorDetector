#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <openvino/openvino.hpp>

#include "logger.hpp"

namespace armor_detector_detail
{

/** OpenVINO backend; the model contains its RGB-u8 HWC preprocessing. */
class OpenVinoArmorBackend
{
 public:
  static constexpr int input_width = 640;
  static constexpr int input_height = 512;
  static constexpr int candidate_count = 20160;
  static constexpr int output_width = 22;
  static constexpr std::size_t input_bytes =
      static_cast<std::size_t>(input_width) * input_height * 3U;

  /** One request per pipeline buffer: another frame cannot overwrite its output. */
  struct Slot
  {
    const OpenVinoArmorBackend* owner{nullptr};
    ov::Tensor input{};
    ov::InferRequest request{};
    ov::Tensor output{};
  };

  bool Configure(const char* model_path)
  {
    try
    {
      if (model_path == nullptr || model_path[0] == '\0')
      {
        XR_LOG_ERROR("ArmorDetector OpenVINO model path is empty");
        return false;
      }
      core_ = std::make_unique<ov::Core>();
      const auto devices = core_->get_available_devices();
      const char* requested = std::getenv("XR_ARMOR_OPENVINO_DEVICE");
      device_ = SelectDevice(devices, requested);
      if (device_.empty())
      {
        XR_LOG_ERROR("ArmorDetector OpenVINO has no available inference device");
        return false;
      }

      const auto model = core_->read_model(model_path);
      if (model->inputs().size() != 1U || model->outputs().size() != 1U ||
          model->input().get_element_type() != ov::element::u8 ||
          model->input().get_partial_shape() !=
              ov::PartialShape{input_height, input_width, 3} ||
          model->output().get_element_type() != ov::element::f32 ||
          model->output().get_partial_shape().is_dynamic() ||
          !OutputShapeSupported(model->output().get_shape()))
      {
        XR_LOG_ERROR(
            "ArmorDetector OpenVINO model contract mismatch: expected RGB u8 "
            "[512,640,3] -> f32 [1,20160,22] or a 2-D candidate matrix");
        return false;
      }
      compiled_ = core_->compile_model(
          model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
      XR_LOG_PASS("ArmorDetector loaded OpenVINO model=%s device=%s input=640x512",
                  model_path, device_.c_str());
      return true;
    }
    catch (const std::exception& exception)
    {
      XR_LOG_ERROR("ArmorDetector OpenVINO initialization failed on %s: %s",
                   device_.c_str(), exception.what());
      return false;
    }
  }

  /** Use the requested device, otherwise the first of NPU, GPU, CPU among the devices. */
  static std::string SelectDevice(const std::vector<std::string>& devices,
                                  const char* requested)
  {
    if (requested != nullptr && requested[0] != '\0' &&
        std::string(requested) != "AUTO_DETECT")
    {
      return requested;
    }
    for (const char* preferred : {"NPU", "GPU", "CPU"})
    {
      const std::string prefix(preferred);
      for (const auto& device : devices)
      {
        if (device == prefix || device.rfind(prefix + ".", 0U) == 0U)
        {
          return device;
        }
      }
    }
    return {};
  }

  static bool OutputShapeSupported(const ov::Shape& shape)
  {
    if (shape.size() == 3U)
    {
      return shape[0] == 1U && shape[1] == candidate_count &&
             shape[2] == output_width;
    }
    return shape.size() == 2U &&
           ((shape[0] == candidate_count && shape[1] == output_width) ||
            (shape[0] == output_width && shape[1] == candidate_count));
  }

  bool InitSlot(uint8_t* data, std::size_t bytes, std::unique_ptr<Slot>& output)
  {
    output.reset();
    if (data == nullptr || bytes != input_bytes)
    {
      return false;
    }
    try
    {
      auto slot = std::make_unique<Slot>();
      slot->owner = this;
      slot->input = ov::Tensor(ov::element::u8, {input_height, input_width, 3}, data);
      slot->request = compiled_.create_infer_request();
      slot->request.set_input_tensor(slot->input);
      output = std::move(slot);
      return true;
    }
    catch (const std::exception& exception)
    {
      XR_LOG_ERROR("ArmorDetector OpenVINO slot initialization failed: %s",
                   exception.what());
      return false;
    }
  }

  bool Infer(Slot& slot)
  {
    if (slot.owner != this)
    {
      return false;
    }
    try
    {
      slot.output = {};
      slot.request.infer();
      slot.output = slot.request.get_output_tensor();
      if (slot.output.get_element_type() != ov::element::f32 ||
          !OutputShapeSupported(slot.output.get_shape()) || !slot.output.is_continuous())
      {
        XR_LOG_ERROR("ArmorDetector OpenVINO inference returned an unsupported tensor");
        slot.output = {};
        return false;
      }
      return true;
    }
    catch (const std::exception& exception)
    {
      XR_LOG_ERROR("ArmorDetector OpenVINO inference failed: %s", exception.what());
      slot.output = {};
      return false;
    }
  }

  /** The returned view lives until this slot is reused or released. */
  static cv::Mat OutputView(Slot& slot)
  {
    const auto shape = slot.output.get_shape();
    const std::size_t offset = shape.size() == 3U ? 1U : 0U;
    return cv::Mat(static_cast<int>(shape[offset]), static_cast<int>(shape[offset + 1U]),
                   CV_32F, slot.output.data<float>());
  }

  [[nodiscard]] const std::string& DeviceName() const { return device_; }

 private:
  std::unique_ptr<ov::Core> core_{};
  ov::CompiledModel compiled_{};
  std::string device_{};
};

}  // namespace armor_detector_detail
