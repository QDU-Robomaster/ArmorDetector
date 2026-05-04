#pragma once

#include <exception>

#include <opencv2/core.hpp>
#include <openvino/openvino.hpp>

namespace armor_detector_detail
{

class OpenVinoArmorNetwork
{
 public:
  bool Configure(DetectorProfile profile, const char* model_path)
  {
    profile_ = profile;
    input_shape_ = ProfileSpecFor(profile).input_shape;
    model_ready_ = false;
    compiled_model_ = ov::CompiledModel();
    infer_request_ = ov::InferRequest();
    output_tensor_ = ov::Tensor();

    try
    {
      auto model = ov_core_.read_model(model_path);
      ov::preprocess::PrePostProcessor post_processor(model);
      auto& input = post_processor.input();

      input.tensor()
          .set_element_type(ov::element::u8)
          .set_shape(ov::PartialShape{1, input_shape_.height,
                                      input_shape_.width, 3})
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
      return true;
    }
    catch (const std::exception& exception)
    {
      XR_LOG_ERROR("ArmorDetector failed to load %s model: %s",
                   DetectorProfileName(profile_), exception.what());
      return false;
    }
  }

  [[nodiscard]] bool Ready() const { return model_ready_; }
  [[nodiscard]] DetectorProfile Profile() const { return profile_; }
  [[nodiscard]] NetworkInputShape InputShape() const { return input_shape_; }

  bool Infer(const cv::Mat& input, cv::Mat& output)
  {
    output.release();
    if (!model_ready_ || input.empty())
    {
      return false;
    }

    try
    {
      ov::Tensor input_tensor(
          ov::element::u8,
          ov::Shape{1, static_cast<size_t>(input_shape_.height),
                    static_cast<size_t>(input_shape_.width), 3},
          input.data);
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
  DetectorProfile profile_{DetectorProfile::YOLO_KEYPOINT_640X640};
  bool model_ready_{false};
  NetworkInputShape input_shape_{};
  ov::Core ov_core_{};
  ov::CompiledModel compiled_model_{};
  ov::InferRequest infer_request_{};
  ov::Tensor output_tensor_{};
};

}  // namespace armor_detector_detail
