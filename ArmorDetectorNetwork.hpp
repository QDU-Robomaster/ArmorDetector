#pragma once

/**
 * @file ArmorDetectorNetwork.hpp
 * @brief ArmorDetector 的 OpenVINO 模型加载和单帧推理封装。
 */

#include <exception>

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
   * @brief 加载并编译指定 detector profile 对应的 OpenVINO 模型。
   * @param profile detector 模型 profile，用于确定输入尺寸和日志名称。
   * @param model_path 模型文件路径。
   * @return 模型加载、预处理构建和 CPU 编译全部成功时返回 true。
   */
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

  /**
   * @brief 当前模型是否已经成功加载并可推理。
   * @return 模型 ready 时返回 true。
   */
  [[nodiscard]] bool Ready() const { return model_ready_; }

  /**
   * @brief 当前已配置的 detector profile。
   * @return detector profile。
   */
  [[nodiscard]] DetectorProfile Profile() const { return profile_; }

  /**
   * @brief 当前模型输入尺寸。
   * @return 输入宽高。
   */
  [[nodiscard]] NetworkInputShape InputShape() const { return input_shape_; }

  /**
   * @brief 对一帧已经 resize 到模型输入尺寸的 BGR 图像执行推理。
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
  DetectorProfile profile_{DetectorProfile::YOLO_KEYPOINT_640X640}; ///< 当前模型 profile。
  bool model_ready_{false};                                         ///< 模型是否可用。
  NetworkInputShape input_shape_{};                                 ///< 当前模型输入宽高。
  ov::Core ov_core_{};                                               ///< OpenVINO runtime core。
  ov::CompiledModel compiled_model_{};                               ///< 已编译的 OpenVINO 模型。
  ov::InferRequest infer_request_{};                                 ///< 复用的同步推理请求。
  ov::Tensor output_tensor_{};                                       ///< 保存输出视图生命周期的 tensor。
};

}  // namespace armor_detector_detail
