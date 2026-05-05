#pragma once

/**
 * @file ArmorDetectorOnnxRuntimeNetwork.hpp
 * @brief ArmorDetector 的 ONNX Runtime CUDA 推理封装。
 */

#include <array>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#if __has_include(<onnxruntime_cxx_api.h>)
#include <onnxruntime_cxx_api.h>
#elif __has_include(<onnxruntime/core/session/onnxruntime_cxx_api.h>)
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#else
#error "ONNX Runtime C++ header was not found."
#endif

namespace armor_detector_detail
{

/**
 * @brief ONNX Runtime CUDA armor keypoint network wrapper.
 *
 * 该类与 OpenVinoArmorNetwork 暴露同一组接口，使 detector 解码、PnP 和发布
 * 链路不需要关心具体推理后端。输入为已经缩放到模型尺寸的 BGR8 图像，输出
 * 为模型原始的 21x6720 FP32 矩阵视图。
 */
class OnnxRuntimeArmorNetwork
{
 public:
  /**
   * @brief 加载 ONNX 模型并创建 CUDA execution provider session。
   * @param model_path ONNX 模型文件路径。
   * @param device_name 兼容 OpenVINO 后端配置；当前 CUDA 后端固定使用 device 0。
   * @param performance_mode 兼容 OpenVINO 后端配置；当前 CUDA 后端不使用该参数。
   * @return session 创建成功时返回 true。
   */
  bool Configure(const char* model_path, const char* device_name,
                 const char* performance_mode)
  {
    (void)device_name;
    (void)performance_mode;

    model_ready_ = false;
    session_.reset();
    output_tensors_.clear();

    try
    {
      Ort::SessionOptions session_options;
      session_options.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_ALL);

      OrtCUDAProviderOptions cuda_options{};
      cuda_options.device_id = 0;
      cuda_options.arena_extend_strategy = 0;
      cuda_options.gpu_mem_limit = std::numeric_limits<std::size_t>::max();
      cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
      cuda_options.do_copy_in_default_stream = 1;
      session_options.AppendExecutionProvider_CUDA(cuda_options);

      session_ = std::make_unique<Ort::Session>(env_, model_path,
                                                session_options);
      input_buffer_.resize(static_cast<std::size_t>(input_shape_.width) *
                           static_cast<std::size_t>(input_shape_.height) * 3U);
      model_ready_ = true;
      XR_LOG_PASS("ArmorDetector loaded %s model on ONNX Runtime CUDA path=%s",
                  detector_model_name, model_path);
      return true;
    }
    catch (const std::exception& exception)
    {
      XR_LOG_ERROR("ArmorDetector failed to load %s model on ONNX Runtime CUDA: %s",
                   detector_model_name, exception.what());
      return false;
    }
  }

  /**
   * @brief 当前模型是否已经成功加载并可推理。
   * @return 模型 ready 时返回 true。
   */
  [[nodiscard]] bool Ready() const { return model_ready_; }

  /**
   * @brief 当前模型输入尺寸。
   * @return 输入宽高。
   */
  [[nodiscard]] NetworkInputShape InputShape() const { return input_shape_; }

  /**
   * @brief 对一帧已经 resize 到模型输入尺寸的 BGR 图像执行推理。
   * @param input 模型输入尺寸的 BGR8 图像。
   * @param output 输出矩阵视图，形状为 21x6720。
   * @return 推理成功且输出形状匹配时返回 true。
   */
  bool Infer(const cv::Mat& input, cv::Mat& output)
  {
    output.release();
    if (!model_ready_ || session_ == nullptr || input.empty())
    {
      return false;
    }
    if (input.type() != CV_8UC3 || input.cols != input_shape_.width ||
        input.rows != input_shape_.height)
    {
      XR_LOG_ERROR(
          "ArmorDetector ONNX input invalid: rows=%d cols=%d type=%d expected=%dx%d BGR8",
          input.rows, input.cols, input.type(), input_shape_.height,
          input_shape_.width);
      return false;
    }

    try
    {
      FillInputTensor(input);

      const std::array<int64_t, 4> input_dims = {
          1, 3, input_shape_.height, input_shape_.width};
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          memory_info_, input_buffer_.data(), input_buffer_.size(),
          input_dims.data(), input_dims.size());

      output_tensors_ = session_->Run(
          Ort::RunOptions{nullptr}, input_names_.data(), &input_tensor, 1,
          output_names_.data(), 1);
      if (output_tensors_.empty() || !output_tensors_[0].IsTensor())
      {
        XR_LOG_ERROR("ArmorDetector ONNX output is empty");
        return false;
      }

      auto shape_info = output_tensors_[0].GetTensorTypeAndShapeInfo();
      const std::vector<int64_t> shape = shape_info.GetShape();
      if (shape.size() != 3U || shape[0] != 1 ||
          shape[1] != direct_keypoint_output_width ||
          shape[2] != direct_keypoint_candidate_count)
      {
        XR_LOG_ERROR(
            "ArmorDetector ONNX output shape invalid: rank=%u expected=1x21x6720",
            static_cast<unsigned>(shape.size()));
        return false;
      }

      output = cv::Mat(direct_keypoint_output_width,
                       direct_keypoint_candidate_count, CV_32F,
                       output_tensors_[0].GetTensorMutableData<float>());
      return !output.empty();
    }
    catch (const std::exception& exception)
    {
      XR_LOG_ERROR("ArmorDetector ONNX inference failed: %s", exception.what());
      return false;
    }
  }

 private:
  /**
   * @brief 将 BGR HWC u8 图像转换为 BGR NCHW FP32/255 tensor。
   * @param input 模型尺寸 BGR8 图像。
   */
  void FillInputTensor(const cv::Mat& input)
  {
    const int width = input_shape_.width;
    const int height = input_shape_.height;
    const std::size_t plane_size =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);

    for (int y = 0; y < height; ++y)
    {
      const auto* row = input.ptr<unsigned char>(y);
      for (int x = 0; x < width; ++x)
      {
        const std::size_t hw =
            static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
            static_cast<std::size_t>(x);
        const int pixel_index = x * 3;
        input_buffer_[hw] =
            static_cast<float>(row[pixel_index]) / 255.0F;
        input_buffer_[plane_size + hw] =
            static_cast<float>(row[pixel_index + 1]) / 255.0F;
        input_buffer_[plane_size * 2U + hw] =
            static_cast<float>(row[pixel_index + 2]) / 255.0F;
      }
    }
  }

  bool model_ready_{false};                         ///< 模型是否可用。
  NetworkInputShape input_shape_{};                 ///< 当前模型输入宽高。
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "ArmorDetector"};  ///< ONNX Runtime 环境。
  Ort::MemoryInfo memory_info_ =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault); ///< CPU tensor 内存描述。
  std::unique_ptr<Ort::Session> session_{};         ///< CUDA session。
  std::vector<float> input_buffer_{};               ///< 复用的 NCHW 输入缓冲。
  std::vector<Ort::Value> output_tensors_{};        ///< 保存输出视图生命周期。
  std::array<const char*, 1> input_names_{{"image"}};                 ///< ONNX 输入名。
  std::array<const char*, 1> output_names_{{"armor_keypoint_output"}}; ///< ONNX 输出名。
};

}  // namespace armor_detector_detail
