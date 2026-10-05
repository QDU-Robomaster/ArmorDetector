#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <hailo/hailort.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "CameraTypes.hpp"
#include "InferenceBackend.hpp"
#include "logger.hpp"

/**
 * @brief Hailo-8L 后端，读 v4 HEF。原始 Bayer 字节就是 HEF 的 UINT8 输入；8 个输出保持
 *        UINT8，由解码器按各自的量化参数反量化。每个槽一组页对齐缓冲和 bindings。
 *        Hailo-8L backend for the v4 HEF. The raw Bayer bytes are the HEF's UINT8
 *        input; the 8 outputs stay UINT8 and the decoder dequantises them with their
 *        own parameters. Each slot has page-aligned buffers and its own bindings.
 */
class HailoBackend : public InferenceBackend
{
 public:
  /// 输出层名（HEF 内名，前面还有网络名前缀）/ Output layer names inside the HEF.
  static constexpr std::array<std::array<const char*, 4>, 2> OUTPUT_LAYERS = {{
      {"conv41", "conv39", "conv42", "conv40"},  // stride 8：obj、color、size、corner
      {"conv50", "conv48", "conv51", "conv49"},  // stride 16
  }};
  static constexpr auto TIMEOUT = std::chrono::milliseconds(1000);

  HailoBackend(const std::string& hef_path, std::size_t slots)
  {
    vdevice_ = Take(hailort::VDevice::create(), "VDevice::create");
    model_ = Take(vdevice_->create_infer_model(hef_path), "create_infer_model");
    model_->set_batch_size(1);
    REQUIRE(model_->get_input_names().size() == 1);
    input_name_ = model_->get_input_names().front();
    auto input = Take(model_->input(input_name_), "input");
    input.set_format_type(HAILO_FORMAT_TYPE_UINT8);
    REQUIRE(input.get_frame_size() == CameraTypes::FRAME_BYTES);

    for (std::size_t s = 0; s < 2; ++s)
    {
      for (std::size_t k = 0; k < 4; ++k)
      {
        Output& out = outputs_[s][k];
        out.name = FindOutput(OUTPUT_LAYERS[s][k]);
        auto stream = Take(model_->output(out.name), "output");
        stream.set_format_type(HAILO_FORMAT_TYPE_UINT8);
        const hailo_3d_image_shape_t shape = stream.shape();
        const hailo_quant_info_t quant = stream.get_quant_infos().front();
        out.view = {nullptr,
                    ArmorV4::TensorView::Type::U8,
                    true,
                    static_cast<int>(shape.height),
                    static_cast<int>(shape.width),
                    static_cast<int>(shape.features),
                    quant.qp_scale,
                    quant.qp_zp};
        out.frame_size = stream.get_frame_size();
      }
    }
    configured_ = std::make_unique<hailort::ConfiguredInferModel>(
        Take(model_->configure(), "configure"));

    for (std::size_t i = 0; i < slots; ++i)
    {
      Slot slot;
      slot.bindings = Take(configured_->create_bindings(), "create_bindings");
      slot.input = PageBuffer(CameraTypes::FRAME_BYTES);
      REQUIRE(slot.bindings.input(input_name_)
                  ->set_buffer(hailort::MemoryView(
                      slot.input.get(), CameraTypes::FRAME_BYTES)) == HAILO_SUCCESS);
      for (std::size_t s = 0; s < 2; ++s)
      {
        for (std::size_t k = 0; k < 4; ++k)
        {
          const Output& out = outputs_[s][k];
          slot.outputs[s][k] = PageBuffer(out.frame_size);
          REQUIRE(slot.bindings.output(out.name)->set_buffer(hailort::MemoryView(
                      slot.outputs[s][k].get(), out.frame_size)) == HAILO_SUCCESS);
        }
      }
      slots_.push_back(std::move(slot));
    }
  }

  ~HailoBackend() override
  {
    for (Slot& slot : slots_)
    {
      if (slot.job)
      {
        slot.job->wait(TIMEOUT);
      }
    }
  }

  std::size_t Slots() const override { return slots_.size(); }
  uint8_t* Input(std::size_t slot) override { return slots_[slot].input.get(); }

  bool Start(std::size_t slot) override
  {
    if (configured_->wait_for_async_ready(TIMEOUT, 1) != HAILO_SUCCESS)
    {
      return false;
    }
    auto job = configured_->run_async(slots_[slot].bindings,
                                      [](const hailort::AsyncInferCompletionInfo&) {});
    if (!job.has_value())
    {
      return false;
    }
    slots_[slot].job.emplace(job.release());
    return true;
  }

  bool Wait(std::size_t slot) override
  {
    std::optional<hailort::AsyncInferJob>& job = slots_[slot].job;
    const bool ok = job && job->wait(TIMEOUT) == HAILO_SUCCESS;
    job.reset();
    return ok;
  }

  ArmorV4::Outputs Outputs(std::size_t slot) override
  {
    ArmorV4::Outputs outputs{};
    for (std::size_t s = 0; s < 2; ++s)
    {
      ArmorV4::TensorView* views[4] = {&outputs[s].obj, &outputs[s].color,
                                       &outputs[s].size, &outputs[s].corner};
      for (std::size_t k = 0; k < 4; ++k)
      {
        *views[k] = outputs_[s][k].view;
        views[k]->data = slots_[slot].outputs[s][k].get();
      }
    }
    return outputs;
  }

 private:
  struct FreeDeleter
  {
    void operator()(uint8_t* p) const { std::free(p); }
  };
  using Buffer = std::unique_ptr<uint8_t, FreeDeleter>;

  struct Output
  {
    std::string name;
    ArmorV4::TensorView view;
    std::size_t frame_size = 0;
  };

  struct Slot
  {
    hailort::ConfiguredInferModel::Bindings bindings;
    Buffer input;
    std::array<std::array<Buffer, 4>, 2> outputs;
    std::optional<hailort::AsyncInferJob> job;
  };

  template <typename T>
  static T Take(hailort::Expected<T> expected, const char* what)
  {
    if (!expected.has_value())
    {
      XR_LOG_ERROR("HailoRT %s failed: %d", what, static_cast<int>(expected.status()));
      REQUIRE(false);
    }
    return expected.release();
  }

  static Buffer PageBuffer(std::size_t size)
  {
    constexpr std::size_t PAGE =
        16384;  // HailoRT 建议 16 KiB 对齐 / HailoRT asks for 16 KiB
    return Buffer(
        static_cast<uint8_t*>(std::aligned_alloc(PAGE, (size + PAGE - 1) / PAGE * PAGE)));
  }

  /// 输出名形如 `<网络名>/conv41` / Output names look like `<network>/conv41`.
  std::string FindOutput(const std::string& layer) const
  {
    for (const std::string& name : model_->get_output_names())
    {
      if (name.size() > layer.size() && name.compare(name.size() - layer.size() - 1,
                                                     std::string::npos, "/" + layer) == 0)
      {
        return name;
      }
    }
    XR_LOG_ERROR("HEF has no output layer %s", layer.c_str());
    REQUIRE(false);
    return {};
  }

  std::unique_ptr<hailort::VDevice> vdevice_;
  std::shared_ptr<hailort::InferModel> model_;
  std::unique_ptr<hailort::ConfiguredInferModel> configured_;
  std::string input_name_;
  std::array<std::array<Output, 4>, 2> outputs_;
  std::vector<Slot> slots_;
};
