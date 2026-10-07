#pragma once

#include <array>
#include <cstdint>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

#include "CameraTypes.hpp"
#include "InferenceBackend.hpp"

/**
 * @brief OpenVINO 后端，读 v4 浮点 ONNX。输入在图前加一步 uint8 → float，主机直接写原始
 *        Bayer 字节；K 个 InferRequest 对应 K 个槽。
 *        OpenVINO backend for the v4 float ONNX. A uint8 → float step is added in front
 *        of the graph so the host writes raw Bayer bytes; K InferRequests are the K
 *        slots.
 */
class OpenVinoBackend : public InferenceBackend
{
 public:
  OpenVinoBackend(const std::string& onnx_path, std::size_t slots,
                  const std::string& device)
  {
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(onnx_path);
    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input("image").tensor().set_element_type(ov::element::u8);
    ppp.input("image").preprocess().convert_element_type(ov::element::f32);
    model = ppp.build();
    compiled_ = core.compile_model(
        model, device, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    for (std::size_t i = 0; i < slots; ++i)
    {
      Slot slot{compiled_.create_infer_request(),
                std::vector<uint8_t>(CameraTypes::FRAME_BYTES)};
      slot.request.set_input_tensor(ov::Tensor(
          ov::element::u8, {1, 1, CameraTypes::FRAME_HEIGHT, CameraTypes::FRAME_WIDTH},
          slot.input.data()));
      slots_.push_back(std::move(slot));
    }
  }

  std::size_t Slots() const override { return slots_.size(); }
  uint8_t* Input(std::size_t slot) override { return slots_[slot].input.data(); }

  bool Start(std::size_t slot) override
  {
    slots_[slot].request.start_async();
    return true;
  }

  bool Wait(std::size_t slot) override
  {
    slots_[slot].request.wait();
    return true;
  }

  ArmorV4::Outputs Outputs(std::size_t slot) override
  {
    static constexpr std::array<std::array<const char*, 4>, 2> NAMES = {{
        {"obj_p3", "color_p3", "size_p3", "corner_p3"},
        {"obj_p4", "color_p4", "size_p4", "corner_p4"},
    }};
    ov::InferRequest& request = slots_[slot].request;
    ArmorV4::Outputs outputs{};
    for (std::size_t s = 0; s < 2; ++s)
    {
      ArmorV4::TensorView* views[4] = {&outputs[s].obj, &outputs[s].color,
                                       &outputs[s].size, &outputs[s].corner};
      for (std::size_t k = 0; k < 4; ++k)
      {
        ov::Tensor tensor = request.get_tensor(NAMES[s][k]);
        const ov::Shape shape = tensor.get_shape();  // [1, C, H, W]
        *views[k] = {tensor.data<float>(),
                     ArmorV4::TensorView::Type::F32,
                     false,
                     static_cast<int>(shape[2]),
                     static_cast<int>(shape[3]),
                     static_cast<int>(shape[1])};
      }
    }
    return outputs;
  }

 private:
  struct Slot
  {
    ov::InferRequest request;
    std::vector<uint8_t> input;
  };

  ov::CompiledModel compiled_;
  std::vector<Slot> slots_;
};
