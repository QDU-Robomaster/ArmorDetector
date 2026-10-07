#pragma once

#include <cstddef>
#include <cstdint>

#include "ArmorV4Decoder.hpp"

/**
 * @brief v4 检测模型的推理后端：K 个槽，每个槽一份输入缓冲和一次在途推理。
 *        Inference backend of the v4 detector: K slots, each with one input buffer and
 *        one inference in flight.
 *
 * 用法：往 `Input(slot)` 写 640×512 原始 Bayer 字节，`Start(slot)` 提交，`Wait(slot)`
 * 等完成，再用 `Outputs(slot)` 读 8 个输出张量。同一个槽在 Wait 之前不再 Start。
 *
 * Usage: write the 640×512 raw Bayer bytes to `Input(slot)`, `Start(slot)` submits,
 * `Wait(slot)` waits, and `Outputs(slot)` reads the 8 output tensors. A slot is not
 * started again before its Wait.
 */
class InferenceBackend
{
 public:
  virtual ~InferenceBackend() = default;

  virtual std::size_t Slots() const = 0;
  virtual uint8_t* Input(std::size_t slot) = 0;
  virtual bool Start(std::size_t slot) = 0;
  virtual bool Wait(std::size_t slot) = 0;
  virtual ArmorV4::Outputs Outputs(std::size_t slot) = 0;
};
