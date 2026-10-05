#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <vector>

/**
 * @brief v4 检测模型（armor-models det-v4）的输出解码：两个尺度合并、sigmoid 门限、
 *        角点不加权、四边形外接框 NMS。与 rm_model.decode（fuse=None）逐项一致。
 *        Output decoding of the v4 detector (armor-models det-v4): both scales merged,
 *        sigmoid threshold, unweighted corners, NMS on the quad bounding boxes. Matches
 *        rm_model.decode (fuse=None) step by step.
 */
namespace ArmorV4
{
inline constexpr std::array<int, 2> STRIDES = {8, 16};
/// 角点偏移 1.0 等于 4 个格子 / A corner offset of 1.0 spans 4 cells.
inline constexpr float CORNER_UNIT = 4.0F;
inline constexpr int MAX_DETECTIONS = 50;

/**
 * @brief 一个输出张量的只读视图：浮点或量化 uint8，CHW 或 HWC。
 *        Read-only view of one output tensor: float or quantised uint8, CHW or HWC.
 */
struct TensorView
{
  enum class Type : uint8_t
  {
    F32,
    U8,
  };

  const void* data = nullptr;
  Type type = Type::F32;
  bool hwc = false;
  int height = 0;
  int width = 0;
  int channels = 0;
  float scale = 1.0F;       ///< 仅 U8：(q − zero_point) × scale / U8 only
  float zero_point = 0.0F;  ///< 仅 U8 / U8 only

  float At(int c, int y, int x) const
  {
    const std::size_t i = hwc ? (static_cast<std::size_t>(y) * width + x) * channels + c
                              : (static_cast<std::size_t>(c) * height + y) * width + x;
    return type == Type::F32
               ? static_cast<const float*>(data)[i]
               : (static_cast<float>(static_cast<const uint8_t*>(data)[i]) - zero_point) *
                     scale;
  }
};

/// 一个尺度的四个输出 / The four outputs of one scale.
struct ScaleOutputs
{
  TensorView obj;     ///< [1] logit
  TensorView color;   ///< [2] logit：0 蓝、1 红 / 0 blue, 1 red
  TensorView size;    ///< [2] logit：0 小、1 大 / 0 small, 1 large
  TensorView corner;  ///< [8] LT、LB、RB、RT 的 (x, y) 偏移 / (x, y) offsets
};

/// stride 8、16 两个尺度 / Scales of stride 8 and 16.
using Outputs = std::array<ScaleOutputs, 2>;

/// 帧坐标（640×512）下的一个检测 / One detection in frame coordinates (640×512).
struct Detection
{
  std::array<cv::Point2f, 4> corners;  ///< LT、LB、RB、RT
  float score;
  int color;  ///< 0 蓝、1 红 / 0 blue, 1 red
  int size;   ///< 0 小、1 大 / 0 small, 1 large
};

inline int ArgMax2(const TensorView& t, int y, int x)
{
  return t.At(1, y, x) > t.At(0, y, x) ? 1 : 0;
}

inline std::vector<Detection> Decode(const Outputs& outputs, float conf, float iou)
{
  std::vector<Detection> candidates;
  for (std::size_t s = 0; s < outputs.size(); ++s)
  {
    const ScaleOutputs& o = outputs[s];
    const float stride = static_cast<float>(STRIDES[s]);
    for (int i = 0; i < o.obj.height; ++i)
    {
      for (int j = 0; j < o.obj.width; ++j)
      {
        const float score = 1.0F / (1.0F + std::exp(-o.obj.At(0, i, j)));
        if (!(score > conf))
        {
          continue;
        }
        // 格心按像素中心约定 / Cell centre in the pixel-centre convention.
        const float cx = (static_cast<float>(j) + 0.5F) * stride - 0.5F;
        const float cy = (static_cast<float>(i) + 0.5F) * stride - 0.5F;
        Detection d{{}, score, ArgMax2(o.color, i, j), ArgMax2(o.size, i, j)};
        for (int k = 0; k < 4; ++k)
        {
          d.corners[k] = {o.corner.At(2 * k, i, j) * CORNER_UNIT * stride + cx,
                          o.corner.At(2 * k + 1, i, j) * CORNER_UNIT * stride + cy};
        }
        candidates.push_back(d);
      }
    }
  }
  if (candidates.empty())
  {
    return {};
  }

  std::vector<cv::Rect2d> boxes;
  std::vector<float> scores;
  for (const Detection& d : candidates)
  {
    float x0 = d.corners[0].x, x1 = x0, y0 = d.corners[0].y, y1 = y0;
    for (const cv::Point2f& p : d.corners)
    {
      x0 = std::min(x0, p.x);
      x1 = std::max(x1, p.x);
      y0 = std::min(y0, p.y);
      y1 = std::max(y1, p.y);
    }
    boxes.emplace_back(x0, y0, x1 - x0, y1 - y0);
    scores.push_back(d.score);
  }
  std::vector<int> keep;
  cv::dnn::NMSBoxes(boxes, scores, conf, iou, keep);
  std::vector<Detection> result;
  for (std::size_t n = 0; n < keep.size() && n < MAX_DETECTIONS; ++n)
  {
    result.push_back(candidates[keep[n]]);
  }
  return result;
}
}  // namespace ArmorV4
