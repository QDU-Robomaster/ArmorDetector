#pragma once

#include <array>
#include <cmath>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include "infer/ArmorDetectorInt16Model.hpp"
#include "infer/ArmorDetectorInt8Model.hpp"

namespace armor_detector_infer
{

inline const ModelInferAdapter& resolve_model_infer_adapter(ArmorDetectorModel model)
{
  const auto& resolved = resolve_detector_model_or_default(model);
  return resolved.line == ModelLine::INT8 ? int8_model_infer_adapter()
                                          : int16_model_infer_adapter();
}

inline const ModelInferAdapter& resolve_model_infer_adapter(ModelLine line)
{
  return line == ModelLine::INT8 ? int8_model_infer_adapter()
                                 : int16_model_infer_adapter();
}

inline bool reject_raw_color(const ModelInferAdapter& adapter, int raw_color_id)
{
  return adapter.reject_aux_colors && (raw_color_id == 2 || raw_color_id == 3);
}

inline float decode_confidence(const ModelInferAdapter& adapter,
                               float objectness_logit)
{
  if (adapter.confidence_is_logit)
  {
    return objectness_logit;
  }
  return 1.0F / (1.0F + std::exp(-objectness_logit));
}

template <typename PointT>
inline std::array<PointT, 4> canonicalize_points(
    const ModelInferAdapter& adapter,
    const std::array<PointT, 4>& declared_points)
{
  Eigen::Array<float, 4, 2, Eigen::RowMajor> point_matrix;
  for (int index = 0; index < 4; ++index)
  {
    point_matrix(index, 0) = declared_points[static_cast<std::size_t>(index)].x;
    point_matrix(index, 1) = declared_points[static_cast<std::size_t>(index)].y;
  }

  Eigen::Array<float, 4, 2, Eigen::RowMajor> canonical_matrix;
  for (int index = 0; index < 4; ++index)
  {
    canonical_matrix.row(index) = point_matrix.row(
        adapter.canonical_point_order[static_cast<std::size_t>(index)]);
  }

  std::array<PointT, 4> canonical_points{};
  for (int index = 0; index < 4; ++index)
  {
    canonical_points[static_cast<std::size_t>(index)] =
        PointT(canonical_matrix(index, 0), canonical_matrix(index, 1));
  }
  return canonical_points;
}

}  // namespace armor_detector_infer
