#pragma once

#include <array>
#include <cstdint>

#include "ArmorDetectorTypes.hpp"

#ifndef ARMOR_DETECTOR_INT16_FAST_L_HEF_PATH
#define ARMOR_DETECTOR_INT16_FAST_L_HEF_PATH ""
#endif

#ifndef ARMOR_DETECTOR_INT16_FAST_HEF_PATH
#define ARMOR_DETECTOR_INT16_FAST_HEF_PATH ""
#endif

enum class ArmorDetectorModel : uint8_t
{
  INT8_HEAD_L = 0,
  INT8_GRID_L = 1,
  INT16_HEAD_L = 2,
  INT8_HEAD = 3,
  INT8_GRID = 4,
  INT16_HEAD = 5,
  INT16_FAST_L = 6,
  INT16_FAST = 7,
};

namespace armor_detector_infer
{

enum class ModelLine : uint8_t
{
  INT8 = 0,
  INT16 = 1,
};

struct ResolvedDetectorModel
{
  const char* canonical_name{""};
  ArmorDetectorModel model{ArmorDetectorModel::INT16_HEAD_L};
  ModelLine line{ModelLine::INT16};
  const char* hailort_hef_path{""};
};

struct ModelInferAdapter
{
  ModelLine line{ModelLine::INT16};
  int candidate_count{0};
  int output_width{0};
  int raw_color_begin{0};
  int raw_color_end{0};
  int raw_class_begin{0};
  int raw_class_end{0};
  bool reject_aux_colors{false};
  bool confidence_is_logit{false};
  std::array<int, 4> canonical_point_order{{0, 1, 2, 3}};
  ArmorColor (*decode_color)(int){nullptr};
  ArmorNumber (*decode_number)(int){nullptr};
};

constexpr ArmorDetectorModel default_detector_model = ArmorDetectorModel::INT16_HEAD_L;

inline const char* detector_model_name(ArmorDetectorModel model)
{
  switch (model)
  {
    case ArmorDetectorModel::INT8_HEAD_L:
      return "int8-head-l";
    case ArmorDetectorModel::INT8_GRID_L:
      return "int8-grid-l";
    case ArmorDetectorModel::INT16_HEAD_L:
      return "int16-quality-l";
    case ArmorDetectorModel::INT8_HEAD:
      return "int8-head";
    case ArmorDetectorModel::INT8_GRID:
      return "int8-grid";
    case ArmorDetectorModel::INT16_HEAD:
      return "int16-quality";
    case ArmorDetectorModel::INT16_FAST_L:
      return "int16-fast-l";
    case ArmorDetectorModel::INT16_FAST:
      return "int16-fast";
    default:
      return "int16-quality-l";
  }
}

inline const char* model_line_name(ModelLine line)
{
  return line == ModelLine::INT8 ? "int8" : "int16";
}

inline const ResolvedDetectorModel* resolve_detector_model(ArmorDetectorModel model)
{
  static const ResolvedDetectorModel kVariants[] = {
      {
          "int8-head-l",
          ArmorDetectorModel::INT8_HEAD_L,
          ModelLine::INT8,
          ARMOR_DETECTOR_INT8_HEAD_L_HEF_PATH,
      },
      {
          "int8-grid-l",
          ArmorDetectorModel::INT8_GRID_L,
          ModelLine::INT8,
          ARMOR_DETECTOR_INT8_GRID_L_HEF_PATH,
      },
      {
          "int16-quality-l",
          ArmorDetectorModel::INT16_HEAD_L,
          ModelLine::INT16,
          ARMOR_DETECTOR_INT16_HEAD_L_HEF_PATH,
      },
      {
          "int16-fast-l",
          ArmorDetectorModel::INT16_FAST_L,
          ModelLine::INT16,
          ARMOR_DETECTOR_INT16_FAST_L_HEF_PATH,
      },
      {
          "int8-head",
          ArmorDetectorModel::INT8_HEAD,
          ModelLine::INT8,
          ARMOR_DETECTOR_INT8_HEAD_HEF_PATH,
      },
      {
          "int8-grid",
          ArmorDetectorModel::INT8_GRID,
          ModelLine::INT8,
          ARMOR_DETECTOR_INT8_GRID_HEF_PATH,
      },
      {
          "int16-quality",
          ArmorDetectorModel::INT16_HEAD,
          ModelLine::INT16,
          ARMOR_DETECTOR_INT16_HEAD_HEF_PATH,
      },
      {
          "int16-fast",
          ArmorDetectorModel::INT16_FAST,
          ModelLine::INT16,
          ARMOR_DETECTOR_INT16_FAST_HEF_PATH,
      },
  };

  switch (model)
  {
    case ArmorDetectorModel::INT8_HEAD_L:
      return &kVariants[0];
    case ArmorDetectorModel::INT8_GRID_L:
      return &kVariants[1];
    case ArmorDetectorModel::INT16_HEAD_L:
      return &kVariants[2];
    case ArmorDetectorModel::INT16_FAST_L:
      return &kVariants[3];
    case ArmorDetectorModel::INT8_HEAD:
      return &kVariants[4];
    case ArmorDetectorModel::INT8_GRID:
      return &kVariants[5];
    case ArmorDetectorModel::INT16_HEAD:
      return &kVariants[6];
    case ArmorDetectorModel::INT16_FAST:
      return &kVariants[7];
    default:
      return nullptr;
  }
}

inline const ResolvedDetectorModel& resolve_detector_model_or_default(
    ArmorDetectorModel model)
{
  const auto* resolved = resolve_detector_model(model);
  if (resolved != nullptr)
  {
    return *resolved;
  }
  const auto* fallback = resolve_detector_model(default_detector_model);
  return *fallback;
}

}  // namespace armor_detector_infer

namespace infer = armor_detector_infer;
