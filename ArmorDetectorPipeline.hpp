#pragma once

/**
 * @file ArmorDetectorPipeline.hpp
 * @brief ArmorDetector template implementation aggregator.
 *
 * ArmorDetector.hpp 在类声明之后包含本文件，使各阶段实现保持拆分，同时保留
 * header-only template 实例化能力。
 */

#include "ArmorDetectorRuntime.hpp"
#include "ArmorDetectorInference.hpp"
#include "ArmorDetectorRefine.hpp"
#include "ArmorDetectorPublish.hpp"
