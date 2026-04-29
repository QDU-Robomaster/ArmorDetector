#pragma once

// 仅供 ArmorDetector.hpp 在类声明之后包含。
template <CameraTypes::CameraInfo CameraInfoV>
void ArmorDetector<CameraInfoV>::ShowDebugPreview(const cv::Mat& bgr_img,
                                                  const cv::Mat* binary_debug)
{
  try
  {
    // 预览渲染只能读工作帧，不能修改同步模块背后的共享图像槽位。
    const cv::Mat preview_bgr = bgr_img.clone();

    cv::Mat canvas(preview_bgr.rows, preview_bgr.cols + detail::info_panel_width, CV_8UC3,
                   cv::Scalar(18, 22, 28));
    preview_bgr.copyTo(
        canvas(cv::Rect(0, 0, preview_bgr.cols, preview_bgr.rows)));

    cv::Mat header =
        canvas(cv::Rect(0, 0, preview_bgr.cols, detail::preview_header_height));
    cv::Mat header_overlay = header.clone();
    cv::rectangle(header_overlay, cv::Rect(0, 0, header.cols, header.rows),
                  cv::Scalar(10, 16, 22), cv::FILLED);
    cv::addWeighted(header_overlay, detail::header_bar_alpha, header,
                    1.0 - detail::header_bar_alpha, 0.0, header);

    cv::putText(canvas, "ArmorDetector Preview", cv::Point(18, 34),
                cv::FONT_HERSHEY_DUPLEX, 0.85, cv::Scalar(240, 244, 250), 1,
                cv::LINE_AA);
    cv::putText(canvas,
                cfg_.yolo.use_roi ? "YOLOv5 + OpenVINO [ROI]" : "YOLOv5 + OpenVINO [FULL]",
                cv::Point(18, 52), cv::FONT_HERSHEY_DUPLEX, 0.48,
                cv::Scalar(151, 170, 192), 1, cv::LINE_AA);

    for (std::size_t index = 0; index < armors_msg_.results.size(); ++index)
    {
      const auto& armor = armors_msg_.results[index];
      const cv::Scalar armor_color = detail::color_to_scalar(armor.color);

      std::array<cv::Point, 4> polygon{};
      for (std::size_t point_index = 0; point_index < armor.points.size(); ++point_index)
      {
        polygon[point_index] = armor.points[point_index];
      }

      const cv::Point* polygon_points = polygon.data();
      const int polygon_size = static_cast<int>(polygon.size());
      cv::polylines(canvas, &polygon_points, &polygon_size, 1, true, armor_color, 2,
                    cv::LINE_AA);
      cv::rectangle(canvas, armor.box, armor_color, 1, cv::LINE_AA);

      for (const auto& point : armor.points)
      {
        cv::circle(canvas, point, static_cast<int>(detail::point_radius), armor_color,
                   cv::FILLED, cv::LINE_AA);
      }

      std::ostringstream label;
      label << detail::armor_number_to_string(armor.number) << " "
            << detail::armor_type_to_string(armor.type) << " "
            << std::fixed << std::setprecision(2) << armor.confidence;
      const cv::Point label_origin(
          std::max(armor.box.x, 6),
          std::max(armor.box.y - 6, 24));
      detail::draw_label_chip(canvas, label.str(), label_origin, armor_color);
    }

    const int panel_x = preview_bgr.cols + 18;
    int panel_y = 42;
    cv::putText(canvas, "Frame Stats", cv::Point(panel_x, panel_y),
                cv::FONT_HERSHEY_DUPLEX, 0.72, cv::Scalar(243, 246, 250), 1,
                cv::LINE_AA);
    panel_y += 28;
    detail::draw_info_row(canvas, panel_x, panel_y, "frame",
                          std::to_string(metrics_msg_.frame_index),
                          cv::Scalar(240, 244, 250));
    panel_y += 24;
    detail::draw_info_row(canvas, panel_x, panel_y, "armors",
                          std::to_string(metrics_msg_.armor_count),
                          cv::Scalar(128, 226, 142));
    panel_y += 24;
    detail::draw_info_row(canvas, panel_x, panel_y, "refined",
                          std::to_string(metrics_msg_.refined_count),
                          cv::Scalar(91, 196, 255));
    panel_y += 24;
    detail::draw_info_row(canvas, panel_x, panel_y, "discarded",
                          std::to_string(metrics_msg_.discarded_count),
                          cv::Scalar(255, 166, 77));
    panel_y += 24;
    detail::draw_info_row(canvas, panel_x, panel_y, "detector_ms",
                          detail::format_float(metrics_msg_.detector_latency_ms, 2),
                          cv::Scalar(255, 214, 102));
    panel_y += 24;
    detail::draw_info_row(canvas, panel_x, panel_y, "publish_ms",
                          detail::format_float(metrics_msg_.publish_latency_ms, 2),
                          cv::Scalar(221, 235, 255));

    panel_y += 36;
    cv::putText(canvas, "Detector Config", cv::Point(panel_x, panel_y),
                cv::FONT_HERSHEY_DUPLEX, 0.72, cv::Scalar(243, 246, 250), 1,
                cv::LINE_AA);
    panel_y += 28;
    const ArmorColor configured_target_color =
        detail::detect_color_from_config(cfg_.detect_color);
    detail::draw_info_row(canvas, panel_x, panel_y, "target_color",
                          detail::target_color_name(configured_target_color),
                          cv::Scalar(240, 244, 250));
    panel_y += 24;
    detail::draw_info_row(canvas, panel_x, panel_y, "score_thres",
                          detail::format_float(cfg_.yolo.score_threshold, 2),
                          cv::Scalar(240, 244, 250));
    panel_y += 24;
    detail::draw_info_row(canvas, panel_x, panel_y, "min_conf",
                          detail::format_float(cfg_.yolo.min_confidence, 2),
                          cv::Scalar(240, 244, 250));
    panel_y += 24;
    detail::draw_info_row(canvas, panel_x, panel_y, "traditional_refine",
                          cfg_.yolo.use_traditional_refine ? "on" : "off",
                          cfg_.yolo.use_traditional_refine
                              ? cv::Scalar(128, 226, 142)
                              : cv::Scalar(255, 166, 77));

    panel_y += 36;
    cv::putText(canvas, "Detections", cv::Point(panel_x, panel_y),
                cv::FONT_HERSHEY_DUPLEX, 0.72, cv::Scalar(243, 246, 250), 1,
                cv::LINE_AA);
    panel_y += 26;

    if (armors_msg_.results.empty())
    {
      cv::putText(canvas, "No target in current frame", cv::Point(panel_x, panel_y),
                  cv::FONT_HERSHEY_DUPLEX, 0.54, cv::Scalar(151, 170, 192), 1,
                  cv::LINE_AA);
    }
    else
    {
      const int armor_count = std::min(static_cast<int>(armors_msg_.results.size()),
                                       detail::max_debug_armors);
      for (int index = 0; index < armor_count; ++index)
      {
        const auto& armor = armors_msg_.results[index];
        const cv::Scalar armor_color = detail::color_to_scalar(armor.color);
        const cv::Rect item_rect(panel_x - 10, panel_y - 18,
                                 detail::info_panel_width - 32, 52);
        cv::rectangle(canvas, item_rect, cv::Scalar(32, 39, 48), cv::FILLED,
                      cv::LINE_AA);
        cv::rectangle(canvas, item_rect, armor_color, 1, cv::LINE_AA);

        const std::string name =
            detail::armor_display_name(armor.number, armor.type);
        cv::putText(canvas, name, cv::Point(panel_x, panel_y), cv::FONT_HERSHEY_DUPLEX,
                    0.56, cv::Scalar(245, 247, 250), 1, cv::LINE_AA);

        std::ostringstream value;
        value << "conf=" << std::fixed << std::setprecision(2) << armor.confidence
              << "  px=" << static_cast<int>(armor.center.x) << ","
              << static_cast<int>(armor.center.y);
        cv::putText(canvas, value.str(), cv::Point(panel_x, panel_y + 20),
                    cv::FONT_HERSHEY_DUPLEX, 0.44, armor_color, 1, cv::LINE_AA);
        panel_y += 60;
      }
    }

    if (binary_debug != nullptr && !binary_debug->empty())
    {
      cv::Mat preview_binary;
      cv::cvtColor(*binary_debug, preview_binary, cv::COLOR_GRAY2BGR);

      const int thumb_width = detail::info_panel_width - 36;
      const int thumb_height = std::max(120, thumb_width * preview_binary.rows /
                                                 std::max(1, preview_binary.cols));
      cv::Mat thumb_resized;
      cv::resize(preview_binary, thumb_resized, cv::Size(thumb_width, thumb_height));

      int thumb_y =
          std::max(preview_bgr.rows - thumb_resized.rows - 26, panel_y + 16);
      if ((thumb_y + thumb_resized.rows) > canvas.rows)
      {
        thumb_y = canvas.rows - thumb_resized.rows - 16;
      }
      cv::putText(canvas, "Binary Threshold", cv::Point(panel_x, thumb_y - 8),
                  cv::FONT_HERSHEY_DUPLEX, 0.56, cv::Scalar(243, 246, 250), 1,
                  cv::LINE_AA);
      thumb_resized.copyTo(
          canvas(cv::Rect(panel_x, thumb_y, thumb_resized.cols, thumb_resized.rows)));
      cv::rectangle(canvas, cv::Rect(panel_x, thumb_y, thumb_resized.cols, thumb_resized.rows),
                    cv::Scalar(91, 196, 255), 1, cv::LINE_AA);
    }

    cv::Mat display = canvas;
    if (std::abs(cfg_.debug.overlay_scale - 1.0) > 1e-6)
    {
      cv::resize(canvas, display, cv::Size(), cfg_.debug.overlay_scale,
                 cfg_.debug.overlay_scale);
    }

    cv::imshow("armor_detector_debug", display);
    cv::waitKey(std::max(cfg_.debug.wait_key_ms, 1));
  }
  catch (const cv::Exception& exception)
  {
    preview_available_ = false;
    if (!preview_warned_)
    {
      preview_warned_ = true;
      XR_LOG_WARN("ArmorDetector preview disabled: %s", exception.what());
    }
  }
}

template <CameraTypes::CameraInfo CameraInfoV>
bool ArmorDetector<CameraInfoV>::ShouldShowPreview()
{
  if (!cfg_.debug.preview || !preview_available_)
  {
    return false;
  }

  const char* display = std::getenv("DISPLAY");
  const char* wayland_display = std::getenv("WAYLAND_DISPLAY");
  if (display == nullptr && wayland_display == nullptr)
  {
    preview_available_ = false;
    if (!preview_warned_)
    {
      preview_warned_ = true;
      XR_LOG_WARN("ArmorDetector preview disabled because DISPLAY is unavailable");
    }
    return false;
  }

  return true;
}
