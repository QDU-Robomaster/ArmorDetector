#include "ArmorDetector.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace
{
void Check(bool condition, const char* message)
{
  if (!condition)
  {
    throw std::runtime_error(message);
  }
}

void CheckModelContract()
{
  static_assert(static_cast<unsigned>(ArmorDetectorModel::INT8_HEAD_L) == 0U);
  static_assert(static_cast<unsigned>(ArmorDetectorModel::INT8_GRID_L) == 1U);
  static_assert(static_cast<unsigned>(ArmorDetectorModel::INT16_HEAD_L) == 2U);
  static_assert(static_cast<unsigned>(ArmorDetectorModel::INT8_HEAD) == 3U);
  static_assert(static_cast<unsigned>(ArmorDetectorModel::INT8_GRID) == 4U);
  static_assert(static_cast<unsigned>(ArmorDetectorModel::INT16_HEAD) == 5U);
  static_assert(static_cast<unsigned>(ArmorDetectorModel::INT16_FAST_L) == 6U);
  static_assert(static_cast<unsigned>(ArmorDetectorModel::INT16_FAST) == 7U);
  static_assert(infer::default_detector_model == ArmorDetectorModel::INT16_HEAD_L);
  for (unsigned index = 0U; index < 8U; ++index)
  {
    const auto* model = infer::resolve_detector_model(static_cast<ArmorDetectorModel>(index));
    Check(model != nullptr && model->backend == infer::DetectorBackend::HAILORT,
          "existing model/backend binding changed");
  }
  const auto* model = infer::resolve_detector_model(ArmorDetectorModel::OPENVINO_640X512);
  Check(model != nullptr && model->backend == infer::DetectorBackend::OPENVINO,
        "OpenVINO model must select OpenVINO, not a detected alternative");
  const auto& adapter = infer::resolve_model_infer_adapter(model->model);
  Check(adapter.candidate_count == 20160 && adapter.output_width == 22,
        "legacy output schema changed");
  Check(adapter.canonical_point_order == std::array<int, 4>{0, 3, 2, 1},
        "legacy corner ordering changed");
  Check(adapter.decode_color(0) == ArmorColor::BLUE &&
            adapter.decode_color(1) == ArmorColor::RED && infer::reject_raw_color(adapter, 2),
        "legacy color mapping changed");
  Check(adapter.decode_number(0) == ArmorNumber::GUARD &&
            adapter.decode_number(6) == ArmorNumber::OUTPOST &&
            adapter.decode_number(7) == ArmorNumber::BASE,
        "legacy class mapping changed");
  const float raw = 0.60F;
  const float confidence = infer::decode_confidence(adapter, raw);
  Check(confidence > 0.619F &&
            infer::objectness_prefilter_value(model->model, raw, confidence) < 0.619F,
        "OpenVINO must gate raw logits, not sigmoid probabilities");
  Check(infer::objectness_prefilter_value(ArmorDetectorModel::INT16_HEAD_L, raw,
                                         confidence) == confidence,
        "Hailo prefilter behavior changed");
  Check(infer::resolve_detector_model(static_cast<ArmorDetectorModel>(255)) == nullptr,
        "invalid enum must not resolve to another model");
}

void CheckUnavailableBackend()
{
  using Network = armor_detector_detail::ArmorDetectorNetwork;
  Network network;
  Check(!network.Configure(static_cast<ArmorDetectorModel>(255)) && !network.Ready(),
        "invalid selection must fail");
#if !defined(ARMOR_DETECTOR_HAVE_HAILORT)
  for (unsigned index = 0U; index < 8U; ++index)
  {
    Check(!network.Configure(static_cast<ArmorDetectorModel>(index)) && !network.Ready(),
          "Hailo model silently used an available alternative backend");
  }
#endif
#if !defined(ARMOR_DETECTOR_HAVE_OPENVINO)
  Check(!network.Configure(ArmorDetectorModel::OPENVINO_640X512) && !network.Ready(),
        "OpenVINO model must fail in an SDK-free build");
#endif
}

#if defined(ARMOR_DETECTOR_HAVE_OPENVINO)
void CheckActualOpenVino()
{
  using Backend = armor_detector_detail::OpenVinoArmorBackend;
  using Network = armor_detector_detail::ArmorDetectorNetwork;
  Check(Backend::SelectDevice({"CPU", "GPU.0", "NPU"}, nullptr) == "NPU",
        "historical device priority changed");
  Check(Backend::SelectDevice({"CPU", "GPU.0"}, nullptr) == "GPU.0",
        "GPU suffix selection failed");
  Check(Backend::SelectDevice({}, nullptr).empty(), "empty device list must fail");
  Check(Backend::SelectDevice({"CPU"}, "GPU") == "GPU",
        "explicit missing device must not silently become CPU");
  Check(!Backend::OutputShapeSupported({2, 20160, 22}) &&
            !Backend::OutputShapeSupported({1, 20160, 23}) &&
            !Backend::OutputShapeSupported({1, 22, 20160}) &&
            Backend::OutputShapeSupported({22, 20160}),
        "output shape validation failed");

  Network network;
  Check(network.Configure(ArmorDetectorModel::OPENVINO_640X512) && network.Ready(),
        "actual bundled OpenVINO model failed to initialize");
  Check(!network.UsesHailoRt() && network.AsyncQueueSize() == 0U &&
            network.BackendName().rfind("OPENVINO:", 0) == 0U,
        "OpenVINO selected the wrong backend/mode");
  Network::RawOutputSlot first, second;
  Check(network.InitRawOutputSlot(first) && network.InitRawOutputSlot(second),
        "per-frame inference request allocation failed");
  cv::Mat input_a, input_b;
  Check(network.InitRawInputView(first, input_a) && network.InitRawInputView(second, input_b),
        "slot input views failed");
  Check(input_a.data != input_b.data && network.IsRawInputView(first, input_a) &&
            !network.IsRawInputView(second, input_a), "input slots alias each other");
  input_a.setTo(cv::Scalar(0, 0, 0));
  cv::RNG random(20260524);
  random.fill(input_b, cv::RNG::UNIFORM, 0, 256);
  Network::HailoRawTimingSnapshot timing_a, timing_b;
  Network::HailoDecodeTimingSnapshot decode;
  cv::Mat output_a, output_b;
  Check(network.InferRaw(input_a, first, timing_a) && first.Valid() && timing_a.valid &&
            timing_a.complete_ns >= timing_a.call_begin_ns, "first inference failed");
  Check(network.DecodeRaw(first, output_a, decode) && decode.valid &&
            output_a.total() == 20160U * 22U && cv::checkRange(output_a),
        "first output is invalid");
  const cv::Mat retained = output_a.clone();
  Check(network.InferRaw(input_b, second, timing_b) &&
            network.DecodeRaw(second, output_b, decode), "second inference failed");
  Check(output_a.data != output_b.data && cv::norm(output_a, retained, cv::NORM_INF) == 0.0,
        "next inference overwrote the postprocessing slot");
  Check(cv::norm(output_b, retained, cv::NORM_INF) > 0.0,
        "distinct real inputs unexpectedly produced identical tensors");

  // Independent direct OpenVINO oracle with the original NUC HWC-u8 tensor contract.
  {
    ov::Core core;
    auto compiled = core.compile_model(
        ARMOR_DETECTOR_OPENVINO_640X512_PATH, "CPU",
        ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    auto request = compiled.create_infer_request();
    request.set_input_tensor(ov::Tensor(ov::element::u8, {512, 640, 3}, input_a.data));
    request.infer();
    auto tensor = request.get_output_tensor();
    cv::Mat oracle(20160, 22, CV_32F, tensor.data<float>());
    Check(cv::norm(retained, oracle, cv::NORM_INF) <= 1.0e-4,
          "backend output differs from direct historical-contract inference");
  }

  cv::Mat padded(512, 641, CV_8UC3);
  Check(!network.InferRaw(padded(cv::Rect(0, 0, 640, 512)), first, timing_a) &&
            !first.Valid() && !timing_a.valid, "non-contiguous input was accepted");
  Check(!network.DecodeRaw(first, output_a, decode) && output_a.empty(),
        "failed inference exposed stale output");
  Check(!network.InferRaw(cv::Mat(512, 640, CV_32FC3), first, timing_a),
        "f32 input was accepted as the model uint8 input");
  Check(!network.InferRaw(cv::Mat(511, 640, CV_8UC3), first, timing_a),
        "wrong input dimensions were accepted");
  Check(network.InferRaw(input_a, first, timing_a), "slot could not be reused after rejection");
  Check(network.DecodeRaw(first, output_a, decode), "reused slot decode failed");
  bool worker_ok = false;
  std::jthread worker([&] { worker_ok = network.InferRaw(input_b, second, timing_b); });
  for (int iteration = 0; iteration < 20; ++iteration)
  {
    cv::Mat view;
    Network::HailoDecodeTimingSnapshot local_decode;
    Check(network.DecodeRaw(first, view, local_decode) &&
              cv::norm(view, retained, cv::NORM_INF) == 0.0,
          "inference/postprocess overlap corrupted a retained output");
  }
  worker.join();
  Check(worker_ok, "overlapped inference failed");
  cv::Mat compatibility_output;
  Check(network.Infer(input_a, compatibility_output) &&
            cv::norm(compatibility_output, retained, cv::NORM_INF) <= 1.0e-4,
        "synchronous compatibility API failed");
  Check(network.Configure(ArmorDetectorModel::OPENVINO_640X512), "reconfigure failed");
  Check(!network.IsRawInputView(first, input_a) &&
            !network.InferRaw(input_a, first, timing_a),
        "a stale request survived model generation change");
  Check(network.InitRawOutputSlot(first) && network.InitRawInputView(first, input_a),
        "slot reinitialization failed");
  auto missing = *infer::resolve_detector_model(ArmorDetectorModel::OPENVINO_640X512);
  missing.openvino_model_path = "/nonexistent/armor-detector-test.onnx";
  Check(!network.Configure(missing) && !network.Ready() && network.BackendName() == "NONE",
        "model-load failure silently reused a previously ready model");
  std::cout << "real OpenVINO CPU inference, direct oracle, isolated outputs, overlap, "
               "invalid inputs and stale-generation checks passed\n";
}
#endif
}  // namespace

int main(int argc, char** argv)
{
  try
  {
    Check(argc == 1 || (argc == 2 && std::string(argv[1]) == "--unavailable-device"),
          "unknown test argument");
    CheckModelContract();
    CheckUnavailableBackend();
    if (argc == 2)
    {
      armor_detector_detail::ArmorDetectorNetwork network;
      Check(!network.Configure(ArmorDetectorModel::OPENVINO_640X512) &&
                !network.Ready() && network.BackendName() == "NONE",
            "explicit unavailable OpenVINO device silently fell back");
      std::cout << "Unavailable OpenVINO device rejected without fallback\n";
      return 0;
    }
#if defined(ARMOR_DETECTOR_HAVE_OPENVINO)
    CheckActualOpenVino();
#endif
    std::cout << "ArmorDetector backend contract checks passed\n";
    return 0;
  }
  catch (const std::exception& exception)
  {
    std::cerr << "FAIL: " << exception.what() << '\n';
    return 1;
  }
}
