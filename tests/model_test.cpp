// 依赖模型文件的黄金测试（OpenVINO）：与 Python 参考（ONNX Runtime + rm_model.decode +
// num-v1）的结果逐项对比，再整条模块跑一遍。没有设置 ARMOR_MODELS_DIR 与
// ARMOR_DETECTOR_GOLDEN_DIR 时跳过（返回 77）。
//
// Golden test with model files (OpenVINO): compares with the Python reference (ONNX
// Runtime + rm_model.decode + num-v1) item by item, then runs the whole Module. Skipped
// (exit 77) unless ARMOR_MODELS_DIR and ARMOR_DETECTOR_GOLDEN_DIR are set.
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "ArmorDetector.hpp"
#include "OpenVinoBackend.hpp"
#include "libxr.hpp"

namespace
{
void Expect(bool condition, const char* message)
{
  if (!condition)
  {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

struct Gold
{
  float score;
  int color;
  int size;
  int number;
  std::array<cv::Point2f, 4> corners;
};

std::map<std::string, std::vector<Gold>> ReadGold(const std::string& dir)
{
  std::map<std::string, std::vector<Gold>> gold;
  std::ifstream file(dir + "/gold.txt");
  std::string line;
  while (std::getline(file, line))
  {
    std::istringstream in(line);
    std::string frame;
    Gold g{};
    in >> frame >> g.score >> g.color >> g.size >> g.number;
    for (cv::Point2f& p : g.corners)
    {
      in >> p.x >> p.y;
    }
    gold[frame].push_back(g);
  }
  return gold;
}

std::vector<uint8_t> ReadPgm(const std::string& path)
{
  std::ifstream file(path, std::ios::binary);
  std::string magic;
  int w = 0, h = 0, max = 0;
  file >> magic >> w >> h >> max;
  file.get();
  std::vector<uint8_t> data(static_cast<std::size_t>(w) * h);
  file.read(reinterpret_cast<char*>(data.data()),
            static_cast<std::streamsize>(data.size()));
  Expect(magic == "P5" && w == 640 && h == 512 && file.good(), "golden PGM");
  return data;
}

float CornerError(const std::array<cv::Point2f, 4>& a,
                  const std::array<cv::Point2f, 4>& b)
{
  float e = 0.0F;
  for (int k = 0; k < 4; ++k)
  {
    e = std::max(e, static_cast<float>(cv::norm(a[k] - b[k])));
  }
  return e;
}

/// 后端 + 解码 + 编号与 Python 逐项一致 / Backend, decoding and numbers match Python.
void TestAgainstPython(const std::string& models,
                       const std::map<std::string, std::vector<Gold>>& gold,
                       const std::string& golden_dir)
{
  OpenVinoBackend backend(models + "/armor_det_v4.onnx", 1, "CPU");
  NumberClassifier numbers(models + "/armor_num_v1.onnx");
  float worst = 0.0F;
  std::size_t count = 0;
  for (const auto& [frame, expected] : gold)
  {
    const std::vector<uint8_t> bayer = ReadPgm(golden_dir + "/frames/" + frame);
    std::memcpy(backend.Input(0), bayer.data(), bayer.size());
    Expect(backend.Start(0) && backend.Wait(0), "inference");
    const auto dets = ArmorV4::Decode(backend.Outputs(0), 0.4F, 0.3F);
    Expect(dets.size() == expected.size(), "same number of detections");
    const cv::Mat gray = NumberClassifier::Gray(bayer.data(), 640, 512);
    for (std::size_t i = 0; i < dets.size(); ++i)
    {
      const Gold& g = expected[i];
      Expect(dets[i].color == g.color && dets[i].size == g.size, "colour and size");
      Expect(std::fabs(dets[i].score - g.score) < 1e-3F, "score");
      worst = std::max(worst, CornerError(dets[i].corners, g.corners));
      Expect(static_cast<int>(numbers.Classify(gray, dets[i].corners)) == g.number,
             "number");
      ++count;
    }
  }
  std::printf("python parity: %zu detections, worst corner error %.5f px\n", count,
              worst);
  Expect(worst < 0.02F, "corners within 0.02 px of the Python reference");
}

/// 整个模块：同步帧进，检测帧出 / The whole Module: synced frames in, detected out.
void TestModule(const std::string& models,
                const std::map<std::string, std::vector<Gold>>& gold,
                const std::string& golden_dir)
{
  LibXR::Topic synced =
      LibXR::Topic::CreateTopic<const AutoAim::SyncedFrame*>("gold_synced");
  auto* detector =
      new ArmorDetector({"gold", models, "armor_det_v4.onnx", "armor_num_v1.onnx", "CPU",
                         0.4F, 0.3F, TargetColor::RED, 2});
  struct Received
  {
    std::mutex mutex;
    std::vector<AutoAim::DetectedFrame> frames;
  };
  auto* received = new Received();
  auto callback = LibXR::Topic::Callback::Create(
      [](bool, Received* r, const AutoAim::DetectedFrame* d)
      {
        std::lock_guard<std::mutex> lock(r->mutex);
        r->frames.push_back({{d->synced.sequence, {}, d->synced.imu}, d->armors});
      },
      received);
  AutoAim::RequireTopic<const AutoAim::DetectedFrame*>("gold_detected")
      .RegisterCallback(callback);

  ImagePool pool(2);
  uint64_t sequence = 0;
  std::size_t expected_armors = 0;
  for (const auto& [frame, expected] : gold)
  {
    ImagePool::Handle writing;
    Expect(pool.Acquire(writing) == LibXR::ErrorCode::OK, "image slot");
    const std::vector<uint8_t> bayer = ReadPgm(golden_dir + "/frames/" + frame);
    std::memcpy(writing->data.data(), bayer.data(), bayer.size());
    writing->geometry = CameraBase::WIDE_GEOMETRY;
    {
      AutoAim::SyncedFrame s{++sequence, SharedFrame(std::move(writing)), {}};
      const AutoAim::SyncedFrame* payload = &s;
      synced.Publish(payload);
    }
    for (int t = 0; t < 2000; ++t)
    {
      {
        std::lock_guard<std::mutex> lock(received->mutex);
        if (received->frames.size() == sequence)
        {
          break;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::lock_guard<std::mutex> lock(received->mutex);
    if (received->frames.size() != sequence)
    {
      std::fprintf(stderr, "frame %s: sent %llu, received %zu\n", frame.c_str(),
                   static_cast<unsigned long long>(sequence), received->frames.size());
      detector->OnMonitor();
    }
    Expect(received->frames.size() == sequence, "one detected frame per synced frame");
    const auto& armors = received->frames.back().armors;
    std::vector<const Gold*> wanted;
    for (const Gold& g : expected)
    {
      if (g.color == 1 && g.number != static_cast<int>(ArmorNumber::NEGATIVE))
      {
        wanted.push_back(&g);  // 只留红色且是装甲板 / Red and a real armor only
      }
    }
    Expect(armors.size() == wanted.size(), "colour filter and negatives");
    for (std::size_t i = 0; i < armors.size(); ++i)
    {
      Expect(armors[i].color == ArmorColor::RED &&
                 static_cast<int>(armors[i].number) == wanted[i]->number,
             "armor colour and number");
      for (int k = 0; k < 4; ++k)
      {
        // 原生坐标 = 2·x + 79.5 / Native = 2x + 79.5 in WIDE.
        Expect(std::fabs(armors[i].corners[k].x -
                         (2.0F * wanted[i]->corners[k].x + 79.5F)) < 0.05F &&
                   std::fabs(armors[i].corners[k].y -
                             (2.0F * wanted[i]->corners[k].y + 23.5F)) < 0.05F,
               "corners in native pixels");
      }
    }
    expected_armors += wanted.size();
  }
  std::printf("module: %llu frames, %zu red armors\n",
              static_cast<unsigned long long>(sequence), expected_armors);
  detector->OnMonitor();
  delete detector;
}
}  // namespace

int main()
{
  const char* models = std::getenv("ARMOR_MODELS_DIR");
  const char* golden = std::getenv("ARMOR_DETECTOR_GOLDEN_DIR");
  if (models == nullptr || golden == nullptr)
  {
    std::puts(
        "armor_detector_model_test skipped: set ARMOR_MODELS_DIR and "
        "ARMOR_DETECTOR_GOLDEN_DIR");
    return 77;
  }
  LibXR::PlatformInit();
  const auto gold = ReadGold(golden);
  Expect(!gold.empty(), "golden data");
  TestAgainstPython(models, gold, golden);
  TestModule(models, gold, golden);
  std::puts("armor_detector_model_test passed");
  return 0;
}
