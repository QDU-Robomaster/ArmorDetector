#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "../ArmorDetectorPipeline.hpp"

namespace
{

using armor_detector_pipeline::CompletionMark;
using armor_detector_pipeline::FixedSpscQueue;
using armor_detector_pipeline::NextGeneration;
using armor_detector_pipeline::OrderedAsyncCompletions;
using armor_detector_pipeline::TryHandoffBufferOwnership;
using armor_detector_pipeline::WorkItem;

void Expect(bool condition, const std::string& message)
{
  if (!condition)
  {
    throw std::runtime_error(message);
  }
}

template <typename Actual, typename Expected>
void ExpectEqual(const Actual& actual, const Expected& expected,
                 const std::string& message)
{
  if (!(actual == expected))
  {
    std::ostringstream stream;
    const auto append = [&stream](const auto& value)
    {
      using Value = std::remove_cvref_t<decltype(value)>;
      if constexpr (std::is_enum_v<Value>)
      {
        stream << +static_cast<std::underlying_type_t<Value>>(value);
      }
      else
      {
        stream << value;
      }
    };
    stream << message << " actual=";
    append(actual);
    stream << " expected=";
    append(expected);
    throw std::runtime_error(stream.str());
  }
}

void TestNextGenerationSkipsZeroOnWrap()
{
  constexpr uint64_t maximum = std::numeric_limits<uint64_t>::max();
  static_assert(NextGeneration(0U) == 1U);
  static_assert(NextGeneration(1U) == 2U);
  static_assert(NextGeneration(maximum - 1U) == maximum);
  static_assert(NextGeneration(maximum) == 1U);

  ExpectEqual(NextGeneration(maximum), uint64_t{1}, "generation wrap must skip zero");
}

void TestFixedSpscQueueCapacityFifoAndWraparound()
{
  FixedSpscQueue<int, 3> queue;
  int value = 0;

  Expect(queue.Empty(), "new queue must be empty");
  ExpectEqual(queue.Size(), std::size_t{0}, "new queue size");
  Expect(!queue.TryPop(value), "empty queue pop must fail");

  Expect(queue.TryPush(10), "push 10");
  Expect(queue.TryPush(11), "push 11");
  Expect(queue.TryPush(12), "push 12");
  ExpectEqual(queue.Size(), std::size_t{3}, "queue must expose full capacity");
  Expect(!queue.TryPush(99), "fourth push into capacity-three queue must fail");

  Expect(queue.TryPop(value), "pop 10");
  ExpectEqual(value, 10, "first FIFO value");
  Expect(queue.TryPop(value), "pop 11");
  ExpectEqual(value, 11, "second FIFO value");
  ExpectEqual(queue.Size(), std::size_t{1}, "size after two pops");

  Expect(queue.TryPush(13), "push 13 across ring boundary");
  Expect(queue.TryPush(14), "push 14 across ring boundary");
  ExpectEqual(queue.Size(), std::size_t{3}, "wrapped queue size");

  for (const int expected : {12, 13, 14})
  {
    Expect(queue.TryPop(value), "wrapped FIFO pop");
    ExpectEqual(value, expected, "wrapped FIFO order");
  }
  Expect(queue.Empty(), "queue must be empty after wrapped drain");
  ExpectEqual(queue.Size(), std::size_t{0}, "drained queue size");
  Expect(!queue.TryPop(value), "drained queue pop must fail");
}

void TestFixedSpscQueueWaitPopWakes()
{
  using namespace std::chrono_literals;

  FixedSpscQueue<int, 1> queue;
  auto consumer = std::async(std::launch::async, [&queue]() { return queue.WaitPop(); });

  Expect(consumer.wait_for(10ms) == std::future_status::timeout,
         "consumer must block while the queue is empty");
  Expect(queue.TryPush(77), "push must wake the blocked consumer");
  Expect(consumer.wait_for(1s) == std::future_status::ready,
         "blocked consumer must wake after push");
  ExpectEqual(consumer.get(), 77, "woken consumer value");
  Expect(queue.Empty(), "queue must be empty after blocking pop");
}

void TestFixedSpscQueueWaitPushWakes()
{
  using namespace std::chrono_literals;

  FixedSpscQueue<int, 1> queue;
  Expect(queue.TryPush(41), "fill queue before blocking producer");
  auto producer = std::async(std::launch::async,
                             [&queue]()
                             {
                               queue.WaitPush(42);
                               return true;
                             });

  Expect(producer.wait_for(10ms) == std::future_status::timeout,
         "producer must block while the queue is full");
  int value = 0;
  Expect(queue.TryPop(value), "pop must make room for blocked producer");
  ExpectEqual(value, 41, "value before blocked push");
  Expect(producer.wait_for(1s) == std::future_status::ready,
         "blocked producer must wake after pop");
  Expect(producer.get(), "blocked producer completion");
  Expect(queue.TryPop(value), "pop value written by blocked producer");
  ExpectEqual(value, 42, "value after blocked push");
}

template <std::size_t Size>
void ExpectIdPermutation(std::array<std::size_t, Size> ids, const std::string& message)
{
  std::sort(ids.begin(), ids.end());
  for (std::size_t index = 0; index < ids.size(); ++index)
  {
    ExpectEqual(ids[index], index, message);
  }
}

struct HandoffContext
{
  std::shared_ptr<int> image{};
  uint64_t frame_timestamp_us{0};
  WorkItem infer_identity{};
};

void TestBufferOwnershipHandoffMaintainsPermutationAndContext()
{
  std::array<std::size_t, 3> infer_buffer_ids{0, 1, 2};
  std::array<std::size_t, 2> post_buffer_ids{3, 4};
  std::array<HandoffContext, 2> infer_contexts{};
  std::array<HandoffContext, 2> post_contexts{};
  const std::array images{std::make_shared<int>(7), std::make_shared<int>(8)};
  for (std::size_t index = 0; index < infer_contexts.size(); ++index)
  {
    infer_contexts[index].image = images[index];
    infer_contexts[index].frame_timestamp_us = 1200U + index;
    infer_contexts[index].infer_identity = {.slot_id = static_cast<uint8_t>(index),
                                            .generation = 9U + index};
    Expect(TryHandoffBufferOwnership(infer_buffer_ids[index], post_buffer_ids[index], 5U,
                                     infer_contexts[index], post_contexts[index]),
           "valid handoff must succeed");
    Expect(!infer_contexts[index].image, "infer context must be empty after handoff");
    Expect(post_contexts[index].image == images[index],
           "post context must retain image ownership");
    ExpectEqual(post_contexts[index].frame_timestamp_us, uint64_t{1200U + index},
                "frame timestamp must move with context");
    Expect(
        armor_detector_pipeline::SameWorkItem(
            post_contexts[index].infer_identity,
            WorkItem{.slot_id = static_cast<uint8_t>(index), .generation = 9U + index}),
        "original infer identity must move with context");
    ExpectIdPermutation(
        std::array{infer_buffer_ids[0], infer_buffer_ids[1], infer_buffer_ids[2],
                   post_buffer_ids[0], post_buffer_ids[1]},
        "handoff must retain the five-buffer permutation");
  }

  const auto replacement = std::make_shared<int>(9);
  infer_contexts[0].image = replacement;
  infer_contexts[0].frame_timestamp_us = 1300U;
  infer_contexts[0].infer_identity = {.slot_id = 0, .generation = 10U};
  post_contexts[0] = {};
  Expect(infer_contexts[0].image == replacement,
         "retiring old post work must not touch the reused infer context");
  Expect(TryHandoffBufferOwnership(infer_buffer_ids[2], post_buffer_ids[0], 5U,
                                   infer_contexts[0], post_contexts[0]),
         "prepared infer slot must hand off through a reused post slot");
  ExpectIdPermutation(
      std::array{infer_buffer_ids[0], infer_buffer_ids[1], infer_buffer_ids[2],
                 post_buffer_ids[0], post_buffer_ids[1]},
      "reused handoff must retain the five-buffer permutation");

  std::size_t invalid_infer_id = 5U;
  std::size_t valid_post_id = 0U;
  HandoffContext invalid_context{.image = std::make_shared<int>(10)};
  HandoffContext untouched_post{};
  Expect(!TryHandoffBufferOwnership(invalid_infer_id, valid_post_id, 5U, invalid_context,
                                    untouched_post),
         "out-of-range handoff must fail");
  ExpectEqual(invalid_infer_id, std::size_t{5}, "failed handoff infer ID");
  ExpectEqual(valid_post_id, std::size_t{0}, "failed handoff post ID");
  Expect(invalid_context.image && !untouched_post.image,
         "failed handoff must not move context ownership");
}

void TestOrderedAsyncCompletionsRetiresInSubmissionOrder()
{
  OrderedAsyncCompletions<2> completions;
  const WorkItem first{.slot_id = 0, .generation = 7};
  const WorkItem second{.slot_id = 1, .generation = 8};

  Expect(completions.TryRegister(first), "register first async request");
  Expect(completions.TryRegister(second), "register second async request");
  Expect(!completions.TryRegister({.slot_id = 0, .generation = 9}),
         "two-request ledger must reject a third in-flight request");
  ExpectEqual(completions.Size(), std::size_t{2}, "registered async count");

  Expect(completions.MarkCompleted(second, true) == CompletionMark::MARKED,
         "mark second complete first");
  Expect(!completions.PopCompleted().has_value(),
         "completed suffix must wait for the FIFO head");

  Expect(completions.MarkCompleted(first, true) == CompletionMark::MARKED,
         "mark FIFO head complete");
  for (const auto& expected : {first, second})
  {
    const auto retired = completions.PopCompleted();
    Expect(retired.has_value(), "continuous completed prefix must retire");
    Expect(armor_detector_pipeline::SameWorkItem(retired->item, expected),
           "retirement must preserve submission order");
    Expect(retired->ok, "successful completion must preserve status");
  }
  Expect(completions.Empty(), "ordered completion ledger must drain");
}

void TestOrderedAsyncCompletionsRejectsGenerationMismatchAndDuplicate()
{
  OrderedAsyncCompletions<2> completions;
  const WorkItem current{.slot_id = 3, .generation = 11};
  const WorkItem stale{.slot_id = 3, .generation = 10};

  Expect(completions.TryRegister(current), "register current generation");
  Expect(!completions.TryRegister(current), "duplicate registration must fail");
  Expect(completions.MarkCompleted(stale, true) == CompletionMark::NOT_FOUND,
         "stale generation completion must not match current slot");
  Expect(completions.MarkCompleted(current, true) == CompletionMark::MARKED,
         "current generation completion must match");
  Expect(completions.MarkCompleted(current, true) == CompletionMark::DUPLICATE,
         "duplicate completion must be rejected");

  const auto retired = completions.PopCompleted();
  Expect(retired.has_value() && retired->ok,
         "current generation must retire exactly once");
  Expect(!completions.PopCompleted().has_value(),
         "duplicate callback must not create a second retirement");
}

void TestOrderedAsyncCompletionsFailureAdvancesHead()
{
  OrderedAsyncCompletions<2> completions;
  const WorkItem failed{.slot_id = 0, .generation = 1};
  const WorkItem success{.slot_id = 1, .generation = 1};
  Expect(completions.TryRegister(failed), "register failing request");
  Expect(completions.TryRegister(success), "register following request");
  Expect(!completions.TryRegister({.slot_id = 2, .generation = 1}),
         "ledger capacity must be enforced");

  Expect(completions.MarkCompleted(success, true) == CompletionMark::MARKED,
         "mark following success");
  Expect(completions.MarkCompleted(failed, false) == CompletionMark::MARKED,
         "mark head failure");
  const auto first = completions.PopCompleted();
  const auto second = completions.PopCompleted();
  Expect(first.has_value() && !first->ok,
         "failed head must retire instead of blocking the queue");
  Expect(second.has_value() && second->ok, "success behind failed head must then retire");
}

void TestCompletedSuccessWaitsWithoutLosingLedgerHead()
{
  OrderedAsyncCompletions<2> completions;
  const WorkItem item{.slot_id = 0, .generation = 3};
  Expect(completions.TryRegister(item), "register successful request");
  Expect(completions.MarkCompleted(item, true) == CompletionMark::MARKED,
         "mark successful request");

  const auto waiting = completions.PeekCompleted();
  Expect(waiting.has_value() && waiting->ok,
         "completed success must remain visible while post slots are busy");
  ExpectEqual(completions.Size(), std::size_t{1},
              "peeking must not retire a successful completion");
  const auto retired = completions.PopCompleted();
  Expect(
      retired.has_value() && armor_detector_pipeline::SameWorkItem(retired->item, item),
      "successful completion must retire after post capacity is available");
}

void TestFailedCompletionRetiresWithoutPostCapacity()
{
  OrderedAsyncCompletions<2> completions;
  const WorkItem item{.slot_id = 1, .generation = 5};
  Expect(completions.TryRegister(item), "register failing request");
  Expect(completions.MarkCompleted(item, false) == CompletionMark::MARKED,
         "mark failing request");

  const auto failed = completions.PeekCompleted();
  Expect(failed.has_value() && !failed->ok,
         "failed completion must be identifiable without a post slot");
  const auto retired = completions.PopCompleted();
  Expect(retired.has_value() && !retired->ok, "failed completion must retire directly");
  Expect(completions.Empty(), "failed completion must not hold infer ownership");
}

using TestFunction = void (*)();

struct NamedTest
{
  const char* name;
  TestFunction function;
};

}  // namespace

int main()
{
  constexpr std::array tests{
      NamedTest{"generation/wrap-skips-zero", TestNextGenerationSkipsZeroOnWrap},
      NamedTest{"spsc/capacity-fifo-wrap", TestFixedSpscQueueCapacityFifoAndWraparound},
      NamedTest{"spsc/wait-pop-wakes", TestFixedSpscQueueWaitPopWakes},
      NamedTest{"spsc/wait-push-wakes", TestFixedSpscQueueWaitPushWakes},
      NamedTest{"pipeline/buffer-handoff-permutation",
                TestBufferOwnershipHandoffMaintainsPermutationAndContext},
      NamedTest{"async/order-preserved",
                TestOrderedAsyncCompletionsRetiresInSubmissionOrder},
      NamedTest{"async/generation-and-duplicate",
                TestOrderedAsyncCompletionsRejectsGenerationMismatchAndDuplicate},
      NamedTest{"async/failure-advances-head",
                TestOrderedAsyncCompletionsFailureAdvancesHead},
      NamedTest{"async/success-waits-for-post",
                TestCompletedSuccessWaitsWithoutLosingLedgerHead},
      NamedTest{"async/failure-needs-no-post",
                TestFailedCompletionRetiresWithoutPostCapacity},
  };

  try
  {
    for (const auto& test : tests)
    {
      test.function();
      std::cout << "[PASS] " << test.name << '\n';
    }
    return 0;
  }
  catch (const std::exception& exception)
  {
    std::cerr << "[FAIL] " << exception.what() << '\n';
    return 1;
  }
}
