#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <type_traits>
#include <utility>

namespace armor_detector_pipeline
{

enum class InferSlotState : uint8_t
{
  FREE = 0,
  PREPARING,
  INFER_QUEUED,
  INFER_RUNNING,
  OUTPUT_QUEUED,
  OUTPUT_RUNNING,
};

enum class PostSlotState : uint8_t
{
  FREE = 0,
  FUSING,
  POST_QUEUED,
  POST_RUNNING,
  POST_DONE,
};

struct WorkItem
{
  uint8_t slot_id{0};
  uint64_t generation{0};
};

constexpr bool SameWorkItem(const WorkItem& lhs, const WorkItem& rhs)
{
  return lhs.slot_id == rhs.slot_id && lhs.generation == rhs.generation;
}

enum class CompletionMark : uint8_t
{
  MARKED = 0,
  NOT_FOUND,
  DUPLICATE,
};

struct CompletedWorkItem
{
  WorkItem item{};
  bool ok{false};
};

/** Fixed-capacity submission-order ledger with out-of-order completion marking.
 */
template <std::size_t Capacity>
class OrderedAsyncCompletions
{
 public:
  static_assert(Capacity > 0, "OrderedAsyncCompletions requires non-zero capacity");

  bool TryRegister(const WorkItem& item)
  {
    if (item.generation == 0U || size_ == Capacity || Find(item) != Capacity)
    {
      return false;
    }

    entries_[Index(size_)] = Entry{.item = item};
    ++size_;
    return true;
  }

  CompletionMark MarkCompleted(const WorkItem& item, bool ok)
  {
    const std::size_t offset = Find(item);
    if (offset == Capacity)
    {
      return CompletionMark::NOT_FOUND;
    }

    Entry& entry = entries_[Index(offset)];
    if (entry.completed)
    {
      return CompletionMark::DUPLICATE;
    }
    entry.completed = true;
    entry.ok = ok;
    return CompletionMark::MARKED;
  }

  [[nodiscard]] bool FrontCompleted() const
  {
    return size_ != 0U && entries_[head_].completed;
  }

  [[nodiscard]] std::optional<CompletedWorkItem> PeekCompleted() const
  {
    if (!FrontCompleted())
    {
      return std::nullopt;
    }

    const Entry& entry = entries_[head_];
    return CompletedWorkItem{.item = entry.item, .ok = entry.ok};
  }

  std::optional<CompletedWorkItem> PopCompleted()
  {
    if (!FrontCompleted())
    {
      return std::nullopt;
    }

    const Entry entry = entries_[head_];
    entries_[head_] = {};
    head_ = (head_ + 1U) % Capacity;
    --size_;
    return CompletedWorkItem{.item = entry.item, .ok = entry.ok};
  }

  [[nodiscard]] bool Empty() const { return size_ == 0U; }
  [[nodiscard]] std::size_t Size() const { return size_; }

 private:
  struct Entry
  {
    WorkItem item{};
    bool completed{false};
    bool ok{false};
  };

  [[nodiscard]] std::size_t Index(std::size_t offset) const
  {
    return (head_ + offset) % Capacity;
  }

  [[nodiscard]] std::size_t Find(const WorkItem& item) const
  {
    for (std::size_t offset = 0; offset < size_; ++offset)
    {
      if (SameWorkItem(entries_[Index(offset)].item, item))
      {
        return offset;
      }
    }
    return Capacity;
  }

  std::array<Entry, Capacity> entries_{};
  std::size_t head_{0};
  std::size_t size_{0};
};

constexpr uint64_t NextGeneration(uint64_t generation)
{
  ++generation;
  return generation == 0U ? 1U : generation;
}

template <typename Context>
bool TryHandoffBufferOwnership(std::size_t& infer_buffer_id, std::size_t& post_buffer_id,
                               std::size_t buffer_count, Context& infer_context,
                               Context& post_context) noexcept
{
  static_assert(std::is_nothrow_default_constructible_v<Context>);
  static_assert(std::is_nothrow_move_assignable_v<Context>);

  if (infer_buffer_id >= buffer_count || post_buffer_id >= buffer_count ||
      infer_buffer_id == post_buffer_id)
  {
    return false;
  }

  std::swap(infer_buffer_id, post_buffer_id);
  post_context = std::move(infer_context);
  infer_context = {};
  return true;
}

template <typename T, std::size_t Capacity>
class FixedSpscQueue
{
 public:
  static_assert(Capacity > 0, "FixedSpscQueue requires non-zero capacity");

  bool TryPush(const T& value)
  {
    const std::size_t head = head_.load(std::memory_order_relaxed);
    const std::size_t next = Next(head);
    if (next == tail_.load(std::memory_order_acquire))
    {
      return false;
    }

    storage_[head] = value;
    head_.store(next, std::memory_order_release);
    {
      std::lock_guard<std::mutex> lock(wait_mutex_);
      wait_cv_.notify_one();
    }
    return true;
  }

  void WaitPush(const T& value)
  {
    while (!TryPush(value))
    {
      std::unique_lock<std::mutex> lock(wait_mutex_);
      not_full_cv_.wait(lock, [this]() { return Size() < Capacity; });
    }
  }

  bool TryPop(T& value)
  {
    const std::size_t tail = tail_.load(std::memory_order_relaxed);
    if (tail == head_.load(std::memory_order_acquire))
    {
      return false;
    }

    value = storage_[tail];
    tail_.store(Next(tail), std::memory_order_release);
    {
      std::lock_guard<std::mutex> lock(wait_mutex_);
      not_full_cv_.notify_one();
    }
    return true;
  }

  T WaitPop()
  {
    T value{};
    while (!TryPop(value))
    {
      std::unique_lock<std::mutex> lock(wait_mutex_);
      wait_cv_.wait(lock, [this]() { return !Empty(); });
    }
    return value;
  }

  [[nodiscard]] bool Empty() const
  {
    return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::size_t Size() const
  {
    const std::size_t head = head_.load(std::memory_order_acquire);
    const std::size_t tail = tail_.load(std::memory_order_acquire);
    return head >= tail ? head - tail : storage_.size() - tail + head;
  }

 private:
  static constexpr std::size_t Next(std::size_t index)
  {
    return (index + 1U) % (Capacity + 1U);
  }

  std::array<T, Capacity + 1U> storage_{};
  std::atomic<std::size_t> head_{0};
  std::atomic<std::size_t> tail_{0};
  mutable std::mutex wait_mutex_{};
  std::condition_variable wait_cv_{};
  std::condition_variable not_full_cv_{};
};

}  // namespace armor_detector_pipeline
