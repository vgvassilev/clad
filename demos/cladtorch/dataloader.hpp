#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <glob.h>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpt2 {

// Error-checking utility functions similar to the C version
namespace utils {
inline FILE* fopen_check(const char* path, const char* mode) {
  FILE* fp = fopen(path, mode);
  if (fp == nullptr)
    throw std::runtime_error("Failed to open file: " + std::string(path) +
                             " with mode: " + std::string(mode));
  return fp;
}

inline void fread_check(void* ptr, size_t size, size_t nmemb, FILE* stream) {
  size_t result = fread(ptr, size, nmemb, stream);
  if (result != nmemb) {
    if (feof(stream)) {
      throw std::runtime_error("Unexpected end of file");
    } else if (ferror(stream)) {
      throw std::runtime_error("File read error");
    } else {
      throw std::runtime_error("Partial read. Expected " +
                               std::to_string(nmemb) + " elements, read " +
                               std::to_string(result));
    }
  }
}

inline void fseek_check(FILE* fp, long off, int whence) {
  if (fseek(fp, off, whence) != 0) {
    throw std::runtime_error(
        "Failed to seek in file. Offset: " + std::to_string(off) +
        ", Whence: " + std::to_string(whence));
  }
}
} // namespace utils

class DataLoader {
private:
  static constexpr int HEADER_SIZE = 256;
  static constexpr uint32_t MAGIC_NUMBER = 20240520;
  static constexpr int DATA_VERSION = 1;

  // Distributed training variables
  int process_rank_;
  int num_processes_;

  // Batch and token information
  size_t B_;
  size_t T_;
  size_t num_tokens_;
  size_t shard_num_samples_;

  // Shards and current position
  glob_t glob_result_;
  size_t current_shard_idx_;
  size_t current_sample_idx_;

  // File handle
  std::unique_ptr<FILE, decltype(&fclose)> tokens_file_;

  // Data buffers
  std::vector<uint16_t> buffer_;
  std::vector<int> inputs_;
  std::vector<int> targets_;

  // Random shuffle related variables
  std::mt19937 rng_;
  // mt19937_state shuffle_rng_;
  bool should_shuffle_;
  std::vector<int> shard_indices_;
  std::vector<int> intra_shard_indices_;

  // Sizes in bytes
  size_t total_batch_size_bytes_;
  size_t local_batch_offset_bytes_;
  size_t header_bytes_;
  int64_t file_size_bytes_;

  bool init_ok_;

  void validate_file_header(FILE* file) {
    int header[HEADER_SIZE];
    utils::fread_check(header, sizeof(int), HEADER_SIZE, file);
    if (header[0] != MAGIC_NUMBER) {
      throw std::runtime_error(
          "Bad magic number in data file. "
          "Are you passing in a correct file? "
          "The data encoding may have changed, re-run data preprocessing.");
    }
    if (header[1] != DATA_VERSION)
      throw std::runtime_error("Bad version in data file");
  }

  int64_t load_shard(int shard_index) {
    if (should_shuffle_)
      shard_index = shard_indices_[shard_index];

    const char* filename = glob_result_.gl_pathv[shard_index];

    // Close previous file if open
    if (tokens_file_)
      tokens_file_.reset();

    // Open new file
    FILE* file = utils::fopen_check(filename, "rb");
    tokens_file_.reset(file);

    // Validate header
    validate_file_header(file);

    // Get number of tokens from header[2]
    utils::fseek_check(file, 2 * sizeof(int), SEEK_SET);
    int ntok_int;
    utils::fread_check(&ntok_int, sizeof(int), 1, file);
    int64_t ntok = ntok_int;

    if (ntok <= 0)
      throw std::runtime_error("Invalid token count in data file");

    // Determine file size and validate consistency
    utils::fseek_check(file, 0, SEEK_END);
    file_size_bytes_ = ftell(file);
    utils::fseek_check(file, 0, SEEK_SET);

    int64_t expected_file_size =
        HEADER_SIZE * sizeof(int) + ntok * sizeof(uint16_t);
    if (file_size_bytes_ != expected_file_size)
      throw std::runtime_error("File size is not as expected");

    // Calculate shard samples
    shard_num_samples_ =
        (ntok * sizeof(uint16_t) - sizeof(uint16_t)) / total_batch_size_bytes_;

    return ntok;
  }

  void prepare_intra_shard_indices() {
    if (should_shuffle_) {
      intra_shard_indices_.resize(shard_num_samples_);
      for (size_t i = 0; i < shard_num_samples_; ++i)
        intra_shard_indices_[i] = static_cast<int>(i);
      std::shuffle(intra_shard_indices_.begin(), intra_shard_indices_.end(),
                   rng_);
    }
  }

  void advance() {
    if (current_shard_idx_ == glob_result_.gl_pathc - 1) {
      reset();
      return;
    }

    current_shard_idx_ = (current_shard_idx_ + 1) % glob_result_.gl_pathc;
    current_sample_idx_ = 0;
    load_shard(static_cast<int>(current_shard_idx_));

    if (should_shuffle_)
      prepare_intra_shard_indices();
  }

public:
  DataLoader(const std::string& filename_pattern, size_t B, size_t T,
             int process_rank = 0, int num_processes = 1,
             bool should_shuffle = false)
      : process_rank_(process_rank), num_processes_(num_processes), B_(B),
        T_(T), shard_num_samples_(0), current_shard_idx_(0),
        current_sample_idx_(0), tokens_file_(nullptr, fclose),
        rng_(37 + process_rank_), should_shuffle_(should_shuffle),
        total_batch_size_bytes_(num_processes_ * B_ * T_ * sizeof(uint16_t)),
        local_batch_offset_bytes_(process_rank_ * B_ * T_ * sizeof(uint16_t)),
        header_bytes_(HEADER_SIZE * sizeof(int)), file_size_bytes_(0),
        init_ok_(false) {
    std::memset(&glob_result_, 0, sizeof(glob_result_));

    // Glob to get list of files
    int glob_status = glob(filename_pattern.c_str(), 0, nullptr, &glob_result_);
    if (glob_status != 0)
      throw std::runtime_error("Failed to glob pattern: " + filename_pattern);

    if (glob_result_.gl_pathc == 0)
      throw std::runtime_error("No files found matching pattern: " +
                               filename_pattern);

    // Initialize shuffle state if needed
    if (should_shuffle_) {
      // rng_.seed(42 + process_rank_);
      shard_indices_.resize(glob_result_.gl_pathc);
      for (size_t i = 0; i < glob_result_.gl_pathc; ++i)
        shard_indices_[i] = static_cast<int>(i);
    }

    // Validate all shards
    int64_t ntok_total = 0;
    for (size_t shard_index = 0; shard_index < glob_result_.gl_pathc;
         ++shard_index) {
      int64_t shard_ntok = load_shard(static_cast<int>(shard_index));
      if (shard_ntok < static_cast<int64_t>(num_processes_ * B_ * T_ + 1))
        throw std::runtime_error("Shard has insufficient tokens");
      ntok_total += shard_ntok;
    }

    // Allocate buffers
    buffer_.resize(B_ * T_ + 1);
    inputs_.resize(B_ * T_);
    targets_.resize(B_ * T_);
    num_tokens_ = ntok_total;

    init_ok_ = true;
    reset();
  }

  ~DataLoader() {
    if (init_ok_)
      globfree(&glob_result_);
  }

  // Disable copy operations
  DataLoader(const DataLoader&) = delete;
  DataLoader& operator=(const DataLoader&) = delete;

  // Enable move operations
  DataLoader(DataLoader&& other) noexcept = default;
  DataLoader& operator=(DataLoader&& other) noexcept = default;

  void reset() {
    if (!init_ok_)
      throw std::runtime_error("DataLoader not initialized");

    current_shard_idx_ = 0;
    current_sample_idx_ = 0;

    if (should_shuffle_)
      std::shuffle(shard_indices_.begin(), shard_indices_.end(), rng_);
    load_shard(static_cast<int>(current_shard_idx_));
    if (should_shuffle_)
      prepare_intra_shard_indices();
  }

  void load_batch() {
    if (!init_ok_)
      throw std::runtime_error("DataLoader not initialized");

    if (current_sample_idx_ >= shard_num_samples_)
      throw std::runtime_error("Current sample index out of bounds");

    size_t idx =
        should_shuffle_
            ? static_cast<size_t>(intra_shard_indices_[current_sample_idx_])
            : current_sample_idx_;

    size_t global_batch_offset_bytes = idx * total_batch_size_bytes_;
    int64_t current_offset =
        header_bytes_ + global_batch_offset_bytes + local_batch_offset_bytes_;

    // Read B*T+1 tokens from file
    utils::fseek_check(tokens_file_.get(), static_cast<long>(current_offset),
                       SEEK_SET);
    utils::fread_check(buffer_.data(), sizeof(uint16_t), B_ * T_ + 1,
                       tokens_file_.get());

    // Decode buffer into inputs and targets
    for (size_t i = 0; i < B_ * T_; ++i) {
      inputs_[i] = static_cast<int>(buffer_[i]);
      targets_[i] = static_cast<int>(buffer_[i + 1]);
    }
  }

  void next_batch() {
    if (current_sample_idx_ >= shard_num_samples_)
      advance();
    load_batch();
    current_sample_idx_ += 1;
  }

  void resume(size_t current_shard_idx, size_t current_sample_idx) {
    if (!init_ok_)
      throw std::runtime_error("DataLoader not initialized");

    current_shard_idx_ = current_shard_idx;
    current_sample_idx_ = current_sample_idx;
    load_shard(static_cast<int>(current_shard_idx_));
  }

  // Getters
  size_t batch_size() const { return B_; }
  size_t sequence_length() const { return T_; }
  size_t num_tokens() const { return num_tokens_; }
  size_t current_shard_idx() const { return current_shard_idx_; }
  size_t current_sample_idx() const { return current_sample_idx_; }
  bool is_initialized() const { return init_ok_; }

  const int* inputs() const { return inputs_.data(); }
  const int* targets() const { return targets_.data(); }
};

class EvalLoader {
private:
  static constexpr int HEADER_SIZE = 256;
  static constexpr uint32_t MAGIC_NUMBER = 20240522;
  static constexpr int DATA_VERSION = 1;
  static constexpr int ASSUMED_NUM_COMPLETIONS = 4;

  // Distributed training variables
  int process_rank_;
  int num_processes_;

  // Hyperparameters
  size_t B_;
  size_t T_;

  // File handling
  std::unique_ptr<FILE, decltype(&fclose)> eval_file_;
  std::vector<uint16_t> buffer_;

  // Public variables
  int num_examples_;
  int num_batches_;
  int start_example_index_;
  int end_example_index_;
  int current_example_index_;

  // Data arrays
  std::vector<int> inputs_;
  std::vector<int> targets_;
  std::vector<char> mask_;
  std::vector<int> label_;
  int num_completions_;

  bool init_ok_;

  static constexpr int ceil_div(int m, int n) { return (m + n - 1) / n; }

  void validate_file_header(FILE* file) {
    int header[HEADER_SIZE];
    utils::fread_check(header, sizeof(int), HEADER_SIZE, file);

    if (header[0] != MAGIC_NUMBER)
      throw std::runtime_error("Bad magic number in eval file");

    if (header[1] != DATA_VERSION)
      throw std::runtime_error("Bad version in eval file");
  }

  void next_example(int example_batch_index) {
    int batch_dim_offset = example_batch_index * ASSUMED_NUM_COMPLETIONS;

    // Read example header
    uint16_t example_header[3];
    utils::fread_check(example_header, sizeof(uint16_t), 3, eval_file_.get());

    // Validate header
    if (example_header[0] != 65535)
      throw std::runtime_error("Invalid example delimiter");

    if (example_header[2] != current_example_index_)
      throw std::runtime_error("Example index mismatch");

    // Read rest of example
    size_t example_bytes = example_header[1] - sizeof(uint16_t) * 3;
    utils::fread_check(buffer_.data(), sizeof(char), example_bytes,
                       eval_file_.get());

    // Process example
    int label = static_cast<int>(buffer_[0]);
    int can_fit_examples = static_cast<int>(B_ / ASSUMED_NUM_COMPLETIONS);

    if (label < 0 || label >= ASSUMED_NUM_COMPLETIONS)
      throw std::runtime_error("Invalid label");

    if (example_batch_index >= can_fit_examples)
      throw std::runtime_error("Example batch index out of bounds");

    label_[example_batch_index] = label;

    int num_completions = static_cast<int>(buffer_[1]);
    if (num_completions != ASSUMED_NUM_COMPLETIONS)
      throw std::runtime_error("Unexpected number of completions");

    num_completions_ = num_completions;

    // Process context
    int context_length = static_cast<int>(buffer_[2]);
    uint16_t* context_tokens_start = &buffer_[3];

    if (context_length <= 0 || context_length >= static_cast<int>(T_))
      throw std::runtime_error("Invalid context length");

    for (int b = 0; b < num_completions; ++b) {
      for (int i = 0; i < context_length; ++i) {
        int boff = batch_dim_offset + b;
        int tok_cur = static_cast<int>(context_tokens_start[i]);
        inputs_[boff * T_ + i] = tok_cur;
      }
    }

    // Process completions
    uint16_t* completions_iter = buffer_.data() + 3 + context_length;
    for (int c = 0; c < num_completions; ++c) {
      int coff = batch_dim_offset + c;
      int completion_length = static_cast<int>(completions_iter[0]);
      uint16_t* completion_tokens_start = completions_iter + 1;

      if (completion_length <= 0 ||
          context_length + completion_length >= static_cast<int>(T_))
        throw std::runtime_error("Invalid completion length");

      for (int i = 0; i < completion_length; ++i) {
        int tok_cur = static_cast<int>(completion_tokens_start[i]);
        inputs_[coff * T_ + context_length + i] = tok_cur;
        targets_[coff * T_ + context_length + i - 1] = tok_cur;
        mask_[coff * T_ + context_length + i - 1] = 1;
      }

      completions_iter += 1 + completion_length;
    }

    current_example_index_ += 1;
  }

public:
  EvalLoader()
      : process_rank_(0), num_processes_(1), B_(0), T_(0),
        eval_file_(nullptr, fclose), num_examples_(0), num_batches_(0),
        start_example_index_(0), end_example_index_(0),
        current_example_index_(0), num_completions_(0), init_ok_(false) {}

  explicit EvalLoader(const std::string& filename, size_t B, size_t T,
                      int process_rank = 0, int num_processes = 1)
      : EvalLoader() {
    init(filename, B, T, process_rank, num_processes);
  }

  // Disable copy operations
  EvalLoader(const EvalLoader&) = delete;
  EvalLoader& operator=(const EvalLoader&) = delete;

  // Enable move operations
  EvalLoader(EvalLoader&& other) noexcept = default;
  EvalLoader& operator=(EvalLoader&& other) noexcept = default;

  void init(const std::string& filename, size_t B, size_t T,
            int process_rank = 0, int num_processes = 1) {
    process_rank_ = process_rank;
    num_processes_ = num_processes;
    B_ = B;
    T_ = T;

    // Open file
    FILE* file = utils::fopen_check(filename.c_str(), "rb");
    eval_file_.reset(file);

    // Validate header
    validate_file_header(file);

    // Get file info from header
    utils::fseek_check(file, 2 * sizeof(int), SEEK_SET);

    utils::fread_check(&num_examples_, sizeof(int), 1, file);

    if (num_examples_ < num_processes_)
      throw std::runtime_error("Not enough examples for all processes");

    int longest_example_bytes;
    utils::fread_check(&longest_example_bytes, sizeof(int), 1, file);

    if (longest_example_bytes <= 0 ||
        longest_example_bytes >=
            static_cast<int>((1 + ASSUMED_NUM_COMPLETIONS) * T_ * 2))
      throw std::runtime_error("Invalid longest example size");

    // Allocate buffers
    int can_fit_examples = static_cast<int>(B_ / ASSUMED_NUM_COMPLETIONS);
    buffer_.resize(longest_example_bytes);
    inputs_.resize(B_ * T_);
    targets_.resize(B_ * T_);
    mask_.resize(B_ * T_);
    label_.resize(can_fit_examples);

    init_ok_ = true;
    reset();
  }

  void reset() {
    if (!init_ok_)
      throw std::runtime_error("EvalLoader not initialized");

    int examples_per_process = ceil_div(num_examples_, num_processes_);
    int can_fit_examples = static_cast<int>(B_ / ASSUMED_NUM_COMPLETIONS);

    if (can_fit_examples == 0) {
      throw std::runtime_error(
          "Batch size too small for assumed number of completions. "
          "Disable HellaSwag eval or increase batch size.");
    }

    num_batches_ = ceil_div(examples_per_process, can_fit_examples);

    start_example_index_ = examples_per_process * process_rank_;
    end_example_index_ = examples_per_process * (process_rank_ + 1);

    if (end_example_index_ > num_examples_)
      end_example_index_ = num_examples_;

    // Seek to start example
    int64_t header_bytes = HEADER_SIZE * sizeof(int);
    utils::fseek_check(eval_file_.get(), static_cast<long>(header_bytes),
                       SEEK_SET);

    for (int i = 0; i < start_example_index_; ++i) {
      uint16_t example_header[3];
      utils::fread_check(example_header, sizeof(uint16_t), 3, eval_file_.get());

      if (example_header[0] != 65535)
        throw std::runtime_error("Invalid example delimiter during seek");

      if (example_header[2] != i)
        throw std::runtime_error("Example index mismatch during seek");

      size_t remaining_bytes = example_header[1] - sizeof(uint16_t) * 3;
      if (remaining_bytes == 0)
        throw std::runtime_error("Invalid remaining bytes in example");
      utils::fseek_check(eval_file_.get(), static_cast<long>(remaining_bytes),
                         SEEK_CUR);
    }

    current_example_index_ = start_example_index_;
  }

  void next_batch() {
    if (!init_ok_)
      throw std::runtime_error("EvalLoader not initialized");

    // Clear mask
    std::fill(mask_.begin(), mask_.end(), 0);

    int can_fit_examples = static_cast<int>(B_ / ASSUMED_NUM_COMPLETIONS);
    for (int i = 0; i < can_fit_examples; ++i) {
      if (current_example_index_ >= end_example_index_)
        break;
      next_example(i);
    }
  }

  int stat_losses(const float* losses) const {
    if (!init_ok_)
      throw std::runtime_error("EvalLoader not initialized");

    int correct = 0;
    int can_fit_examples = static_cast<int>(B_ / ASSUMED_NUM_COMPLETIONS);

    for (int i = 0; i < can_fit_examples; ++i) {
      float min_loss = 0.0f;
      int min_loss_index = -1;
      bool active = false;

      for (int b = 0; b < ASSUMED_NUM_COMPLETIONS; ++b) {
        int boff = i * ASSUMED_NUM_COMPLETIONS + b;

        float average_loss = 0.0f;
        int count = 0;

        for (size_t t = 0; t < T_; ++t) {
          if (mask_[boff * T_ + t] == 1) {
            active = true;
            average_loss += losses[boff * T_ + t];
            count++;
          }
        }

        if (count > 0)
          average_loss /= count;

        if (b == 0 || average_loss < min_loss) {
          min_loss = average_loss;
          min_loss_index = b;
        }
      }

      if (active && min_loss_index == label_[i])
        correct += 1;
    }

    return correct;
  }

  // Getters
  size_t batch_size() const { return B_; }
  size_t sequence_length() const { return T_; }
  int num_examples() const { return num_examples_; }
  int num_batches() const { return num_batches_; }
  int current_example_index() const { return current_example_index_; }
  bool is_initialized() const { return init_ok_; }

  const int* inputs() const { return inputs_.data(); }
  const int* targets() const { return targets_.data(); }
  const char* mask() const { return mask_.data(); }
  const int* labels() const { return label_.data(); }
};

} // namespace gpt2