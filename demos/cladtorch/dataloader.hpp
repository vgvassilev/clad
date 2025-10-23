#ifndef DATALOADER_HPP
#define DATALOADER_HPP

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

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-*, *-avoid-c-arrays)
namespace gpt2 {

// Error-checking utility functions similar to the C version
namespace utils {
inline FILE* fopen_check(const char* path, const char* mode) {
  FILE* fp = fopen(path, mode); // NOLINT
  if (fp == nullptr)
    throw std::runtime_error("Failed to open file: " + std::string(path) +
                             " with mode: " + std::string(mode));
  return fp;
}

inline void fread_check(void* ptr, size_t size, size_t nmemb, FILE* stream) {
  size_t result = fread(ptr, size, nmemb, stream);
  if (result != nmemb) {
    if (feof(stream))
      throw std::runtime_error("Unexpected end of file");
    }
    if (ferror(stream)) {
      throw std::runtime_error("File read error");
    }
    throw std::runtime_error("Partial read. Expected " +
                              std::to_string(nmemb) + " elements, read " +
                              std::to_string(result));
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
  int m_ProcessRank;
  int m_NumProcesses;

  // Batch and token information
  size_t m_B;
  size_t m_T;
  size_t m_num_tokens;
  size_t m_shard_num_samples{};

  // Shards and current position
  glob_t m_glob_result{};
  size_t m_current_shard_idx{};
  size_t m_current_sample_idx{};

  // File handle
  std::unique_ptr<FILE, decltype(&fclose)> m_TokenFile;

  // Data buffers
  std::vector<uint16_t> m_Buffer;
  std::vector<int> m_inputs;
  std::vector<int> m_targets;

  // Random shuffle related variables
  std::mt19937 m_rng;
  // mt19937_state shuffle_rng_;
  bool m_should_shuffle;
  std::vector<int> m_shard_indices;
  std::vector<int> m_intra_shard_indices;

  // Sizes in bytes
  size_t m_total_batch_size_bytes;
  size_t m_local_batch_offset_bytes;
  size_t m_header_bytes;
  uint64_t m_file_size_bytes{};

  bool m_init_ok{};

  static void validate_file_header(FILE* file) {
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

  uint64_t load_shard(int shard_index) {
    if (m_should_shuffle)
      shard_index = m_shard_indices[shard_index];

    const char* filename = m_GlobResult.gl_pathv[shard_index];

    // Close previous file if open
    if (m_TokenFile)
      m_TokenFile.reset();

    // Open new file
    FILE* file = utils::fopen_check(filename, /*mode=*/"rb");
    m_tokens_file.reset(file);

    // Validate header
    validate_file_header(file);

    // Get number of tokens from header[2]
    utils::fseek_check(file, 2 * sizeof(int), SEEK_SET);
    int ntok_int = 0;
    utils::fread_check(&ntok_int, sizeof(int), /*nmemb=*/1, file);
    int64_t ntok = ntok_int;

    if (ntok <= 0)
      throw std::runtime_error("Invalid token count in data file");

    // Determine file size and validate consistency
    utils::fseek_check(file, /*off=*/0, SEEK_END);
    m_file_size_bytes = ftell(file);
    utils::fseek_check(file, /*off=*/0, SEEK_SET);

    uint64_t expected_file_size =
        HEADER_SIZE * sizeof(int) + ntok * sizeof(uint16_t);
    if (m_file_size_bytes != expected_file_size)
      throw std::runtime_error("File size is not as expected");

    // Calculate shard samples
    m_ShardNumSamples =
        (ntok * sizeof(uint16_t) - sizeof(uint16_t)) / m_total_batch_size_bytes;

    return ntok;
  }

  void prepare_intra_shard_indices() {
    if (m_should_shuffle) {
      m_intra_shard_indices.resize(m_ShardNumSamples);
      for (size_t i = 0; i < m_ShardNumSamples; ++i)
        m_intra_shard_indices[i] = static_cast<int>(i);
      std::shuffle(m_intra_shard_indices.begin(), m_intra_shard_indices.end(),
                   m_rng);
    }
  }

  void advance() {
    if (m_CurrentShardIdx == m_GlobResult.gl_pathc - 1) {
      reset();
      return;
    }

    m_CurrentShardIdx = (m_CurrentShardIdx + 1) % m_GlobResult.gl_pathc;
    m_CurrentSampleIdx = 0;
    load_shard(static_cast<int>(m_CurrentShardIdx));

    if (m_should_shuffle)
      prepare_intra_shard_indices();
  }

public:
  DataLoader(const std::string& filename_pattern, size_t B, size_t T,
             int process_rank = 0, int num_processes = 1,
             bool should_shuffle = false)
      : m_ProcessRank(process_rank), m_NumProcesses(num_processes), m_B(B),
        m_T(T), m_TokenFile(nullptr, fclose), m_rng(37 + m_ProcessRank),
        m_should_shuffle(should_shuffle),
        m_total_batch_size_bytes(m_NumProcesses * m_B * m_T * sizeof(uint16_t)),
        m_local_batch_offset_bytes(m_ProcessRank * m_B * m_T *
                                   sizeof(uint16_t)),
        m_header_bytes(HEADER_SIZE * sizeof(int)) {
    std::memset(&m_glob_result, 0, sizeof(m_glob_result));

    // Glob to get list of files
    int glob_status = glob(filename_pattern.c_str(), 0, nullptr, &m_GlobResult);
    if (glob_status != 0)
      throw std::runtime_error("Failed to glob pattern: " + filename_pattern);

    if (m_GlobResult.gl_pathc == 0)
      throw std::runtime_error("No files found matching pattern: " +
                               filename_pattern);

    // Initialize shuffle state if needed
    if (m_should_shuffle) {
      // rng_.seed(42 + process_rank_);
      m_shard_indices.resize(m_GlobResult.gl_pathc);
      for (size_t i = 0; i < m_GlobResult.gl_pathc; ++i)
        m_shard_indices[i] = static_cast<int>(i);
    }

    // Validate all shards
    uint64_t ntok_total = 0;
    for (size_t shard_index = 0; shard_index < m_glob_result.gl_pathc;
         ++shard_index) {
      uint64_t shard_ntok = load_shard(static_cast<int>(shard_index));
      if (shard_ntok < static_cast<uint64_t>(m_num_processes * m_B * m_T + 1))
        throw std::runtime_error("Shard has insufficient tokens");
      ntok_total += shard_ntok;
    }

    // Allocate buffers
    m_Buffer.resize(m_B * m_T + 1);
    m_inputs.resize(m_B * m_T);
    m_targets.resize(m_B * m_T);
    m_NumTokens = ntok_total;

    m_init_ok = true;
    reset();
  }

  ~DataLoader() {
    if (m_init_ok)
      globfree(&m_GlobResult);
  }

  // Disable copy operations
  DataLoader(const DataLoader&) = delete;
  DataLoader& operator=(const DataLoader&) = delete;

  // Enable move operations
  DataLoader(DataLoader&& other) noexcept = default;
  DataLoader& operator=(DataLoader&& other) noexcept = default;

  void reset() {
    if (!m_init_ok)
      throw std::runtime_error("DataLoader not initialized");

    m_CurrentShardIdx = 0;
    m_CurrentSampleIdx = 0;

    if (m_should_shuffle)
      std::shuffle(m_shard_indices.begin(), m_shard_indices.end(), m_rng);
    load_shard(static_cast<int>(m_CurrentShardIdx));
    if (m_should_shuffle)
      prepare_intra_shard_indices();
  }

  void load_batch() {
    if (!m_init_ok)
      throw std::runtime_error("DataLoader not initialized");

    if (m_CurrentSampleIdx >= m_ShardNumSamples)
      throw std::runtime_error("Current sample index out of bounds");

    size_t idx =
        m_should_shuffle
            ? static_cast<size_t>(m_intra_shard_indices[m_CurrentSampleIdx])
            : m_CurrentSampleIdx;

    size_t global_batch_offset_bytes = idx * m_total_batch_size_bytes;
    uint64_t current_offset =
        m_header_bytes + global_batch_offset_bytes + m_local_batch_offset_bytes;

    // Read B*T+1 tokens from file
    utils::fseek_check(m_TokenFile.get(), static_cast<long>(current_offset),
                       SEEK_SET);
    utils::fread_check(m_Buffer.data(), sizeof(uint16_t), m_B * m_T + 1,
                       m_TokenFile.get());

    // Decode buffer into inputs and targets
    for (size_t i = 0; i < m_B * m_T; ++i) {
      m_inputs[i] = static_cast<int>(m_Buffer[i]);
      m_targets[i] = static_cast<int>(m_Buffer[i + 1]);
    }
  }

  void next_batch() {
    if (m_CurrentSampleIdx >= m_ShardNumSamples)
      advance();
    load_batch();
    m_CurrentSampleIdx += 1;
  }

  void resume(size_t current_shard_idx, size_t current_sample_idx) {
    if (!m_init_ok)
      throw std::runtime_error("DataLoader not initialized");

    m_CurrentShardIdx = current_shard_idx;
    m_CurrentSampleIdx = current_sample_idx;
    load_shard(static_cast<int>(m_CurrentShardIdx));
  }

  // Getters
  size_t batch_size() const { return m_B; }
  size_t sequence_length() const { return m_T; }
  size_t num_tokens() const { return m_NumTokens; }
  size_t current_shard_idx() const { return m_CurrentShardIdx; }
  size_t current_sample_idx() const { return m_CurrentSampleIdx; }
  bool is_initialized() const { return m_init_ok; }

  const int* inputs() const { return m_inputs.data(); }
  const int* targets() const { return m_targets.data(); }
};

class EvalLoader {
private:
  static constexpr int HEADER_SIZE = 256;
  static constexpr uint32_t MAGIC_NUMBER = 20240522;
  static constexpr int DATA_VERSION = 1;
  static constexpr int ASSUMED_NUM_COMPLETIONS = 4;

  // Distributed training variables
  int m_process_rank;
  int m_num_processes;

  // Hyperparameters
  size_t m_B;
  size_t m_T;

  // File handling
  std::unique_ptr<FILE, decltype(&fclose)> m_eval_file;
  std::vector<uint16_t> m_buffer;

  // Public variables
  int m_num_examples;
  int m_num_batches;
  int m_start_example_index;
  int m_end_example_index;
  int m_current_example_index;

  // Data arrays
  std::vector<int> m_inputs;
  std::vector<int> m_targets;
  std::vector<char> m_mask;
  std::vector<int> m_label;
  int m_num_completions;

  bool m_init_ok;

  static constexpr int ceil_div(int m, int n) { return (m + n - 1) / n; }

  static void validate_file_header(FILE* file) {
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
    utils::fread_check(example_header, sizeof(uint16_t), /*nmemb=*/3, m_eval_file.get());

    // Validate header
    if (example_header[0] != 65535)
      throw std::runtime_error("Invalid example delimiter");

    if (example_header[2] != m_current_example_index)
      throw std::runtime_error("Example index mismatch");

    // Read rest of example
    size_t example_bytes = example_header[1] - sizeof(uint16_t) * 3;
    utils::fread_check(m_buffer.data(), sizeof(char), example_bytes,
                       m_eval_file.get());

    // Process example
    int label = static_cast<int>(m_buffer[0]);
    int can_fit_examples = static_cast<int>(m_B / ASSUMED_NUM_COMPLETIONS);

    if (label < 0 || label >= ASSUMED_NUM_COMPLETIONS)
      throw std::runtime_error("Invalid label");

    if (example_batch_index >= can_fit_examples)
      throw std::runtime_error("Example batch index out of bounds");

    m_label[example_batch_index] = label;

    int num_completions = static_cast<int>(m_buffer[1]);
    if (num_completions != ASSUMED_NUM_COMPLETIONS)
      throw std::runtime_error("Unexpected number of completions");

    m_num_completions = num_completions;

    // Process context
    int context_length = static_cast<int>(m_buffer[2]);
    uint16_t* context_tokens_start = &m_buffer[3];

    if (context_length <= 0 || context_length >= static_cast<int>(m_T))
      throw std::runtime_error("Invalid context length");

    for (int b = 0; b < num_completions; ++b) {
      for (int i = 0; i < context_length; ++i) {
        int boff = batch_dim_offset + b;
        int tok_cur = static_cast<int>(context_tokens_start[i]);
        m_inputs[boff * m_T + i] = tok_cur;
      }
    }

    // Process completions
    uint16_t* completions_iter = m_buffer.data() + 3 + context_length;
    for (int c = 0; c < num_completions; ++c) {
      int coff = batch_dim_offset + c;
      int completion_length = static_cast<int>(completions_iter[0]);
      uint16_t* completion_tokens_start = completions_iter + 1;

      if (completion_length <= 0 ||
          context_length + completion_length >= static_cast<int>(m_T))
        throw std::runtime_error("Invalid completion length");

      for (int i = 0; i < completion_length; ++i) {
        int tok_cur = static_cast<int>(completion_tokens_start[i]);
        m_inputs[coff * m_T + context_length + i] = tok_cur;
        m_targets[coff * m_T + context_length + i - 1] = tok_cur;
        m_mask[coff * m_T + context_length + i - 1] = 1;
      }

      completions_iter += 1 + completion_length;
    }

    m_current_example_index += 1;
  }

public:
  EvalLoader()
      : m_process_rank(0), m_num_processes(1), m_B(0), m_T(0),
        m_eval_file(nullptr, fclose), m_num_examples(0), m_num_batches(0),
        m_start_example_index(0), m_end_example_index(0),
        m_current_example_index(0), m_num_completions(0), m_init_ok(false) {}

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
  ~EvalLoader() = default;

  void init(const std::string& filename, size_t B, size_t T,
            int process_rank = 0, int num_processes = 1) {
    m_process_rank = process_rank;
    m_num_processes = num_processes;
    m_B = B;
    m_T = T;

    // Open file
    FILE* file = utils::fopen_check(filename.c_str(), /*mode=*/"rb");
    m_eval_file.reset(file);

    // Validate header
    validate_file_header(file);

    // Get file info from header
    utils::fseek_check(file, 2 * sizeof(int), SEEK_SET);

    utils::fread_check(&m_num_examples, sizeof(int), /*nmemb=*/1, file);

    if (m_num_examples < m_num_processes)
      throw std::runtime_error("Not enough examples for all processes");

    int longest_example_bytes = 0;
    utils::fread_check(&longest_example_bytes, sizeof(int), /*nmemb=*/1, file);

    if (longest_example_bytes <= 0 ||
        longest_example_bytes >=
            static_cast<int>((1 + ASSUMED_NUM_COMPLETIONS) * m_T * 2))
      throw std::runtime_error("Invalid longest example size");

    // Allocate buffers
    int can_fit_examples = static_cast<int>(m_B / ASSUMED_NUM_COMPLETIONS);
    m_buffer.resize(longest_example_bytes);
    m_inputs.resize(m_B * m_T);
    m_targets.resize(m_B * m_T);
    m_mask.resize(m_B * m_T);
    m_label.resize(can_fit_examples);

    m_init_ok = true;
    reset();
  }

  void reset() {
    if (!m_init_ok)
      throw std::runtime_error("EvalLoader not initialized");

    int examples_per_process = ceil_div(m_num_examples, m_num_processes);
    int can_fit_examples = static_cast<int>(m_B / ASSUMED_NUM_COMPLETIONS);

    if (can_fit_examples == 0) {
      throw std::runtime_error(
          "Batch size too small for assumed number of completions. "
          "Disable HellaSwag eval or increase batch size.");
    }

    m_num_batches = ceil_div(examples_per_process, can_fit_examples);

    m_start_example_index = examples_per_process * m_process_rank;
    m_end_example_index = examples_per_process * (m_process_rank + 1);

    if (m_end_example_index > m_num_examples)
      m_end_example_index = m_num_examples;

    // Seek to start example
    int64_t header_bytes = HEADER_SIZE * sizeof(int);
    utils::fseek_check(m_eval_file.get(), static_cast<long>(header_bytes),
                       SEEK_SET);

    for (int i = 0; i < m_start_example_index; ++i) {
      uint16_t example_header[3];
      utils::fread_check(example_header, sizeof(uint16_t), /*nmemb=*/3, m_eval_file.get());

      if (example_header[0] != 65535)
        throw std::runtime_error("Invalid example delimiter during seek");

      if (example_header[2] != i)
        throw std::runtime_error("Example index mismatch during seek");

      size_t remaining_bytes = example_header[1] - sizeof(uint16_t) * 3;
      if (remaining_bytes == 0)
        throw std::runtime_error("Invalid remaining bytes in example");
      utils::fseek_check(m_eval_file.get(), static_cast<long>(remaining_bytes),
                         SEEK_CUR);
    }

    m_current_example_index = m_start_example_index;
  }

  void next_batch() {
    if (!m_init_ok)
      throw std::runtime_error("EvalLoader not initialized");

    // Clear mask
    std::fill(m_mask.begin(), m_mask.end(), 0);

    int can_fit_examples = static_cast<int>(m_B / ASSUMED_NUM_COMPLETIONS);
    for (int i = 0; i < can_fit_examples; ++i) {
      if (m_current_example_index >= m_end_example_index)
        break;
      next_example(i);
    }
  }

  int stat_losses(const float* losses) const {
    if (!m_init_ok)
      throw std::runtime_error("EvalLoader not initialized");

    int correct = 0;
    int can_fit_examples = static_cast<int>(m_B / ASSUMED_NUM_COMPLETIONS);

    for (int i = 0; i < can_fit_examples; ++i) {
      float min_loss = 0.0F;
      int min_loss_index = -1;
      bool active = false;

      for (int b = 0; b < ASSUMED_NUM_COMPLETIONS; ++b) {
        int boff = i * ASSUMED_NUM_COMPLETIONS + b;

        float average_loss = 0.0F;
        int count = 0;

        for (size_t t = 0; t < m_T; ++t) {
          if (m_mask[boff * m_T + t] == 1) {
            active = true;
            average_loss += losses[boff * m_T + t];
            count++;
          }
        }

        if (count > 0)
          average_loss /= static_cast<float>(count);

        if (b == 0 || average_loss < min_loss) {
          min_loss = average_loss;
          min_loss_index = b;
        }
      }

      if (active && min_loss_index == m_label[i])
        correct += 1;
    }

    return correct;
  }

  // Getters
  size_t batch_size() const { return m_B; }
  size_t sequence_length() const { return m_T; }
  int num_examples() const { return m_num_examples; }
  int num_batches() const { return m_num_batches; }
  int current_example_index() const { return m_current_example_index; }
  bool is_initialized() const { return m_init_ok; }

  const int* inputs() const { return m_inputs.data(); }
  const int* targets() const { return m_targets.data(); }
  const char* mask() const { return m_mask.data(); }
  const int* labels() const { return m_label.data(); }
};

} // namespace gpt2
// NOLINTEND(cppcoreguidelines-pro-bounds-*, *-avoid-c-arrays)

#endif // DATALOADER_HPP
