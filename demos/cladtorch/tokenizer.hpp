#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpt2 {

class Tokenizer {
private:
  static constexpr uint32_t MAGIC_NUMBER = 20240328;
  static constexpr int HEADER_SIZE = 256;

  uint32_t m_vocab_size = 0;
  int m_eot_token = 0;
  std::vector<std::string> m_token_table;

  void read_tokenizer_file(FILE* file) {
    // Read header
    uint32_t header[HEADER_SIZE]; // NOLINT
    if (fread(header, sizeof(uint32_t), HEADER_SIZE, file) !=
        HEADER_SIZE) // NOLINT
      throw std::runtime_error("Failed to read tokenizer header");
    if (header[0] != MAGIC_NUMBER)
      throw std::runtime_error("Invalid magic number in tokenizer file");
    auto version = header[1];
    m_vocab_size = header[2];

    if (version == 1) {
      // Version 1 didn't include the EOT token id
      // so we assume it is 50256, the EOT in GPT-2
      if (m_vocab_size != 50257) {
        throw std::runtime_error(
            "Expected vocab_size 50257 for tokenizer version 1");
      }
      m_eot_token = 50256;
    } else if (version == 2)
      m_eot_token = static_cast<int>(header[3]);
    else
      throw std::runtime_error("Unsupported tokenizer version: " +
                               std::to_string(version));

    // Read all tokens
    m_token_table.reserve(m_vocab_size);
    for (uint32_t i = 0; i < m_vocab_size; ++i) {
      unsigned char length = 0;
      if (fread(&length, sizeof(unsigned char), 1, file) != 1)
        throw std::runtime_error("Failed to read token length");
      if (length == 0)
        throw std::runtime_error("Invalid token length: 0");

      std::string token(length, '\0');
      if (fread(token.data(), sizeof(char), length, file) != length)
        throw std::runtime_error("Failed to read token data");
      m_token_table.emplace_back(std::move(token));
    }
  }

public:
  explicit Tokenizer(const std::string& filename) { load(filename); }
  // Disable copy operations for simplicity
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  // Enable move operations
  Tokenizer(Tokenizer&&) = default;
  Tokenizer& operator=(Tokenizer&&) = default;
  ~Tokenizer() = default;

  void load(const std::string& filename) {
    auto file = std::unique_ptr<FILE, decltype(&fclose)>(
        fopen(filename.c_str(), "rb"), fclose);
    if (!file) {
      std::cerr << "---\n";
      std::cerr << "WARNING: Failed to open the tokenizer file " << filename
                << '\n';
      std::cerr << "The Tokenizer is a new feature added April 14 2024.\n";
      std::cerr << "Re-run `python train_gpt2.py` to write it\n";
      std::cerr << "---\n";
      throw std::runtime_error("Failed to open tokenizer file");
    }
    try {
      read_tokenizer_file(file.get());
      std::cerr << "Tokenizer loaded: " << m_vocab_size << " tokens from "
                << filename << '\n';
    } catch (const std::exception& e) {
      std::cerr << "Error loading tokenizer: " << e.what() << '\n';
      throw;
    }
  }

  std::string decode(uint32_t token_id) const {
    if (token_id >= m_vocab_size)
      throw std::runtime_error("Invalid token id: " + std::to_string(token_id));
    return m_token_table[token_id];
  }

  void safe_print(uint32_t token_id) const {
    if (token_id >= m_vocab_size) {
      std::cerr << "Invalid token id " << token_id << "!\n";
      return;
    }
    const std::string& token = m_token_table[token_id];
    if (token.empty())
      return;
    // Handle individual byte tokens
    if (token.length() == 1) {
      unsigned char byte_val = static_cast<unsigned char>(token[0]);
      if (!(std::isprint(byte_val) || std::isspace(byte_val)))
        return; // weird byte, don't print it
    }
    std::cout << token;
  }

  // Getters
  uint32_t vocab_size() const { return m_vocab_size; }
  int eot_token() const { return m_eot_token; }
};

} // namespace gpt2

#endif // TOKENIZER_HPP