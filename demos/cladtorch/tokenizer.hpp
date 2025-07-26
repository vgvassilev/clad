#pragma once

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

  uint32_t vocab_size_;
  std::vector<std::string> token_table_;
  bool init_ok_;
  int eot_token_;

  void read_tokenizer_file(FILE* file) {
    // Read header
    uint32_t header[HEADER_SIZE];
    if (fread(header, sizeof(uint32_t), HEADER_SIZE, file) != HEADER_SIZE) {
      throw std::runtime_error("Failed to read tokenizer header");
    }

    if (header[0] != MAGIC_NUMBER) {
      throw std::runtime_error("Invalid magic number in tokenizer file");
    }

    int version = header[1];
    vocab_size_ = header[2];

    if (version == 1) {
      // Version 1 didn't include the EOT token id
      // so we assume it is 50256, the EOT in GPT-2
      if (vocab_size_ != 50257) {
        throw std::runtime_error("Expected vocab_size 50257 for tokenizer version 1");
      }
      eot_token_ = 50256;
    } else if (version == 2) {
      eot_token_ = header[3];
    } else {
      throw std::runtime_error("Unsupported tokenizer version: " + std::to_string(version));
    }

    // Read all tokens
    token_table_.reserve(vocab_size_);
    for (uint32_t i = 0; i < vocab_size_; ++i) {
      unsigned char length;
      if (fread(&length, sizeof(unsigned char), 1, file) != 1) {
        throw std::runtime_error("Failed to read token length");
      }

      if (length == 0) {
        throw std::runtime_error("Invalid token length: 0");
      }

      std::string token(length, '\0');
      if (fread(token.data(), sizeof(char), length, file) != length) {
        throw std::runtime_error("Failed to read token data");
      }

      token_table_.emplace_back(std::move(token));
    }

    init_ok_ = true;
  }

public:
  Tokenizer() : vocab_size_(0), init_ok_(false), eot_token_(0) {}

  explicit Tokenizer(const std::string& filename) : vocab_size_(0), init_ok_(false), eot_token_(0) {
    load(filename);
  }

  // Disable copy operations for simplicity
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  // Enable move operations
  Tokenizer(Tokenizer&&) = default;
  Tokenizer& operator=(Tokenizer&&) = default;

  void load(const std::string& filename) {
    auto file = std::unique_ptr<FILE, decltype(&fclose)>(fopen(filename.c_str(), "rb"), fclose);
    if (!file) {
      std::cerr << "---\n";
      std::cerr << "WARNING: Failed to open the tokenizer file " << filename << '\n';
      std::cerr << "The Tokenizer is a new feature added April 14 2024.\n";
      std::cerr << "Re-run `python train_gpt2.py` to write it\n";
      std::cerr << "---\n";
      init_ok_ = false;
      return;
    }

    try {
      read_tokenizer_file(file.get());
      std::cerr << "Tokenizer loaded: " << vocab_size_ << " tokens from " << filename << '\n';
    } catch (const std::exception& e) {
      std::cerr << "Error loading tokenizer: " << e.what() << '\n';
      init_ok_ = false;
      throw;
    }
  }

  std::string decode(uint32_t token_id) const {
    if (!init_ok_) {
      throw std::runtime_error("Tokenizer not initialized");
    }
    
    if (token_id >= vocab_size_) {
      throw std::runtime_error("Invalid token id: " + std::to_string(token_id));
    }
    
    return token_table_[token_id];
  }

  void safe_print(uint32_t token_id) const {
    if (!init_ok_) {
      return;
    }

    if (token_id >= vocab_size_) {
      std::cerr << "Invalid token id " << token_id << "!\n";
      return;
    }

    const std::string& token = token_table_[token_id];
    if (token.empty()) {
      return;
    }

    // Handle individual byte tokens
    if (token.length() == 1) {
      unsigned char byte_val = static_cast<unsigned char>(token[0]);
      if (!(std::isprint(byte_val) || std::isspace(byte_val))) {
        return; // weird byte, don't print it
      }
    }

    std::cout << token;
  }

  // Getters
  uint32_t vocab_size() const { return vocab_size_; }
  bool is_initialized() const { return init_ok_; }
  int eot_token() const { return eot_token_; }
};

} // namespace gpt2