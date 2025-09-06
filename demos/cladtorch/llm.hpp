#pragma once

#include <cassert>
#include <cladtorch/cladtorch.hpp>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace gpt2 {

namespace ct = cladtorch;
using FTensor = ct::Tensor<float>;
using ITensor = ct::Tensor<int>;

struct Config {
  int max_seq_len;
  int vocab_size;
  int padded_vocab_size;
  int num_layers;
  int num_heads;
  int channels;

  int head_size() const { return channels / num_heads; }
  int mlp_hidden_size() const { return 4 * channels; }
};

class Linear {
public:
  FTensor weight, bias;
  Linear(int in_features, int out_features)
      : weight({out_features, in_features}), bias({out_features}) {}
  FTensor forward(const FTensor& input) const {
    auto linear_out = linear(input, weight, bias);
    return linear_out;
  }
};

class LayerNorm {
public:
  FTensor weight, bias;
  explicit LayerNorm(int channels) : weight({channels}), bias({channels}) {}
  FTensor forward(const FTensor& input) const {
    auto norm_out = input.norm();
    auto weighted = norm_out * weight;
    auto result = weighted + bias;
    return result;
  }
};

class Encoder {
public:
  FTensor wte, wpe;
  Encoder(int padded_vocab_size, int max_seq_len, int channels)
      : wte({padded_vocab_size, channels}), wpe({max_seq_len, channels}) {}
  FTensor forward(const ITensor& input, const ITensor& input_pos) const {
    auto token_embeddings = wte.lookup(input);
    auto position_embeddings = wpe.lookup(input_pos);
    auto embeddings = token_embeddings + position_embeddings;
    return embeddings;
  }
};

class CausalSelfAttention {
public:
  Linear qkv, proj;
  int num_heads, channels, head_size;

  CausalSelfAttention(int num_heads, int channels)
      : qkv(channels, 3 * channels), proj(channels, channels),
        num_heads(num_heads), channels(channels),
        head_size(channels / num_heads) {
    assert(channels % num_heads == 0 &&
           "channels must be divisible by num_heads");
  }

  FTensor forward(const FTensor& input) const {
    const int B = input.size(0);
    const int T = input.size(1);
    std::vector<int> qkv_shape{{B, T, num_heads, head_size}};
    std::vector<int> reshaped_shape{{B, T, channels}};

    // Compute Q, K, V
    auto qkv_out = qkv.forward(input);
    auto qkv_split = qkv_out.split(channels, 2);
    auto q_reshaped = qkv_split[0].reshape(qkv_shape);
    auto q = q_reshaped.transpose(1, 2);
    auto k_reshaped = qkv_split[1].reshape(qkv_shape);
    auto k = k_reshaped.transpose(1, 2);
    auto v_reshaped = qkv_split[2].reshape(qkv_shape);
    auto v = v_reshaped.transpose(1, 2);

    // Attention computation
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_size));
    auto k_transposed = k.transpose(2, 3);
    auto scores_raw = matmul(q, k_transposed);
    auto scores = scores_raw * scale;
    auto weights = softmax(scores, true, 0);
    auto attention_out = matmul(weights, v);

    // Reshape and project
    auto transposed = attention_out.transpose(1, 2);
    auto reshaped = transposed.reshape(reshaped_shape);
    auto projected = proj.forward(reshaped);
    return projected;
  }
};

class Block {
public:
  LayerNorm ln1;
  CausalSelfAttention attn;
  LayerNorm ln2;
  Linear mlp_fc, mlp_proj;
  inline static int nh = 0, ch = 0;
  Block(int num_heads, int channels)
      : ln1(channels), attn(num_heads, channels), ln2(channels),
        mlp_fc(channels, 4 * channels), mlp_proj(4 * channels, channels) {
    nh = num_heads;
    ch = channels;
  }

  Block()
      : ln1(ch), attn(nh, ch), ln2(ch), mlp_fc(ch, 4 * ch),
        mlp_proj(4 * ch, ch) {}
  FTensor forward(const FTensor& input) const {
    // Attention block with residual connection
    auto ln1_out = ln1.forward(input);
    auto attn_out = attn.forward(ln1_out);
    auto x = input + attn_out;

    // MLP block with residual connection
    auto ln2_out = ln2.forward(x);
    auto mlp_fc_out = mlp_fc.forward(ln2_out);
    auto gelu_out = gelu(mlp_fc_out);
    auto mlp_proj_out = mlp_proj.forward(gelu_out);
    auto result = x + mlp_proj_out;
    return result;
  }
  void operator+=(const Block& other) {
    ln1.weight += other.ln1.weight;
    ln1.bias += other.ln1.bias;
    attn.qkv.weight += other.attn.qkv.weight;
    attn.qkv.bias += other.attn.qkv.bias;
    attn.proj.weight += other.attn.proj.weight;
    attn.proj.bias += other.attn.proj.bias;
    ln2.weight += other.ln2.weight;
    ln2.bias += other.ln2.bias;
    mlp_fc.weight += other.mlp_fc.weight;
    mlp_fc.bias += other.mlp_fc.bias;
    mlp_proj.weight += other.mlp_proj.weight;
    mlp_proj.bias += other.mlp_proj.bias;
  }
};

class Transformer {
public:
  Encoder encoder;
  std::vector<Block> blocks;
  LayerNorm ln_f;

  explicit Transformer(const Config& config)
      : encoder(config.padded_vocab_size, config.max_seq_len, config.channels),
        ln_f(config.channels) {
    blocks.reserve(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i)
      blocks.emplace_back(config.num_heads, config.channels);
  }

  FTensor forward(const ITensor& input, const ITensor& input_pos) const {
    auto x = encoder.forward(input, input_pos);
    for (int i = 0; i < blocks.size(); i++) {
      auto block_out = blocks[i].forward(x);
      x = block_out;
    }
    auto final_norm = ln_f.forward(x);
    return final_norm;
  }
};

class GPT2 {
private:
  static constexpr int MAGIC_NUMBER = 20240326;
  static constexpr int VERSION = 3;
  static constexpr int HEADER_SIZE = 256;

public:
  template <typename Func> void for_each_parameter(Func func) {
    // Embedding parameters
    func(&transformer.encoder.wte);
    func(&transformer.encoder.wpe);

    // Block parameters in checkpoint order
    for (auto& block : transformer.blocks)
      func(&block.ln1.weight);
    for (auto& block : transformer.blocks)
      func(&block.ln1.bias);
    for (auto& block : transformer.blocks)
      func(&block.attn.qkv.weight);
    for (auto& block : transformer.blocks)
      func(&block.attn.qkv.bias);
    for (auto& block : transformer.blocks)
      func(&block.attn.proj.weight);
    for (auto& block : transformer.blocks)
      func(&block.attn.proj.bias);
    for (auto& block : transformer.blocks)
      func(&block.ln2.weight);
    for (auto& block : transformer.blocks)
      func(&block.ln2.bias);
    for (auto& block : transformer.blocks)
      func(&block.mlp_fc.weight);
    for (auto& block : transformer.blocks)
      func(&block.mlp_fc.bias);
    for (auto& block : transformer.blocks)
      func(&block.mlp_proj.weight);
    for (auto& block : transformer.blocks)
      func(&block.mlp_proj.bias);

    // Final layer norm
    func(&transformer.ln_f.weight);
    func(&transformer.ln_f.bias);
  }

private:
  static Config read_config_from_file(FILE* file) {
    int header[HEADER_SIZE];
    if (fread(header, sizeof(int), HEADER_SIZE, file) != HEADER_SIZE)
      throw std::runtime_error("Failed to read checkpoint header");

    if (header[0] != MAGIC_NUMBER)
      throw std::runtime_error("Invalid magic number in checkpoint");
    if (header[1] != VERSION)
      throw std::runtime_error("Unsupported checkpoint version");

    Config config{
        header[2], // max_seq_len
        header[3], // vocab_size
        header[7], // padded_vocab_size
        header[4], // num_layers
        header[5], // num_heads
        header[6]  // channels
    };

    std::cerr << "[GPT-2 Config] seq_len:" << config.max_seq_len
              << " vocab:" << config.vocab_size
              << " layers:" << config.num_layers
              << " heads:" << config.num_heads
              << " channels:" << config.channels << '\n';

    return config;
  }

  void load_weights_from_file(FILE* file) {
    num_parameters = 0;
    for_each_parameter([&](FTensor* tensor) {
      const int elements = tensor->num_elements();
      if (fread(tensor->data(), sizeof(float), static_cast<size_t>(elements),
                file) != static_cast<size_t>(elements))
        throw std::runtime_error("Failed to read tensor data");
      num_parameters += elements;
    });
  }

public:
  Config config;
  Transformer transformer;
  int num_parameters = 0;

  explicit GPT2(const Config& cfg) : config(cfg), transformer(cfg) {}

  explicit GPT2(const std::string& checkpoint_path)
      : config(load_config_from_checkpoint(checkpoint_path)),
        transformer(config) {
    load_weights_from_checkpoint(checkpoint_path);
  }

  ND ITensor get_input_pos(int B, int T) const {
    ITensor input_pos({B, T}); // Create position indices
    for (int b = 0; b < B; ++b)
      for (int t = 0; t < T; ++t)
        input_pos.data()[b * T + t] =
            t; // Fill with sequential positions 0, 1, ..., T-1 for each batch
    return input_pos;
  }

  FTensor forward(const ITensor& input) const {
    const int B = input.size(0);
    const int T = input.size(1);
    ITensor input_pos = get_input_pos(B, T); // Get position indices

    auto hidden = transformer.forward(input, input_pos);
    auto weight_transposed = transformer.encoder.wte.transpose(0, 1);
    auto logits = matmul(hidden, weight_transposed);
    auto probabilities = softmax(logits, false, config.vocab_size);
    return probabilities;
  }

  std::vector<FTensor*> get_parameter_tensors() {
    std::vector<FTensor*> params;
    for_each_parameter([&](FTensor* tensor) { params.push_back(tensor); });
    return params;
  }

  static Config load_config_from_checkpoint(const std::string& path) {
    auto file = std::unique_ptr<FILE, decltype(&fclose)>(
        fopen(path.c_str(), "rb"), fclose);
    if (!file)
      throw std::runtime_error("Could not open checkpoint: " + path);
    return read_config_from_file(file.get());
  }

  void load_weights_from_checkpoint(const std::string& path) {
    auto file = std::unique_ptr<FILE, decltype(&fclose)>(
        fopen(path.c_str(), "rb"), fclose);
    if (!file)
      throw std::runtime_error("Could not open checkpoint: " + path);

    // Skip header
    fseek(file.get(), HEADER_SIZE * sizeof(int), SEEK_SET);
    load_weights_from_file(file.get());

    std::cerr << "Loaded " << num_parameters << " parameters from " << path
              << '\n';
  }

  void load_checkpoint(const std::string& path) {
    auto file = std::unique_ptr<FILE, decltype(&fclose)>(
        fopen(path.c_str(), "rb"), fclose);
    if (!file)
      throw std::runtime_error("Could not open checkpoint: " + path);

    auto file_config = read_config_from_file(file.get());

    // Verify config matches
    if (file_config.max_seq_len != config.max_seq_len ||
        file_config.vocab_size != config.vocab_size ||
        file_config.num_layers != config.num_layers ||
        file_config.num_heads != config.num_heads ||
        file_config.channels != config.channels ||
        file_config.padded_vocab_size != config.padded_vocab_size) {
      throw std::runtime_error("Configuration mismatch with checkpoint");
    }

    load_weights_from_file(file.get());
    std::cerr << "Checkpoint loaded: " << num_parameters << " parameters"
              << '\n';
  }
};

} // namespace gpt2