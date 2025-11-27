#include "dataloader.hpp"
#include "llm.hpp"
#include "tokenizer.hpp"
#include <clad/Differentiator/CladtorchBuiltins.h>
#include <clad/Differentiator/Differentiator.h>
#include <clad/Differentiator/STLBuiltins.h>

using namespace gpt2;

static uint32_t random_u32(uint64_t* state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1DULL) >> 32;
}

// random float32 in [0, 1)
static float random_f32(uint64_t* state) {
  return static_cast<float>(random_u32(state) >> 8) / 16777216.0F;
}

static int sample_mult(const float* probs, int n, float coin) {
  // sample index from probs (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  float cdf = 0.0F;
  for (int i = 0; i < n; i++) {
    cdf += probs[i]; // NOLINT
    if (coin < cdf)
      return i;
  }
  return n - 1; // in case of rounding errors
}

static float gpt2_loss(const GPT2& model, const ITensor& input,
                       const ITensor& targets) {
  auto probs = model.forward(input);
  auto loss = cross_entropy_loss(probs, targets);
  return loss.scalar();
}

int main() {
  GPT2 model("gpt2_124M.bin");
  const Config config = model.config;

  int B = 4;
  int T = 64;

  Tokenizer tokenizer("gpt2_tokenizer.bin");

  const std::string tiny_shakespeare_train =
      "data/tinyshakespeare/tiny_shakespeare_train.bin";
  const std::string tiny_shakespeare_val =
      "data/tinyshakespeare/tiny_shakespeare_val.bin";
  const std::string& train_token = tiny_shakespeare_train;
  const std::string& val_token = tiny_shakespeare_val;
  DataLoader train_loader(train_token, B, T, /*process_rank=*/0,
                          /*num_processes=*/1, /*should_shuffle=*/true);
  DataLoader val_loader(val_token, B, T, /*process_rank=*/0,
                        /*num_processes=*/1, /*should_shuffle=*/false);

  std::cout << "train dataset num_batches: "
            << train_loader.num_tokens() / (B * T) << '\n';
  std::cout << "val dataset num_batches: " << val_loader.num_tokens() / (B * T)
            << '\n';
  int val_num_batches = 5;

  uint64_t rng_state = 1337;
  const int gen_max_length = 64;

  // Initialize with end-of-text token
  ITensor gen_tokens({1, T}, tokenizer.eot_token());

  GPT2 d_model(config);

  auto grad = clad::gradient(gpt2_loss, "0");
  grad.dump(); // Dump the gradient function for debugging

  struct timespec start {};
  struct timespec end {};
  for (int step = 0; step <= 40; step++) {
    // once in a while, estimate the validation loss
    if (step % 10 == 0) {
      float val_loss = 0.0F;
      // Reset the validation loader to start from the beginning
      val_loader.reset();
      for (int i = 0; i < val_num_batches; i++) {
        val_loader.next_batch();
        const ITensor input = ITensor({B, T}, val_loader.inputs());
        const ITensor targets = ITensor({B, T}, val_loader.targets());
        val_loss += gpt2_loss(model, input, targets);
      }
      val_loss /= (float)val_num_batches;
      std::cout << "val loss: " << val_loss << '\n';
    }

    // once in a while do model inference to print generated text
    if (step > 0 && step % 20 == 0) {
      gen_tokens.fill(tokenizer.eot_token());

      std::cout << "generating:\n---\n";
      for (int t = 1; t < gen_max_length; t++) {
        auto probs_t = model.forward(gen_tokens);
        float* probs = new float[config.padded_vocab_size]; // NOLINT
        // Get probabilities for the first batch
        for (int v = 0; v < config.padded_vocab_size; v++)
          probs[v] = probs_t.at(0, t - 1, v); // NOLINT

        float coin = random_f32(&rng_state);
        int next_token = sample_mult(probs, model.config.vocab_size, coin);
        gen_tokens.at(0, t) = next_token; // Use the first batch for generation
        delete[] probs;                   // NOLINT

        tokenizer.safe_print(next_token);
        std::cout << std::flush;
      }
      std::cout << "\n---\n";
    }

    // Perform a training step
    train_loader.next_batch();
    const ITensor input = ITensor({B, T}, train_loader.inputs());
    const ITensor targets = ITensor({B, T}, train_loader.targets());

    clock_gettime(CLOCK_MONOTONIC, &start);
    auto mean_loss = gpt2_loss(model, input, targets);
    d_model.for_each_parameter(
        [&](FTensor* t) { t->fill(0); }); // Zero out gradients
    grad.execute(model, input, targets, &d_model);
    std::vector<FTensor*> params = model.get_parameter_tensors();
    std::vector<FTensor*> grads = d_model.get_parameter_tensors();
    for (size_t i = 0; i < params.size(); ++i) {
      // Update parameters with a learning rate of 1e-4
      *params[i] += (*grads[i]) * -1e-4F;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_elapsed_s = (double)(end.tv_sec - start.tv_sec) +
                            ((double)(end.tv_nsec - start.tv_nsec) / 1e9);
    std::cout << "step " << step << " train Loss: " << mean_loss << " (took "
              << time_elapsed_s * 1000 << " ms)" << '\n';
  }
}