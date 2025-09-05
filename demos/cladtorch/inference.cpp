#include "llm.hpp"
#include "tokenizer.hpp"
#include <iomanip>
using namespace gpt2;

uint32_t random_u32(uint64_t* state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

// random float32 in [0, 1)
float random_f32(uint64_t* state) {
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float* probs, int n, float coin) {
  // sample index from probs (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probs[i];
    if (coin < cdf)
      return i;
  }
  return n - 1; // in case of rounding errors
}

int main() {
  GPT2 model("gpt2_124M.bin");
  const Config config = model.config;

  int B = 1;
  int T = 64;

  Tokenizer tokenizer("gpt2_tokenizer.bin");

  uint64_t rng_state = 1337;
  const int gen_max_length = 64;
  ITensor gen_tokens({B, T});
  gen_tokens.fill(tokenizer.eot_token()); // Initialize with end-of-text token

  std::cout << "generating:\n---\n";
  std::cerr << std::setprecision(4);
  struct timespec start, end; // timers for benchmarking per token
  // clock_gettime(CLOCK_MONOTONIC, &start);
  for (int t = 1; t < gen_max_length; t++) {
    clock_gettime(CLOCK_MONOTONIC, &start);
    auto probs_t = model.forward(gen_tokens);
    clock_gettime(CLOCK_MONOTONIC, &end);

    float* probs = new float[config.padded_vocab_size];
    for (int v = 0; v < config.padded_vocab_size; v++)
      probs[v] =
          probs_t.at(0, t - 1, v); // Get probabilities for the first batch

    float coin = random_f32(&rng_state);
    int next_token = sample_mult(probs, model.config.vocab_size, coin);
    gen_tokens.at(0, t) = next_token; // Use the first batch for generation
    delete[] probs;
    double time_taken = (end.tv_sec - start.tv_sec) * 1000.0 +
                        (end.tv_nsec - start.tv_nsec) / 1e6;
    std::cerr << "[step " << t << ", " << time_taken << "ms] ";

    tokenizer.safe_print(next_token);
    std::cout << std::flush;
  }
  // clock_gettime(CLOCK_MONOTONIC, &end);
  // double time_taken = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec -
  // start.tv_nsec) / 1e6; std::cerr << "\nTime taken: " << time_taken << " ms
  // for " << gen_max_length - 1
  //           << " tokens, " << (gen_max_length - 1) / (time_taken / 1000.0) <<
  //           " tokens/sec\n";
  std::cout << "\n---\n";
}