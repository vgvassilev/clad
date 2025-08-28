/*
This file trains the GPT-2 model.
*/
#include "llm_opt.hpp"
#include "tokenizer.hpp"

// sampler
unsigned int random_u32(uint64_t* state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(uint64_t* state) { // random float32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float* probabilities, int n, float coin) {
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf)
      return i;
  }
  return n - 1; // in case of rounding errors
}

float gpt2forw(GPT2* model, const int* inputs, const int* targets) {
  model->forward(inputs, targets);
  return model->mean_loss;
}

// ----------------------------------------------------------------------------
// main training loop
int main() {

  // build the GPT-2 model from a checkpoint
  GPT2 model("gpt2_124M.bin");
  GPT2 d_model("gpt2_124M.bin");
  
  printf("[GPT-2]\n");
  printf("max_seq_len: %d\n", model.config.max_seq_len);
  printf("vocab_size: %d\n", model.config.vocab_size);
  printf("padded_vocab_size: %d\n", model.config.padded_vocab_size);
  printf("num_layers: %d\n", model.config.num_layers);
  printf("num_heads: %d\n", model.config.num_heads);
  printf("channels: %d\n", model.config.channels);
  printf("num_parameters: %zu\n", model.num_parameters);

  // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
  std::string tiny_stories_train =
    "/Users/rohan/Developer/projects/gsoc25/workspace/ml/llm.c/dev/data/tinystories/TinyStories_train.bin";
  std::string tiny_stories_val =
    "/Users/rohan/Developer/projects/gsoc25/workspace/ml/llm.c/dev/data/tinystories/TinyStories_val.bin";
  std::string tiny_shakespeare_train =
    "/Users/rohan/Developer/projects/gsoc25/workspace/ml/llm.c/dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
  std::string tiny_shakespeare_val =
    "/Users/rohan/Developer/projects/gsoc25/workspace/ml/llm.c/dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
  std::string train_tokens = access(tiny_shakespeare_train.c_str(), F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
  std::string val_tokens = access(tiny_shakespeare_val.c_str(), F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
  int B = 4;  // batch size 4 (i.e. 4 independent token sequences will be trained on)
  int T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
  gpt2::DataLoader train_loader(train_tokens, B, T, 0, 1, true);
  gpt2::DataLoader val_loader(val_tokens, B, T, 0, 1, false);
  printf("train dataset num_batches: %zu\n", train_loader.num_tokens() / (B * T));
  printf("val dataset num_batches: %zu\n", val_loader.num_tokens() / (B * T));
  int val_num_batches = 5;

  auto grad = clad::gradient(gpt2forw, "0");
  grad.dump();
  model.allocate(B, T);
  d_model.allocate(B, T);
  d_model.zero_all();
  // grad.execute(model, ...)

  // build the Tokenizer
  gpt2::Tokenizer tokenizer("gpt2_tokenizer.bin");
  // tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

  // some memory for generating samples from the model
  uint64_t rng_state = 1337;
  std::vector<int> gen_tokens(B*T);
  const int genT = 64; // number of steps of inference we will do

  // train
  struct timespec start, end;
  for (int step = 0; step <= 40; step++) {
    // once in a while estimate the validation loss
    if (step % 10 == 0) {
      float val_loss = 0.0f;
      val_loader.reset();
      for (int i = 0; i < val_num_batches; i++) {
        val_loader.next_batch();
        model.forward(val_loader.inputs(), val_loader.targets());
        val_loss += model.mean_loss;
      }
      val_loss /= val_num_batches;
      printf("val loss %f\n", val_loss);
    }

    // once in a while do model inference to print generated text
    if (step > 0 && step % 20 == 0) {
      // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
      gen_tokens.assign(B*T, tokenizer.eot_token());
      // now sample from the model autoregressively
      printf("generating:\n---\n");
      for (int t = 1; t < genT; t++) {
        // note that inference is very wasteful here because for each token
        // we re-calculate the forward pass for all of (B,T) positions from scratch
        // but the inference here is just for sanity checking anyway
        // and we can maybe optimize a bit more later, with careful tests
        model.forward(gen_tokens.data(), nullptr);
        // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
        // we're in principle running B "inference streams" in parallel here
        // but only using position 0
        // get the Vp-dimensional vector probs[0, t-1, :]
        float* probs = model.acts.probs + (t - 1) * model.config.padded_vocab_size;
        float coin = random_f32(&rng_state);
        // note we're only sampling from the first V elements, ignoring padding
        // (the probabilities in the padded region should be zero anyway)
        int next_token = sample_mult(probs, model.config.vocab_size, coin);
        gen_tokens[t] = next_token;
        // print the generated token, either using the Tokenizer or a fallback
        tokenizer.safe_print(next_token);
        fflush(stdout);
      }
      printf("\n---\n");
    }

    // do a training step
    clock_gettime(CLOCK_MONOTONIC, &start);
    train_loader.next_batch();
    d_model.zero_all();
    grad.execute(&model, train_loader.inputs(), train_loader.targets(), &d_model);
    model.update(&d_model, 1e-3f);
    // gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
  }
}
