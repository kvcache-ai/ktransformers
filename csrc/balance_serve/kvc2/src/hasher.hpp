#ifndef __HASHER_HPP_
#define __HASHER_HPP_

#include "defs.h"
#include "xxhash.h"

namespace kvc2 {

const uint64_t hash_seed = 4123512;
const uint64_t check_hash_seed = 1025753;

using TokensHash = XXH64_hash_t;
struct TokensHasher {
  XXH64_state_t* state;
  TokensHasher() {
    state = XXH64_createState();
    reset();
  }
  ~TokensHasher() { XXH64_freeState(state); }

  TokensHasher(TokensHasher& other) = delete;
  TokensHasher& operator=(TokensHasher& other) = delete;
  TokensHasher(TokensHasher&& other) = delete;
  TokensHasher& operator=(TokensHasher&& other) = delete;
  TokensHash get() { return XXH64_digest(state); }
  void reset(size_t seed = hash_seed) { XXH64_reset(state, seed); }
  TokensHash update(Token* data, TokenLength length) {
    XXH64_update(state, data, length * sizeof(Token));
    return get();
  }

  TokensHash update_raw(void* data, size_t size) {
    XXH64_update(state, data, size);
    return get();
  }

  static TokensHash hash(Token* data, TokenLength length) { return XXH64(data, length * sizeof(Token), hash_seed); }
};
}  // namespace kvc2
#endif