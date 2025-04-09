#include "xxhash.h"
#include <iostream>

int main() {
  std::string t = "hello world";
  XXH64_hash_t hash = XXH64(t.data(), t.size(), 123);
  std::cout << hash << std::endl;
  {
    /* create a hash state */
    XXH64_state_t* const state = XXH64_createState();
    if (state == NULL)
      abort();

    if (XXH64_reset(state, 123) == XXH_ERROR)
      abort();

    if (XXH64_update(state, t.data(), 5) == XXH_ERROR)
      abort();

    if (XXH64_update(state, t.data() + 5, t.size() - 5) == XXH_ERROR)
      abort();
    /* Produce the final hash value */
    XXH64_hash_t const hash = XXH64_digest(state);

    /* State could be re-used; but in this example, it is simply freed  */
    XXH64_freeState(state);
    std::cout << hash << std::endl;
  }

  return 0;
}
