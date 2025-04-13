#include <tbb/concurrent_hash_map.h>
#include <iostream>

int main() {
  tbb::concurrent_hash_map<int, int> map;
  map.insert({1, 2});
  decltype(map)::accessor a;
  std::cout << map.find(a, 1) << std::endl;

  return 0;
}
