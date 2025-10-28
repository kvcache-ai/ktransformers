#ifndef CPUINFER_REDUCE_HPP
#define CPUINFER_REDUCE_HPP

#include <cmath>

template <typename T>
void reduce_sum(T** data, size_t data_groups_count, size_t begin, size_t end) {
  if (data_groups_count <= 1) {
  } else if (data_groups_count == 2) {
    for (size_t i = begin; i < end; i++) {
      data[0][i] += data[1][i];
    }
  } else {
    int part1 = data_groups_count / 2;
    reduce_sum(data, part1, begin, end);
    reduce_sum(data + part1, data_groups_count - part1, begin, end);
    for (size_t i = begin; i < end; i++) {
      data[0][i] += data[part1][i];
    }
  }
}

#endif