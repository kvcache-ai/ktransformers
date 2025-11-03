#include "arm_kml.hpp"

int main() {
  const size_t M = 128, N = 64;
  float16_t* a = new float16_t[M * N];
  float16_t* b = new float16_t[M * N];
  float16_t* c = new float16_t[M * M];
  float16_t* c_check = new float16_t[M * M];
  for (size_t i = 0; i < M * N; i++) {
    a[i] = static_cast<double>(std::rand()) / RAND_MAX / 10.0;
    b[i] = static_cast<double>(std::rand()) / RAND_MAX / 10.0;
  }

  arm_kml::MatRef<float16_t> aref(a, M, N, M, CblasColMajor);
  arm_kml::MatRef<float16_t> bref(b, N, M, M, CblasColMajor);
  arm_kml::MatRef<float16_t> cref(c, M, M, M, CblasColMajor);
  {
    memset(c, 0, M * M * sizeof(float16_t));
    memset(c_check, 0, M * M * sizeof(float16_t));
    arm_kml::mul_mat(aref, bref, cref);
  }
}
