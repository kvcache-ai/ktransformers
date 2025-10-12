#ifndef CPUINFER_ROPE_HPP
#define CPUINFER_ROPE_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

template <typename T, typename E, typename A>
concept ROPE_APPLIER = requires(T t, E* emb, int size, int pos_start, int pos_len, A* v) {
  // must be thread safe and efficient

  // apply embeddings with pos_start to v, v is vector of size
  { T::apply_single(emb, v, size, pos_start) } -> std::same_as<void>;

  // for every v i, apply embeddings with pos_start + i to v[i], v is vector of size
  { T::apply_multiple(emb, v, size, pos_start, pos_len) } -> std::same_as<void>;
};

template <typename T, typename A>
concept ROPE_ANGLE = requires(T t, size_t at) {
  { t.cos(at) } -> std::same_as<float*>;
  { t.sin(at) } -> std::same_as<float*>;
  { t.init(at) } -> std::same_as<void>;
};

template <typename E, typename A>
  requires ROPE_ANGLE<E, A>
struct Rope {
 public:
  static void apply_single(E& emb, A* v, int size, int pos_start) {
    if (size == 0) {
      return;
    }
    if (size % 2 != 0) {
      throw std::invalid_argument("Rope::apply_single: 'size' (head_dim) must be even for LLaMA-style RoPE.");
    }

    const float* cos = emb.cos(pos_start);
    const float* sin = emb.sin(pos_start);

    thread_local static std::vector<float> v2;
    if (v2.size() < size) {
      v2.resize(size);
    }

    for (int i = 0; i < size / 2; i++) {
      float a = v[2 * i], b = v[2 * i + 1];
      v2[i] = cos[i] * a - sin[i] * b;
      v2[i + size / 2] = sin[i] * a + cos[i] * b;
    }

    for (int i = 0; i < size; i++) {
      v[i] = v2[i];
    }
  }

  static void apply_multiple(E& emb, A* v_block_start, int size_per_vector, int pos_start, int pos_len) {
    if (size_per_vector == 0 || pos_len == 0) {
      return;
    }
    if (size_per_vector % 2 != 0) {
      throw std::invalid_argument("Rope::apply_multiple: 'size_per_vector' (head_dim) must be even.");
    }

    for (int i = 0; i < pos_len; ++i) {
      apply_single(emb, v_block_start + size_per_vector * i, size_per_vector, pos_start + i);
    }
  }
};

class RotaryEmbeddingBase {
 public:
  virtual ~RotaryEmbeddingBase() = default;
  virtual void init(size_t seq_len) {
    calculate_inv_freq();
    set_cos_sin_cache(seq_len);
    this->max_seq_len_cached_ = seq_len;
  }

 protected:
  RotaryEmbeddingBase(size_t dim, size_t max_pos_embeddings, double base_val)
      : dim_(dim), max_position_embeddings_(max_pos_embeddings), base_(base_val), max_seq_len_cached_(0) {}

  virtual void calculate_inv_freq() = 0;
  virtual void set_cos_sin_cache(size_t seq_len) = 0;

  size_t dim_;
  size_t max_position_embeddings_;
  double base_;
  std::vector<double> inv_freq_;
  size_t max_seq_len_cached_;
};

class DeepseekV3RotaryEmbedding : public RotaryEmbeddingBase {
 public:
  DeepseekV3RotaryEmbedding(size_t dim, size_t max_position_embeddings = 2048, double base = 10000.0f)
      : RotaryEmbeddingBase(dim, max_position_embeddings, base) {
    if (this->dim_ % 2 != 0 || this->dim_ < 0) {
      throw std::invalid_argument("Dimension must be even for RotaryEmbedding and >= 0.");
    }

    if (this->max_position_embeddings_ < 0) {
      throw std::invalid_argument("DeepseekV3RotaryEmbedding max_position_embeddings_ must be >= 0.");
    }

    calculate_inv_freq();
    set_cos_sin_cache(this->max_position_embeddings_);
  }

  float* sin(size_t at) { return sin_cached_.data() + at * this->dim_ / 2; }
  float* cos(size_t at) { return cos_cached_.data() + at * this->dim_ / 2; }

 protected:
  void calculate_inv_freq() override {
    this->inv_freq_.resize(this->dim_ / 2);
    for (size_t i = 0; i < this->dim_ / 2; ++i) {
      this->inv_freq_[i] = 1.0 / std::pow(this->base_, 2.0 * i / this->dim_);
    }
  }

  void set_cos_sin_cache(size_t seq_len) override {
    if (this->inv_freq_.empty()) {
      calculate_inv_freq();
    }

    cos_cached_.resize(seq_len * this->dim_ / 2);
    sin_cached_.resize(seq_len * this->dim_ / 2);

    for (size_t i = 0; i < seq_len; ++i) {
      for (size_t j = 0; j < this->inv_freq_.size(); ++j) {
        double freq = static_cast<double>(i) * this->inv_freq_[j];
        double cos_val = std::cos(freq);
        double sin_val = std::sin(freq);
        size_t idx1 = i * this->dim_ / 2 + j;

        cos_cached_.at(idx1) = cos_val;
        sin_cached_.at(idx1) = sin_val;
      }
    }
    this->max_seq_len_cached_ = seq_len;
  }

  std::vector<float> cos_cached_;
  std::vector<float> sin_cached_;
};

inline double yarn_find_correction_dim(double num_rotations, double dim, double base, double max_position_embeddings) {
  return (dim * std::log(max_position_embeddings / (num_rotations * static_cast<double>(2.0f) * M_PI))) /
         (static_cast<double>(2.0f) * std::log(base));
}

inline std::pair<size_t, size_t> yarn_find_correction_range(double low_rot, double high_rot, size_t dim,
                                                            double base = 10000,
                                                            double max_position_embeddings = 2048) {
  double low_f = std::floor(yarn_find_correction_dim(low_rot, static_cast<double>(dim), base, max_position_embeddings));
  double high_f =
      std::ceil(yarn_find_correction_dim(high_rot, static_cast<double>(dim), base, max_position_embeddings));

  size_t low = static_cast<size_t>(std::max(0.0, low_f));
  size_t high = static_cast<size_t>(std::min(static_cast<double>(dim - 1), high_f));
  return std::pair{low, high};
}

inline std::vector<double> yarn_linear_ramp_mask(double min_val, double max_val, size_t dim) {
  if (std::abs(min_val - max_val) < 1e-6f) {
    max_val += 0.001;
  }
  std::vector<double> ramp_func(dim);
  for (size_t i = 0; i < dim; ++i) {
    double linear_func = (static_cast<double>(i) - min_val) / (max_val - min_val);
    ramp_func[i] = std::clamp(linear_func, 0.0, 1.0);
  }
  return ramp_func;
}

inline double yarn_get_mscale(double scale = 1.0, double mscale = 1.0) {
  if (scale <= 1.0) {
    return 1.0;
  }
  return 0.1 * mscale * std::log(scale) + 1.0;
}

class DeepseekV3YarnRotaryEmbedding : public DeepseekV3RotaryEmbedding {
 public:
  DeepseekV3YarnRotaryEmbedding(size_t dim, size_t max_position_embeddings = 2048, double base = 10000.0f,
                                double scaling_factor = 1.0, size_t original_max_position_embeddings = 4096,
                                double beta_fast = 32.0, double beta_slow = 1.0, double mscale_val = 1.0,
                                double mscale_all_dim_val = 0.0)
      : DeepseekV3RotaryEmbedding(dim, 0, base),
        scaling_factor_(scaling_factor),
        original_max_position_embeddings_(original_max_position_embeddings),
        beta_fast_(beta_fast),
        beta_slow_(beta_slow),
        mscale_(mscale_val),
        mscale_all_dim_(mscale_all_dim_val) {
    if (this->dim_ % 2 != 0 || this->dim_ < 0) {
      throw std::invalid_argument("Dimension must be even for RotaryEmbedding and >= 0.");
    }

    if (this->max_position_embeddings_ < 0) {
      throw std::invalid_argument("DeepseekV3YarnRotaryEmbedding: max_position_embeddings_ must be >= 0.");
    }
    calculate_inv_freq();
    set_cos_sin_cache(max_position_embeddings);
  }

 protected:
  void calculate_inv_freq() override {
    if (this->dim_ == 0) {
      this->inv_freq_.clear();
      return;
    }
    size_t dim_half = this->dim_ / 2;
    this->inv_freq_.resize(dim_half);

    std::vector<double> freq_extra(dim_half);
    std::vector<double> freq_inter(dim_half);
    for (size_t i = 0; i < dim_half; ++i) {
      double freq_index = 2.0 * i / this->dim_;
      freq_extra[i] = 1.0 / std::pow(this->base_, freq_index);
      freq_inter[i] = 1.0f / (scaling_factor_ * std::pow(this->base_, freq_index));
    }

    auto [low_idx_f, high_idx_f] =
        yarn_find_correction_range(beta_fast_, beta_slow_, this->dim_, this->base_, original_max_position_embeddings_);

    size_t low_idx = static_cast<size_t>(low_idx_f);
    size_t high_idx = static_cast<size_t>(high_idx_f);

    std::vector<double> inv_freq_mask_ramp;
    inv_freq_mask_ramp = yarn_linear_ramp_mask(low_idx, high_idx, dim_half);

    for (size_t i = 0; i < dim_half; ++i) {
      double mask_val = 1.0 - inv_freq_mask_ramp[i];
      this->inv_freq_[i] = freq_inter[i] * (1.0 - mask_val) + freq_extra[i] * mask_val;
    }
  }

  void set_cos_sin_cache(size_t seq_len) override {
    if (this->inv_freq_.empty() || this->inv_freq_.size() != this->dim_ / 2) {
      calculate_inv_freq();
    }

    this->cos_cached_.resize(seq_len * this->dim_ / 2);
    this->sin_cached_.resize(seq_len * this->dim_ / 2);

    // printf("scaling_factor %f, mscale %f, mscale all dim %f\n", scaling_factor_, mscale_, mscale_all_dim_);
    double scale_factor_val = yarn_get_mscale(scaling_factor_, mscale_);
    double scale_all_dim_factor_val = yarn_get_mscale(scaling_factor_, mscale_all_dim_);
    double actual_mscale = 1.0;
    if (std::abs(scale_all_dim_factor_val) > 1e-6f) {
      actual_mscale = scale_factor_val / scale_all_dim_factor_val;
    }
    // printf("actual_mscale: %f, %f, %f\n", actual_mscale, scale_factor_val, scale_all_dim_factor_val);

    for (size_t i = 0; i < seq_len; ++i) {
      for (size_t j = 0; j < this->inv_freq_.size(); ++j) {
        double freq = static_cast<double>(i) * this->inv_freq_[j];
        double cos_val = std::cos(freq) * actual_mscale;
        double sin_val = std::sin(freq) * actual_mscale;
        size_t idx1 = i * this->dim_ / 2 + j;

        this->cos_cached_.at(idx1) = cos_val;
        this->sin_cached_.at(idx1) = sin_val;
      }
    }
    this->max_seq_len_cached_ = seq_len;
  }

 private:
  double scaling_factor_;
  size_t original_max_position_embeddings_;
  double beta_fast_;
  double beta_slow_;
  double mscale_;
  double mscale_all_dim_;
};

#endif