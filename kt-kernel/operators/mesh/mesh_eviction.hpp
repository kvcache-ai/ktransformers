/**
 * @file mesh_eviction.hpp
 * @brief 驱逐评分：Heat EMA + Markov 层间转移矩阵
 *
 * 驱逐分数 = policy_rank + lookahead_weight * max(heat, markov_prior)
 *
 * Heat EMA：
 * - 跨 token 全局衰减：β=0.5，几何衰减
 * - 单 token 结束后一次性批量更新（GPU→CPU 一次传完，不逐层传）
 * - 满驻留自动跳过：slot 容量 ≥ CPU 专家数时禁用 Heat 更新
 *
 * Markov 转移矩阵：
 * - T[L][s][t] = P(第L+1层激活专家t | 第L层激活专家s)
 * - 稀疏存储：每行只保留概率最高的 K=16 个目标
 * - 只用于驱逐评分，绝对不用于预取
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace mesh {

// ===== Heat EMA 跟踪器 =====

/**
 * @brief 跨 token 全局 Heat EMA
 *
 * 每个 token 结束后，将该 token 所有层的 gating 分数一次性批量传入更新。
 * β=0.5 几何衰减：最近 token 占 0.5，前一个 0.25，再前一个 0.125...
 *
 * 关键设计：所有层的 gating 分数暂存 GPU，单 token 计算结束后一次性批量传到 CPU。
 * 不做逐层 GPU→CPU 传输。
 */
class HeatTracker {
 public:
  HeatTracker(int expert_num, float beta = 0.5f, float gamma = 0.7f)
      : expert_num_(expert_num), beta_(beta), gamma_(gamma) {
    global_heat_.assign(expert_num, 0.0f);
    token_layer_heat_.assign(expert_num, 0.0f);
  }

  // 单 token 结束后批量更新
  // all_layers_gating: [layer][expert_num]，每层的 gating 分数（全长 vector）
  void commit_token(const std::vector<std::vector<float>>& all_layers_gating) {
    if (all_layers_gating.empty()) return;

    // 层内 EMA：gamma * heat_old + (1-gamma) * score
    // 这里取最后一层作为该 token 的代表 heat（也可取平均）
    const auto& last_layer = all_layers_gating.back();
    for (int e = 0; e < expert_num_ && e < (int)last_layer.size(); e++) {
      token_layer_heat_[e] = gamma_ * token_layer_heat_[e] + (1.0f - gamma_) * last_layer[e];
    }

    // 跨 token 全局衰减：heat_new = beta * heat_old + (1-beta) * token_heat
    for (int e = 0; e < expert_num_; e++) {
      global_heat_[e] = beta_ * global_heat_[e] + (1.0f - beta_) * token_layer_heat_[e];
    }
  }

  // 获取某专家的全局 heat
  float heat(int expert_id) const {
    if (expert_id < 0 || expert_id >= expert_num_) return 0.0f;
    return global_heat_[expert_id];
  }

  // 满驻留自动跳过检查
  // slot_cap >= cpu_expert_num 时，驱逐不可能发生，Heat 无价值
  bool should_skip(int slot_cap, int cpu_expert_num) const {
    return slot_cap >= cpu_expert_num;
  }

  int expert_num() const { return expert_num_; }
  const std::vector<float>& global_heat() const { return global_heat_; }

 private:
  int expert_num_;
  float beta_;   // 跨 token 衰减率
  float gamma_;  // 层内 EMA 衰减率
  std::vector<float> global_heat_;       // [expert_num] 跨 token 全局 EMA
  std::vector<float> token_layer_heat_;  // [expert_num] 当前 token 层内 EMA
};

// ===== Markov 层间转移矩阵 =====

/**
 * @brief Markov 层间转移矩阵（稀疏存储）
 *
 * T[L][s][t] = P(第L+1层激活专家t | 第L层激活专家s)
 * 每行只保留概率最高的 K 个目标专家，其余丢弃。
 *
 * 更新算法（每 token 每相邻层对）：
 * 1. 衰减旧行：row[t] *= (1-α)
 * 2. 注入新观测：row[t] += α * w[t]
 * 3. 重新归一化
 *
 * 预测算法：
 * P_predicted[t] = Σ_{s∈top-k} P_L[s] × T[L][s][t]
 * 实际计算量：k × K = 8 × 16 = 128 次乘加
 */
class MarkovTracker {
 public:
  static constexpr int kSparseK = 16;  // 每行稀疏保留的目标专家数

  struct Entry {
    int target_id = -1;
    float probability = 0.0f;
  };

  struct SparseRow {
    Entry entries[kSparseK];  // 按 probability 降序排列
    float row_sum = 1.0f;     // 归一化分母（≈1.0）
  };

  MarkovTracker(int num_layers, int expert_num, float alpha = 0.5f)
      : num_layers_(num_layers), expert_num_(expert_num), alpha_(alpha) {
    // 每对相邻层一个矩阵，共 num_layers-1 个
    matrices_.resize(std::max(0, num_layers - 1));
    for (auto& mat : matrices_) {
      mat.resize(expert_num);
      // 初始化为均匀分布：T[L][s][t] = 1/N
      for (auto& row : mat) {
        for (int i = 0; i < kSparseK && i < expert_num; i++) {
          row.entries[i].target_id = i;
          row.entries[i].probability = 1.0f / expert_num;
        }
        row.row_sum = 1.0f;
      }
    }
  }

  /**
   * @brief 更新某对相邻层的转移矩阵
   *
   * @param layer_idx 第 L 层索引（更新的是 T[L]）
   * @param src_topk 第 L 层的 top-k 专家集合
   * @param src_scores 第 L 层各专家的归一化权重（长度 = expert_num，非 top-k 位置为 0）
   * @param dst_topk 第 L+1 层的 top-k 专家集合
   * @param dst_scores 第 L+1 层各专家的归一化权重
   */
  void update(int layer_idx, const std::vector<int>& src_topk,
              const std::vector<float>& src_scores,
              const std::vector<int>& dst_topk,
              const std::vector<float>& dst_scores) {
    if (layer_idx < 0 || layer_idx >= (int)matrices_.size()) return;
    auto& mat = matrices_[layer_idx];

    // 只更新 src_topk 中的源专家对应的行
    for (int s : src_topk) {
      if (s < 0 || s >= expert_num_) continue;
      SparseRow& row = mat[s];

      // 1. 衰减旧行
      for (int i = 0; i < kSparseK; i++) {
        row.entries[i].probability *= (1.0f - alpha_);
      }

      // 2. 注入新观测：对 dst_topk 中的每个目标 t，row[t] += alpha * w[t]
      for (int t : dst_topk) {
        if (t < 0 || t >= expert_num_) continue;
        float w = (t < (int)dst_scores.size()) ? dst_scores[t] : 0.0f;

        // 查找是否已存在
        int slot = -1;
        for (int i = 0; i < kSparseK; i++) {
          if (row.entries[i].target_id == t) {
            slot = i;
            row.entries[i].probability += alpha_ * w;
            break;
          }
        }
        if (slot < 0) {
          // 找一个空位或概率最低的位置替换
          int min_idx = 0;
          float min_prob = row.entries[0].probability;
          for (int i = 1; i < kSparseK; i++) {
            if (row.entries[i].probability < min_prob) {
              min_prob = row.entries[i].probability;
              min_idx = i;
            }
          }
          float new_prob = alpha_ * w;
          if (new_prob > min_prob) {
            row.entries[min_idx].target_id = t;
            row.entries[min_idx].probability = new_prob;
          }
        }
      }

      // 3. 重新归一化
      float sum = 0.0f;
      for (int i = 0; i < kSparseK; i++) {
        sum += row.entries[i].probability;
      }
      row.row_sum = sum;
      if (sum > 1e-6f) {
        for (int i = 0; i < kSparseK; i++) {
          row.entries[i].probability /= sum;
        }
      }

      // 按 probability 降序排列
      std::sort(row.entries, row.entries + kSparseK,
                [](const Entry& a, const Entry& b) {
                  return a.probability > b.probability;
                });
    }
  }

  /**
   * @brief 预测下一层的专家分布
   *
   * @param layer_idx 当前层 L
   * @param topk 当前层的 top-k 专家集合
   * @param scores 当前层各专家的归一化权重
   * @param prior_out 输出：第 L+1 层的 cross_layer_prior
   */
  void predict(int layer_idx, const std::vector<int>& topk,
               const std::vector<float>& scores,
               std::vector<float>& prior_out) const {
    prior_out.assign(expert_num_, 0.0f);
    if (layer_idx < 0 || layer_idx >= (int)matrices_.size()) return;
    const auto& mat = matrices_[layer_idx];

    // P_predicted[t] = Σ_{s∈top-k} P_L[s] × T[L][s][t]
    for (int s : topk) {
      if (s < 0 || s >= expert_num_) continue;
      float p_s = (s < (int)scores.size()) ? scores[s] : 0.0f;
      if (p_s < 1e-6f) continue;

      const SparseRow& row = mat[s];
      for (int i = 0; i < kSparseK; i++) {
        int t = row.entries[i].target_id;
        if (t < 0) continue;
        prior_out[t] += p_s * row.entries[i].probability;
      }
    }
  }

  int num_layers() const { return num_layers_; }
  int expert_num() const { return expert_num_; }

 private:
  int num_layers_;
  int expert_num_;
  float alpha_;
  // matrices_[L] 是第 L 层到第 L+1 层的转移矩阵
  std::vector<std::vector<SparseRow>> matrices_;  // [num_layers-1][expert_num]
};

// ===== 驱逐评分器 =====

/**
 * @brief 驱逐评分器
 *
 * score = policy_rank + lookahead_weight * max(heat, cross_layer_prior)
 *
 * Markov 严格限制：只用于驱逐评分，绝对不用于预取。
 *
 * B2 fix: cross_layer_prior 按 [layer][expert] 累积，predict_next_layer 在每层结束时
 * 调用一次，把 Markov 预测结果 EMA 进 cross_layer_prior_[layer+1]。
 * score() 直接读 cross_layer_prior_[layer]，不再每次 predict。
 */
class EvictionScorer {
 public:
  EvictionScorer(int num_layers, int expert_num, const MeshConfig& config)
      : heat_(expert_num, config.heat_beta, config.heat_gamma),
        markov_(num_layers, expert_num, config.markov_alpha),
        lookahead_weight_(config.lookahead_weight),
        markov_alpha_(config.markov_alpha),
        num_layers_(num_layers),
        expert_num_(expert_num) {
    cross_layer_prior_.assign(num_layers, std::vector<float>(expert_num, 0.0f));
  }

  // 单 token 结束后批量更新 Heat 和 Markov
  // all_layers_topk: [layer] -> top-k expert ids
  // all_layers_scores: [layer] -> normalized router scores (全长 expert_num)
  void commit_token(const std::vector<std::vector<int>>& all_layers_topk,
                    const std::vector<std::vector<float>>& all_layers_scores) {
    // 更新 Heat
    heat_.commit_token(all_layers_scores);

    // 更新 Markov：每对相邻层
    int n = std::min(all_layers_topk.size(), all_layers_scores.size());
    for (int l = 0; l < n - 1; l++) {
      markov_.update(l, all_layers_topk[l], all_layers_scores[l],
                     all_layers_topk[l + 1], all_layers_scores[l + 1]);
    }
  }

  /**
   * @brief B2: 在第 layer_idx 层结束时，预测第 layer_idx+1 层的 cross_layer_prior
   *
   * cross_layer_prior_[layer+1][t] = α * P_predicted[t] + (1-α) * cross_layer_prior_[layer+1][t]
   *
   * @param layer_idx 当前层 L（预测 L+1）
   * @param topk 当前层的 top-k 专家集合
   * @param scores 当前层各专家的归一化权重（全长 expert_num）
   */
  void predict_next_layer(int layer_idx, const std::vector<int>& topk,
                          const std::vector<float>& scores) {
    if (layer_idx < 0 || layer_idx + 1 >= num_layers_) return;
    std::vector<float> predicted;
    markov_.predict(layer_idx, topk, scores, predicted);
    auto& prior = cross_layer_prior_[layer_idx + 1];
    for (int t = 0; t < expert_num_ && t < (int)predicted.size(); t++) {
      prior[t] = markov_alpha_ * predicted[t] + (1.0f - markov_alpha_) * prior[t];
    }
  }

  // 计算某专家的驱逐评分（分数越低越该被驱逐）
  // B2 fix: 直接读 cross_layer_prior_[layer_idx]，不再每次 predict
  float score(int expert_id, int layer_idx) const {
    float h = heat_.heat(expert_id);
    float m = 0.0f;
    if (layer_idx >= 0 && layer_idx < num_layers_ &&
        expert_id >= 0 && expert_id < expert_num_) {
      m = cross_layer_prior_[layer_idx][expert_id];
    }
    float heat = std::max(h, m);
    return lookahead_weight_ * heat;
  }

  // 选择 victim：从候选专家中选分数最低的
  // candidates: 当前 CACHED 且 active_readers==0 的专家列表
  int select_victim(const std::vector<int>& candidates, int layer_idx) const {
    if (candidates.empty()) return -1;
    int victim = candidates[0];
    float min_score = score(victim, layer_idx);
    for (size_t i = 1; i < candidates.size(); i++) {
      float s = score(candidates[i], layer_idx);
      if (s < min_score) {
        min_score = s;
        victim = candidates[i];
      }
    }
    return victim;
  }

  // 满驻留检查
  bool should_skip_layer(int slot_cap, int cpu_expert_num) const {
    return heat_.should_skip(slot_cap, cpu_expert_num);
  }

  HeatTracker& heat() { return heat_; }
  MarkovTracker& markov() { return markov_; }

 private:
  HeatTracker heat_;
  MarkovTracker markov_;
  float lookahead_weight_;
  float markov_alpha_;
  int num_layers_;
  int expert_num_;
  // B2: [layer][expert] 的 Markov 先验，由 predict_next_layer 累积
  std::vector<std::vector<float>> cross_layer_prior_;
};

}  // namespace mesh
