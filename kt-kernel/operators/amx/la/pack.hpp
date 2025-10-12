#ifndef PACK_HPP
#define PACK_HPP

#pragma once
#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class Packed2DLayout {
 public:
  using index_t = std::size_t;

  struct Dim {
    index_t size;  // > 0
    char dir;      // 'r' or 'c'
  };

  // 构造：dims 必须按从低维到高维给出
  explicit Packed2DLayout(std::vector<Dim> dims) : dims_(std::move(dims)) {
    if (dims_.empty()) throw std::invalid_argument("dims must not be empty");
    rows_ = 1;
    cols_ = 1;

    // 预计算行/列 stride（混合进位权重）
    r_stride_for_dim_.assign(dims_.size(), 0);
    c_stride_for_dim_.assign(dims_.size(), 0);

    index_t r_stride = 1, c_stride = 1;
    for (index_t i = 0; i < dims_.size(); ++i) {
      const auto& d = dims_[i];
      if (d.size == 0) throw std::invalid_argument("dim size must be > 0");
      if (d.dir == 'r') {
        r_stride_for_dim_[i] = r_stride;
        r_stride *= d.size;
        rows_ *= d.size;
      } else if (d.dir == 'c') {
        c_stride_for_dim_[i] = c_stride;
        c_stride *= d.size;
        cols_ *= d.size;
      } else {
        throw std::invalid_argument("dim dir must be 'r' or 'c'");
      }
    }
    numel_ = rows_ * cols_;
  }

  // 基本信息
  index_t dims() const { return static_cast<index_t>(dims_.size()); }
  index_t rows() const { return rows_; }
  index_t cols() const { return cols_; }
  index_t numel() const { return numel_; }
  const std::vector<Dim>& spec() const { return dims_; }
  const std::vector<index_t>& r_strides() const { return r_stride_for_dim_; }
  const std::vector<index_t>& c_strides() const { return c_stride_for_dim_; }

  // ---------- 高维坐标 <-> 2D ----------
  std::pair<index_t, index_t> hd_to_rc(const std::vector<index_t>& hd_idx) const {
    check_hd_index(hd_idx);
    index_t row = 0, col = 0;
    for (index_t i = 0; i < dims(); ++i) {
      const auto& d = dims_[i];
      auto v = hd_idx[i];
      if (v >= d.size) throw std::out_of_range(err_dim(i, v, d.size));
      if (d.dir == 'r')
        row += v * r_stride_for_dim_[i];
      else
        col += v * c_stride_for_dim_[i];
    }
    return {row, col};
  }

  std::vector<index_t> rc_to_hd(index_t row, index_t col) const {
    if (row >= rows_ || col >= cols_)
      throw std::out_of_range("rc out of range: (" + std::to_string(row) + "," + std::to_string(col) +
                              "), expect rows<" + std::to_string(rows_) + ", cols<" + std::to_string(cols_) + ")");
    std::vector<index_t> hd_idx(dims(), 0);
    for (index_t i = 0; i < dims(); ++i) {
      const auto& d = dims_[i];
      if (d.dir == 'r') {
        auto stride = r_stride_for_dim_[i];
        hd_idx[i] = (row / stride) % d.size;
      } else {
        auto stride = c_stride_for_dim_[i];
        hd_idx[i] = (col / stride) % d.size;
      }
    }
    return hd_idx;
  }

  // ---------- 2D <-> offset（行主序），支持自定义 ld ----------
  index_t rc_to_offset(index_t row, index_t col, index_t ld = 0) const {
    if (ld == 0) ld = cols_;
    if (row >= rows_ || col >= cols_) throw std::out_of_range("rc out of range for rc_to_offset");
    return row * ld + col;
  }

  std::pair<index_t, index_t> offset_to_rc(index_t offset, index_t ld = 0) const {
    if (ld == 0) ld = cols_;
    index_t row = offset / ld;
    index_t col = offset % ld;
    if (row >= rows_ || col >= cols_) throw std::out_of_range("offset out of range for given ld");
    return {row, col};
  }

  // ---------- 高维坐标 <-> offset（组合/分解） ----------
  index_t hd_to_offset(const std::vector<index_t>& hd_idx, index_t ld = 0) const {
    auto [r, c] = hd_to_rc(hd_idx);
    return rc_to_offset(r, c, ld);
  }

  std::vector<index_t> offset_to_hd(index_t offset, index_t ld = 0) const {
    auto [r, c] = offset_to_rc(offset, ld);
    return rc_to_hd(r, c);
  }

  // ---------- 工具：把某一组 r/c 维做“混合进位”分解/合成 ----------
  // 给定行坐标 row，分解到所有 'r' 维的 digits（低维在前）
  std::vector<index_t> decompose_row(index_t row) const {
    if (row >= rows_) throw std::out_of_range("row out of range in decompose_row");
    std::vector<index_t> res(dims(), 0);
    for (index_t i = 0; i < dims(); ++i) {
      if (dims_[i].dir == 'r') {
        auto stride = r_stride_for_dim_[i];
        res[i] = (row / stride) % dims_[i].size;
      }
    }
    return res;  // 只有 'r' 维位置含有有效 digit
  }
  // 给定列坐标 col，分解到所有 'c' 维的 digits（低维在前）
  std::vector<index_t> decompose_col(index_t col) const {
    if (col >= cols_) throw std::out_of_range("col out of range in decompose_col");
    std::vector<index_t> res(dims(), 0);
    for (index_t i = 0; i < dims(); ++i) {
      if (dims_[i].dir == 'c') {
        auto stride = c_stride_for_dim_[i];
        res[i] = (col / stride) % dims_[i].size;
      }
    }
    return res;  // 只有 'c' 维位置含有有效 digit
  }
  // 合成行坐标（仅读取 'r' 维的位置）
  index_t compose_row(const std::vector<index_t>& digits) const {
    if (digits.size() != dims()) throw std::invalid_argument("digits dim mismatch");
    index_t row = 0;
    for (index_t i = 0; i < dims(); ++i)
      if (dims_[i].dir == 'r') {
        if (digits[i] >= dims_[i].size) throw std::out_of_range(err_dim(i, digits[i], dims_[i].size));
        row += digits[i] * r_stride_for_dim_[i];
      }
    return row;
  }
  // 合成列坐标（仅读取 'c' 维的位置）
  index_t compose_col(const std::vector<index_t>& digits) const {
    if (digits.size() != dims()) throw std::invalid_argument("digits dim mismatch");
    index_t col = 0;
    for (index_t i = 0; i < dims(); ++i)
      if (dims_[i].dir == 'c') {
        if (digits[i] >= dims_[i].size) throw std::out_of_range(err_dim(i, digits[i], dims_[i].size));
        col += digits[i] * c_stride_for_dim_[i];
      }
    return col;
  }

 private:
  void check_hd_index(const std::vector<index_t>& hd_idx) const {
    if (hd_idx.size() != dims())
      throw std::invalid_argument("hd index dim mismatch: got " + std::to_string(hd_idx.size()) + ", expect " +
                                  std::to_string(dims()));
  }
  static std::string err_dim(index_t i, index_t v, index_t sz) {
    return "hd index out of range at dim " + std::to_string(i) + ": got " + std::to_string(v) + ", expect < " +
           std::to_string(sz);
  }

  std::vector<Dim> dims_;
  std::vector<index_t> r_stride_for_dim_;
  std::vector<index_t> c_stride_for_dim_;
  index_t rows_{1}, cols_{1}, numel_{0};
};

// ===== 示例与自测（可选） =====
// g++ -O2 test.cpp -DPACKED2D_DEMO && ./a.out
#ifdef PACKED2D_DEMO
int main() {
  // 任意数量与顺序的 r/c 维；低 -> 高
  Packed2DLayout p({
      {4, 'r'}, {8, 'c'}, {2, 'r'}, {3, 'c'}  // rows=4*2=8, cols=8*3=24, numel=192
  });

  std::cout << "rows=" << p.rows() << " cols=" << p.cols() << " numel=" << p.numel() << "\n";

  // 高维 -> rc -> offset
  std::vector<std::size_t> hd = {3, 5, 1, 2};
  auto [r, c] = p.hd_to_rc(hd);
  auto off = p.hd_to_offset(hd);
  std::cout << "hd -> rc=(" << r << "," << c << "), off=" << off << "\n";

  // 反向
  auto hd2 = p.offset_to_hd(off);
  std::cout << "offset->hd: ";
  for (auto v : hd2) std::cout << v << " ";
  std::cout << "\n";

  // 只分解/合成行、列
  auto rdigits = p.decompose_row(r);
  auto cdigits = p.decompose_col(c);
  auto r2 = p.compose_row(rdigits);
  auto c2 = p.compose_col(cdigits);
  std::cout << "compose row=" << r2 << " col=" << c2 << "\n";
  return 0;
}
#endif

#endif