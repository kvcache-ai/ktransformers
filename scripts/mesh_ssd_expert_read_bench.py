#!/usr/bin/env python3
"""Benchmark expert-shaped parallel O_DIRECT reads for MESH.

The script indexes AMX safetensors, builds expert read groups that match the
current MESH io_uring layout, compiles a tiny liburing benchmark, and measures
how many expert-shaped reads the SSD can sustain in parallel.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import textwrap
import time


CPP_SOURCE = r"""
#include <liburing.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <unistd.h>

struct Frag {
  int sample = 0;
  int layer = 0;
  int numa = 0;
  int expert = 0;
  std::string name;
  uint64_t offset = 0;
  uint64_t size = 0;
  std::string path;
  int fd = -1;
};

struct Expert {
  int sample = 0;
  int layer = 0;
  int numa = 0;
  int expert = 0;
  std::vector<Frag> frags;
  uint64_t bytes = 0;
};

struct Req {
  const Frag* frag = nullptr;
  void* buffer = nullptr;
  int attempts = 0;
};

static double now_sec() {
  using Clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

static std::vector<int> parse_int_list(const std::string& raw) {
  std::vector<int> values;
  std::stringstream ss(raw);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) values.push_back(std::stoi(item));
  }
  return values;
}

static bool load_slots(const std::string& path, std::vector<Expert>& experts) {
  std::ifstream in(path);
  if (!in) return false;

  std::unordered_map<int, size_t> sample_to_index;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::stringstream ss(line);
    Frag f;
    ss >> f.sample >> f.layer >> f.numa >> f.expert >> f.name >> f.offset >> f.size;
    ss >> std::ws;
    std::getline(ss, f.path);
    if (f.path.empty() || f.size == 0) continue;

    auto it = sample_to_index.find(f.sample);
    if (it == sample_to_index.end()) {
      Expert e;
      e.sample = f.sample;
      e.layer = f.layer;
      e.numa = f.numa;
      e.expert = f.expert;
      experts.push_back(std::move(e));
      it = sample_to_index.emplace(f.sample, experts.size() - 1).first;
    }
    Expert& e = experts[it->second];
    e.frags.push_back(f);
    e.bytes += f.size;
  }
  return !experts.empty();
}

static double percentile(std::vector<double> values, double p) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const double pos = (values.size() - 1) * p;
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = static_cast<size_t>(std::ceil(pos));
  if (lo == hi) return values[lo];
  const double w = pos - static_cast<double>(lo);
  return values[lo] * (1.0 - w) + values[hi] * w;
}

static bool submit_req(io_uring* ring, Req& req, uint64_t idx) {
  io_uring_sqe* sqe = io_uring_get_sqe(ring);
  if (sqe == nullptr) {
    int rc = io_uring_submit(ring);
    if (rc < 0) {
      std::cerr << "io_uring_submit before retry failed: " << std::strerror(-rc) << "\n";
      return false;
    }
    sqe = io_uring_get_sqe(ring);
  }
  if (sqe == nullptr) {
    std::cerr << "io_uring_get_sqe returned null after submit\n";
    return false;
  }
  req.attempts += 1;
  io_uring_prep_read(sqe, req.frag->fd, req.buffer, req.frag->size, req.frag->offset);
  io_uring_sqe_set_data(sqe, reinterpret_cast<void*>(static_cast<uintptr_t>(idx)));
  return true;
}

int main(int argc, char** argv) {
  std::string slots_path;
  std::string conc_raw = "1,2,4,8,16,32";
  int rounds = 64;
  int warmup = 8;
  int retries = 1;
  bool direct = true;
  double max_buffer_gib = 8.0;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--slots") slots_path = need("--slots");
    else if (arg == "--concurrency") conc_raw = need("--concurrency");
    else if (arg == "--rounds") rounds = std::stoi(need("--rounds"));
    else if (arg == "--warmup") warmup = std::stoi(need("--warmup"));
    else if (arg == "--retries") retries = std::stoi(need("--retries"));
    else if (arg == "--direct") direct = std::stoi(need("--direct")) != 0;
    else if (arg == "--max-buffer-gib") max_buffer_gib = std::stod(need("--max-buffer-gib"));
    else {
      std::cerr << "unknown argument: " << arg << "\n";
      return 2;
    }
  }
  if (slots_path.empty()) {
    std::cerr << "--slots is required\n";
    return 2;
  }

  std::vector<Expert> experts;
  if (!load_slots(slots_path, experts)) {
    std::cerr << "no expert slots loaded from " << slots_path << "\n";
    return 2;
  }

  std::unordered_map<std::string, int> fds;
  const int open_flags = O_RDONLY | (direct ? O_DIRECT : 0);
  for (Expert& e : experts) {
    for (Frag& f : e.frags) {
      auto it = fds.find(f.path);
      if (it == fds.end()) {
        int fd = open(f.path.c_str(), open_flags);
        if (fd < 0) {
          std::cerr << "open failed path=" << f.path << " error=" << std::strerror(errno) << "\n";
          return 1;
        }
        it = fds.emplace(f.path, fd).first;
      }
      f.fd = it->second;
    }
  }

  size_t max_frags = 0;
  uint64_t min_expert_bytes = UINT64_MAX;
  uint64_t max_expert_bytes = 0;
  for (const Expert& e : experts) {
    max_frags = std::max(max_frags, e.frags.size());
    min_expert_bytes = std::min(min_expert_bytes, e.bytes);
    max_expert_bytes = std::max(max_expert_bytes, e.bytes);
  }
  std::vector<uint64_t> max_size_by_frag(max_frags, 0);
  for (const Expert& e : experts) {
    for (size_t j = 0; j < e.frags.size(); ++j) {
      max_size_by_frag[j] = std::max(max_size_by_frag[j], e.frags[j].size);
    }
  }
  const uint64_t one_expert_buffer_bytes =
      std::accumulate(max_size_by_frag.begin(), max_size_by_frag.end(), uint64_t{0});
  const uint64_t max_buffer_bytes =
      static_cast<uint64_t>(max_buffer_gib * 1024.0 * 1024.0 * 1024.0);

  std::cout << "# experts=" << experts.size()
            << " max_frags=" << max_frags
            << " expert_bytes_min_mib=" << (double)min_expert_bytes / 1048576.0
            << " expert_bytes_max_mib=" << (double)max_expert_bytes / 1048576.0
            << " direct=" << (direct ? 1 : 0)
            << " rounds=" << rounds
            << " warmup=" << warmup
            << " retries=" << retries << "\n";
  std::cout << "concurrency\texpert_bytes_mib\tlogical_gib\tsubmitted_gib\telapsed_s"
            << "\texperts_s\tlogical_gibs\tsubmitted_gibs"
            << "\tbatch_mean_ms\tbatch_p50_ms\tbatch_p95_ms\tbatch_p99_ms"
            << "\terrors\tretry_success\tshort_reads\tnegative_errors\tbuffer_mib\n";

  std::vector<int> conc_values = parse_int_list(conc_raw);
  for (int conc : conc_values) {
    if (conc <= 0) continue;
    const uint64_t estimated_buffer_bytes = one_expert_buffer_bytes * static_cast<uint64_t>(conc);
    if (estimated_buffer_bytes > max_buffer_bytes) {
      std::cerr << "skip concurrency=" << conc
                << " estimated_buffer_gib=" << (double)estimated_buffer_bytes / 1073741824.0
                << " max_buffer_gib=" << max_buffer_gib << "\n";
      continue;
    }

    std::vector<void*> buffers(conc * max_frags, nullptr);
    bool alloc_ok = true;
    for (int i = 0; i < conc && alloc_ok; ++i) {
      for (size_t j = 0; j < max_frags; ++j) {
        const uint64_t sz = std::max<uint64_t>(4096, max_size_by_frag[j]);
        void* p = nullptr;
        if (posix_memalign(&p, 4096, sz) != 0 || p == nullptr) {
          alloc_ok = false;
          break;
        }
        buffers[i * max_frags + j] = p;
      }
    }
    if (!alloc_ok) {
      std::cerr << "allocation failed for concurrency=" << conc << "\n";
      for (void* p : buffers) free(p);
      continue;
    }

    io_uring ring;
    const unsigned ring_depth = std::max<unsigned>(64, static_cast<unsigned>(conc * max_frags + 32));
    int init_rc = io_uring_queue_init(ring_depth, &ring, 0);
    if (init_rc < 0) {
      std::cerr << "io_uring_queue_init failed: " << std::strerror(-init_rc) << "\n";
      for (void* p : buffers) free(p);
      return 1;
    }

    uint64_t logical_bytes = 0;
    uint64_t submitted_bytes = 0;
    uint64_t persistent_errors = 0;
    uint64_t retry_success = 0;
    uint64_t short_reads = 0;
    uint64_t negative_errors = 0;
    std::vector<double> batch_ms;
    batch_ms.reserve(rounds);

    const int total_rounds = warmup + rounds;
    const double t0_total = now_sec();
    for (int r = 0; r < total_rounds; ++r) {
      std::vector<Req> reqs;
      uint64_t round_logical_bytes = 0;
      reqs.reserve(conc * max_frags);

      for (int c = 0; c < conc; ++c) {
        const Expert& e = experts[(static_cast<size_t>(r) * conc + c) % experts.size()];
        round_logical_bytes += e.bytes;
        for (size_t j = 0; j < e.frags.size(); ++j) {
          Req req;
          req.frag = &e.frags[j];
          req.buffer = buffers[c * max_frags + j];
          reqs.push_back(req);
        }
      }

      const double batch_t0 = now_sec();
      uint64_t round_submitted_bytes = 0;
      for (size_t i = 0; i < reqs.size(); ++i) {
        if (!submit_req(&ring, reqs[i], i)) {
          io_uring_queue_exit(&ring);
          for (void* p : buffers) free(p);
          return 1;
        }
        round_submitted_bytes += reqs[i].frag->size;
      }
      int submit_rc = io_uring_submit(&ring);
      if (submit_rc < 0) {
        std::cerr << "io_uring_submit failed: " << std::strerror(-submit_rc) << "\n";
        io_uring_queue_exit(&ring);
        for (void* p : buffers) free(p);
        return 1;
      }

      size_t remaining = reqs.size();
      while (remaining > 0) {
        io_uring_cqe* cqe = nullptr;
        int wait_rc = io_uring_wait_cqe(&ring, &cqe);
        if (wait_rc < 0) {
          std::cerr << "io_uring_wait_cqe failed: " << std::strerror(-wait_rc) << "\n";
          io_uring_queue_exit(&ring);
          for (void* p : buffers) free(p);
          return 1;
        }
        const uint64_t idx = static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(io_uring_cqe_get_data(cqe)));
        const int res = cqe->res;
        io_uring_cqe_seen(&ring, cqe);
        if (idx >= reqs.size()) {
          persistent_errors++;
          remaining--;
          continue;
        }
        Req& req = reqs[idx];
        const bool ok = res == static_cast<int>(req.frag->size);
        if (ok) {
          if (req.attempts > 1) retry_success++;
          remaining--;
          continue;
        }
        if (res < 0) {
          negative_errors++;
        } else {
          short_reads++;
        }
        if (req.attempts <= retries) {
          if (!submit_req(&ring, req, idx)) {
            persistent_errors++;
            remaining--;
            continue;
          }
          round_submitted_bytes += req.frag->size;
          int retry_submit_rc = io_uring_submit(&ring);
          if (retry_submit_rc < 0) {
            std::cerr << "io_uring_submit retry failed: " << std::strerror(-retry_submit_rc) << "\n";
            persistent_errors++;
            remaining--;
          }
        } else {
          persistent_errors++;
          remaining--;
        }
      }

      const double elapsed_ms = (now_sec() - batch_t0) * 1000.0;
      if (r >= warmup) {
        batch_ms.push_back(elapsed_ms);
        logical_bytes += round_logical_bytes;
        submitted_bytes += round_submitted_bytes;
      }
    }
    const double elapsed_s = now_sec() - t0_total;
    const double measured_s = std::accumulate(batch_ms.begin(), batch_ms.end(), 0.0) / 1000.0;
    const double denom = measured_s > 0.0 ? measured_s : elapsed_s;
    const double expert_bytes_mib = static_cast<double>(max_expert_bytes) / 1048576.0;
    const double logical_gib = static_cast<double>(logical_bytes) / 1073741824.0;
    const double submitted_gib = static_cast<double>(submitted_bytes) / 1073741824.0;
    const double experts_s = (static_cast<double>(conc) * rounds) / denom;
    const double logical_gibs = logical_gib / denom;
    const double submitted_gibs = submitted_gib / denom;
    const double mean_ms = batch_ms.empty()
        ? 0.0
        : std::accumulate(batch_ms.begin(), batch_ms.end(), 0.0) / batch_ms.size();

    std::cout << conc << "\t"
              << std::fixed << std::setprecision(3)
              << expert_bytes_mib << "\t"
              << logical_gib << "\t"
              << submitted_gib << "\t"
              << elapsed_s << "\t"
              << experts_s << "\t"
              << logical_gibs << "\t"
              << submitted_gibs << "\t"
              << mean_ms << "\t"
              << percentile(batch_ms, 0.50) << "\t"
              << percentile(batch_ms, 0.95) << "\t"
              << percentile(batch_ms, 0.99) << "\t"
              << persistent_errors << "\t"
              << retry_success << "\t"
              << short_reads << "\t"
              << negative_errors << "\t"
              << (double)estimated_buffer_bytes / 1048576.0
              << "\n";

    io_uring_queue_exit(&ring);
    for (void* p : buffers) free(p);
  }

  for (auto& item : fds) close(item.second);
  return 0;
}
"""


TENSOR_RE = re.compile(
    r"^blk\.(?P<layer>\d+)\.ffn_(?P<proj>up|gate|down)_exps\."
    r"(?P<expert>\d+)\.numa\.(?P<numa>\d+)\.(?P<kind>weight|scale|mins)$"
)


def read_safetensor_header(path: Path) -> dict[str, dict]:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise RuntimeError(f"{path}: missing safetensors header size")
        header_size = struct.unpack("<Q", raw)[0]
        header = json.loads(f.read(header_size))
    base = 8 + header_size
    out = {}
    for key, info in header.items():
        if key == "__metadata__":
            continue
        begin, end = info["data_offsets"]
        out[key] = {
            "path": str(path),
            "offset": base + int(begin),
            "size": int(end) - int(begin),
            "dtype": info.get("dtype"),
            "shape": info.get("shape"),
        }
    return out


def index_safetensors(root: Path) -> dict[str, dict]:
    files = sorted(root.rglob("*.safetensors")) if root.is_dir() else [root]
    if not files:
        raise FileNotFoundError(f"No safetensors files found under {root}")
    entries: dict[str, dict] = {}
    for path in files:
        entries.update(read_safetensor_header(path))
    return entries


def parse_csv_ints(raw: str | None) -> set[int] | None:
    if raw is None or raw == "" or raw.lower() == "all":
        return None
    return {int(x) for x in raw.split(",") if x.strip()}


def build_expert_groups(
    entries: dict[str, dict],
    layers: set[int] | None,
    numas: set[int] | None,
    include_mins: bool,
) -> list[dict]:
    groups: dict[tuple[int, int, int], dict[str, dict]] = {}
    for key, info in entries.items():
        m = TENSOR_RE.match(key)
        if not m:
            continue
        layer = int(m.group("layer"))
        numa = int(m.group("numa"))
        expert = int(m.group("expert"))
        if layers is not None and layer not in layers:
            continue
        if numas is not None and numa not in numas:
            continue
        frag_name = f"{m.group('proj')}.{m.group('kind')}"
        groups.setdefault((layer, numa, expert), {})[frag_name] = info

    required = ["gate.weight", "gate.scale", "up.weight", "up.scale", "down.weight", "down.scale"]
    if include_mins:
        required += ["gate.mins", "up.mins", "down.mins"]

    result = []
    for (layer, numa, expert), frags in sorted(groups.items()):
        if all(name in frags for name in required):
            ordered = [(name, frags[name]) for name in required]
            result.append({"layer": layer, "numa": numa, "expert": expert, "frags": ordered})
    if not result:
        raise RuntimeError("No complete AMX expert groups found for the requested layer/NUMA filter")
    return result


def choose_samples(groups: list[dict], sample_experts: int, seed: int) -> list[dict]:
    if sample_experts <= 0 or sample_experts >= len(groups):
        return groups
    rng = random.Random(seed)
    sampled = rng.sample(groups, sample_experts)
    return sorted(sampled, key=lambda g: (g["layer"], g["numa"], g["expert"]))


def write_slots_tsv(groups: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# sample layer numa expert frag offset size path\n")
        for sample, group in enumerate(groups):
            for frag_name, info in group["frags"]:
                f.write(
                    f"{sample}\t{group['layer']}\t{group['numa']}\t{group['expert']}\t"
                    f"{frag_name}\t{info['offset']}\t{info['size']}\t{info['path']}\n"
                )


def compile_bench(out_dir: Path) -> Path:
    cxx = shutil.which("g++") or shutil.which("c++")
    if cxx is None:
        raise RuntimeError("g++/c++ not found on PATH")
    source = out_dir / "mesh_iouring_read_bench.cpp"
    binary = out_dir / "mesh_iouring_read_bench"
    source.write_text(CPP_SOURCE, encoding="utf-8")
    cmd = [cxx, "-O2", "-std=c++17", str(source), "-luring", "-o", str(binary)]
    subprocess.run(cmd, check=True)
    return binary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-path", default="/mnt/data2/models/Qwen3.5-35B-A3B-AMXINT4-NUMA2-MESH")
    parser.add_argument("--layers", default="all", help="Comma-separated layer ids, or all")
    parser.add_argument("--numas", default="all", help="Comma-separated NUMA/tp ids, or all")
    parser.add_argument("--sample-experts", type=int, default=512, help="Number of expert groups sampled")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--concurrency", default="1,2,4,8,12,16,24,32,48,64")
    parser.add_argument("--rounds", type=int, default=48)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--direct", type=int, default=1, help="1=O_DIRECT, 0=buffered read")
    parser.add_argument("--include-mins", action="store_true")
    parser.add_argument("--max-buffer-gib", type=float, default=8.0)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    weight_path = Path(args.weight_path).expanduser().resolve()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else Path.home() / f"mesh_ssd_bench_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = index_safetensors(weight_path)
    groups = build_expert_groups(
        entries,
        layers=parse_csv_ints(args.layers),
        numas=parse_csv_ints(args.numas),
        include_mins=args.include_mins,
    )
    samples = choose_samples(groups, args.sample_experts, args.seed)
    slots_tsv = out_dir / "expert_slots.tsv"
    write_slots_tsv(samples, slots_tsv)

    expert_bytes = [sum(info["size"] for _, info in g["frags"]) for g in samples]
    metadata = {
        "weight_path": str(weight_path),
        "out_dir": str(out_dir),
        "total_tensor_entries": len(entries),
        "complete_expert_groups": len(groups),
        "sampled_expert_groups": len(samples),
        "layers": args.layers,
        "numas": args.numas,
        "include_mins": args.include_mins,
        "expert_bytes_min": min(expert_bytes),
        "expert_bytes_max": max(expert_bytes),
        "expert_bytes_mean": sum(expert_bytes) / len(expert_bytes),
        "concurrency": args.concurrency,
        "rounds": args.rounds,
        "warmup": args.warmup,
        "direct": bool(args.direct),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2), flush=True)
    if args.dry_run:
        print(f"Dry run complete. Slots TSV: {slots_tsv}")
        return 0

    binary = compile_bench(out_dir)
    result_tsv = out_dir / "results.tsv"
    cmd = [
        str(binary),
        "--slots",
        str(slots_tsv),
        "--concurrency",
        args.concurrency,
        "--rounds",
        str(args.rounds),
        "--warmup",
        str(args.warmup),
        "--retries",
        str(args.retries),
        "--direct",
        str(args.direct),
        "--max-buffer-gib",
        str(args.max_buffer_gib),
    ]
    print("Running:", " ".join(cmd), flush=True)
    with result_tsv.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        rc = proc.wait()
    print(f"Result TSV: {result_tsv}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
