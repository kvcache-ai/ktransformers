# Four-main PyPI release preparation (draft)

This change prepares the next release while PRs continue landing in the four
repositories. It does **not** choose tonight's version numbers or freeze tonight's
main heads in a source file. `release-four-main.yml` resolves them when an operator
starts a run.

## Implemented in this draft

1. Read `main` from `kvcache-ai/ktransformers`, `sglang`, `transformers`, and
   `accelerate`. Retry the complete snapshot if a merge races two consecutive
   observations. GitHub offers no atomic cross-repository snapshot; a commit
   arriving after the successful observations belongs to the next candidate.
2. Record the four full SHAs and the workflow implementation SHA in
   `source-lock.json`. Downstream jobs fetch only these SHAs, never resolve main
   again. Full workflow reruns create a new snapshot/artifact; rerunning only a
   failed build consumes its original snapshot.
3. Create a fresh CPython 3.12 build environment and source checkouts. Build
   Accelerate, Transformers, SGLang, KTransformers, KT-Kernel and the SGL CUDA
   runtime from source. SGLang is independently checked out, rather than read
   from KT's potentially stale `third_party/sglang` gitlink.
4. Enable all KT CPU variants and CUDA `80;86;89;90;120`, and SGL's below-SM90
   code generation plus FA3. These are build settings; architecture coverage is
   not claimed verified until binary inspection and hardware tests are added.
5. Record wheel hashes, package versions, dependency metadata, toolchain versions,
   submodule states and source SHAs. Reject edits to tracked source during the
   build, missing/duplicate distributions, mismatched wheel metadata, stale
   intra-stack requirements, or direct conflicting upstream package dependencies
   in the user `sglang`/`sft` extras. Transitive conflicts still require the
   isolated dependency-resolution gate below.

Python metadata is checked before native compilation so stale pins fail early.
The full audit then verifies the actual native wheel versions. Partial build
outputs and diagnostics are retained even on failure and never become a release.

The raw SGL wheel is an intermediate build input, not a new public distribution
to upload. A successful raw build remains `publishable: false` in the report.
The workflow uses no PyPI credentials, has no publish job and has no automatic
push/version trigger. It does not change the current `release-pypi.yml` workflow
or any existing PyPI version.

## Before the first source build

- The workflow must be registered on the default branch before GitHub can
  dispatch it. While this PR is a draft, run its CPU tests locally and review
  the implementation; no GPU workflow has been dispatched by preparing the PR.
- Supply the `self-hosted, linux, x64, gpu, kt-cpu` runner, CUDA 12.8 and the
  compiler/system prerequisites. `KT_RELEASE_CUDA_HOME` can override the CUDA
  installation directory. Each run creates a fresh directory under `RUNNER_TEMP`.
- Align version and dependency declarations **in the respective main sources**
  before calling a candidate consistent. The workflow never rewrites runtime
  source or silently corrects stale pins. SGLang's existing
  `SGLANG_KT_VERSION` build interface uses KT's source version.
- The present main sources may still contain incompatible old pins (KT's post1
  version and Transformers post3 pin versus SGLang's KT post2/Transformers post4
  requirements). The audit reports these as failures, not a releasable stack.
- Build directories are retained on the runner for diagnostics. The first
  production iteration must add bounded cleanup/retention after uploads;
  repeated native builds need substantial free disk space.

## Required follow-up within this PR before enabling publication

- **Fresh carrier assembly:** parameterize the already-main SGL carrier tools.
  They currently hard-code post2 versions/CPython 3.12 and contain one-off
  metadata replacements/source overlays. Assembly must consume only this run's
  fresh raw wheels, preserve runtime source, record every generated metadata
  change, verify CUDA binary architecture coverage, and enforce per-wheel size
  limits. Never rebuild from the old frozen post2 carriers.
- **Version ownership:** select new, unused versions for changed artifacts. For
  an unchanged dependency, reuse an existing PyPI wheel only after proving the
  recorded main source and exact artifact hashes match. Never use
  `--skip-existing` to conceal a different file with the same version.
- **Isolated wheelhouse:** resolve all dependencies, record their hashes and
  prove `pip install "ktransformers[sglang]"` and the SFT extra select the expected
  artifacts, without upstream `transformers`/`accelerate` namespace collisions.
  Reconstruct installation environments from these artifacts, not source trees.
- **Hardware gates:** wire independently identified SM89 and SM120 runners,
  isolated caches/venvs, model paths and JSON launch/test profiles. Test GLM
  inference (including native multimodality), Qwen regression and the new Kimi
  SFT path as agreed for the release. A run on the wrong architecture, import-only
  check, missing model, or skipped E2E must not count as a passed hardware gate.
- **Artifact promotion:** depend on all required E2E jobs, download their exact
  artifacts, recheck SHA256 and upload in dependency order with KT last. Upload
  retries must reuse the same files. Retain the manifest/evidence long-term and
  install from production PyPI for the final verification.
- **Cutover:** replace the old release workflow's trigger only after the
  build-only candidate passes. Keep ordinary main merges separate from the
  explicit release trigger. Do not merge the frozen-hotfix upload workflow over
  the source build workflow.

The initial runner audit found only `qj5090-runner-1` registered in this repository;
the independent SM89 runner/profile is therefore an explicit remaining gate.
No exact final package versions are chosen in this draft.

## Local verification

```bash
python -m pytest -q .github/release/test_four_main.py
python .github/release/four_main.py --help
actionlint -config-file .github/release/actionlint.yaml .github/workflows/release-four-main.yml
```

The unit tests need `packaging` and `pytest`; they neither compile GPU kernels nor
access the network. The source audit and artifact hashes are evidence of what was
built, not substitutes for the pending installation and E2E tests.
