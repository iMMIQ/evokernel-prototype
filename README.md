# EvoKernel Prototype

EvoKernel Prototype is a Python-based reproduction of the paper's core framework shape, aimed at building a runnable `drafting -> refining` kernel-synthesis loop with shared memory, value-driven retrieval, and multi-gate verification.

## Goal

This repository is not trying to reproduce the paper's Ascend results first. The immediate target is a pluggable prototype that:

- runs locally with `uv`
- uses a real LLM provider abstraction
- executes a CPU SIMD backend end-to-end
- keeps the backend/verifier boundary clean enough to migrate to Ascend later

## Current Status

The project is in the planning/bootstrap phase.

- Paper notes and algorithm summary are in [EvoKernel_算法流程整理.md](./EvoKernel_%E7%AE%97%E6%B3%95%E6%B5%81%E7%A8%8B%E6%95%B4%E7%90%86.md) and [doc/paper.txt](./doc/paper.txt)
- The approved design spec is in [docs/superpowers/specs/2026-03-26-evokernel-prototype-design.md](./docs/superpowers/specs/2026-03-26-evokernel-prototype-design.md)
- The implementation plan is in [docs/superpowers/plans/2026-03-26-evokernel-prototype.md](./docs/superpowers/plans/2026-03-26-evokernel-prototype.md)

No runnable source code has been checked in yet.

## Planned First Milestone

- Python project managed by `uv`
- pluggable generator/backend/verifier interfaces
- CPU SIMD benchmark tasks: `vector_add`, `reduce_sum`, `matmul_tiled`, `layernorm`
- draft/refine episode loop with memory reuse across tasks

## License

This repository is licensed under the GNU General Public License v3.0. See [LICENSE](./LICENSE).
