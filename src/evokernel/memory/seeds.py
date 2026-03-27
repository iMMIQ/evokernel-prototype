from __future__ import annotations

from evokernel.domain.enums import Stage
from evokernel.domain.models import MemoryItem, VerificationOutcome


def ingest_seed_memory(
    memory_store,
    *,
    backend_id: str,
    backend_constraints: list[str],
) -> list[str]:
    items = build_seed_memory_items(
        backend_id=backend_id,
        backend_constraints=backend_constraints,
    )
    for item in items:
        memory_store.add(item)
    return [item.memory_id for item in items]


def build_seed_memory_items(
    *,
    backend_id: str,
    backend_constraints: list[str],
) -> list[MemoryItem]:
    if backend_id == "cpu_simd":
        return _build_cpu_simd_seed_memory(
            backend_constraints=backend_constraints,
        )
    return []


def _build_cpu_simd_seed_memory(
    *,
    backend_constraints: list[str],
) -> list[MemoryItem]:
    items = [
        _build_seed_item(
            memory_id="seed:cpu_simd:api:core:abi",
            operator_family="general",
            summary="CPU SIMD backend ABI and entrypoint contract.",
            memory_kind="backend_knowledge",
            retrieval_text=(
                "backend=cpu_simd\n"
                "memory_kind=backend_knowledge\n"
                "bundle=static_infrastructure\n"
                "category=abi\n"
                "symbols=evokernel_entry,extern \"C\",float32\n"
                "summary=Respect the CPU SIMD ABI and expose evokernel_entry.\n"
                "constraints:\n"
                + "\n".join(f"- {constraint}" for constraint in backend_constraints)
            ),
            code=(
                "extern \"C\" void evokernel_entry(...);\n"
                "// Return code only. Keep the exported symbol stable."
            ),
            stage=Stage.DRAFTING,
        ),
        _build_seed_item(
            memory_id="seed:cpu_simd:api:core:intrinsics",
            operator_family="general",
            summary="SIMD header and intrinsic coverage for x86/ARM CPU backends.",
            memory_kind="backend_knowledge",
            retrieval_text=(
                "backend=cpu_simd\n"
                "memory_kind=backend_knowledge\n"
                "bundle=static_infrastructure\n"
                "category=intrinsics\n"
                "symbols=immintrin.h,__m128,__m256,_mm_loadu_ps,_mm256_loadu_ps,"
                "vld1q_f32\n"
                "summary=Use portable SIMD headers and keep scalar fallbacks for tail"
                " handling."
            ),
            code=(
                "#include <immintrin.h>\n"
                "// x86 examples: __m128, __m256, _mm_loadu_ps, _mm256_storeu_ps\n"
                "// ARM NEON example: vld1q_f32"
            ),
            stage=Stage.DRAFTING,
        ),
        _build_seed_item(
            memory_id="seed:cpu_simd:api:elementwise:tails",
            operator_family="elementwise",
            summary="Elementwise vector kernels should use contiguous loads and explicit tail cleanup.",
            memory_kind="backend_knowledge",
            retrieval_text=(
                "backend=cpu_simd\n"
                "memory_kind=backend_knowledge\n"
                "category=elementwise\n"
                "keywords=vector_add,contiguous,tail handling,float32\n"
                "summary=Vector kernels should process the aligned bulk with SIMD and"
                " finish the remaining elements with a scalar loop."
            ),
            code=(
                "for (; i + width <= n; i += width) { /* SIMD body */ }\n"
                "for (; i < n; ++i) { out[i] = a[i] + b[i]; }"
            ),
            stage=Stage.DRAFTING,
        ),
        _build_seed_item(
            memory_id="seed:cpu_simd:api:reduction:horizontal",
            operator_family="reduction",
            summary="Reduction kernels need a partial-sum accumulator and a horizontal combine step.",
            memory_kind="backend_knowledge",
            retrieval_text=(
                "backend=cpu_simd\n"
                "memory_kind=backend_knowledge\n"
                "category=reduction\n"
                "keywords=reduce_sum,horizontal reduction,partial sums,float32\n"
                "summary=Accumulate lane-wise partial sums first, then fold them into a"
                " scalar result and keep the empty-input case well defined."
            ),
            code=(
                "float sum = 0.0f;\n"
                "// SIMD partial sums -> horizontal reduction -> scalar tail cleanup"
            ),
            stage=Stage.DRAFTING,
        ),
        _build_seed_item(
            memory_id="seed:cpu_simd:api:matmul:tiling",
            operator_family="matmul",
            summary="Matmul kernels benefit from cache-friendly blocking and register reuse.",
            memory_kind="backend_knowledge",
            retrieval_text=(
                "backend=cpu_simd\n"
                "memory_kind=backend_knowledge\n"
                "category=matmul\n"
                "keywords=matmul,tiling,blocking,register reuse,float32\n"
                "summary=Block the K dimension, keep the inner loops contiguous, and"
                " avoid repeatedly reloading the same tiles."
            ),
            code=(
                "for (int i0 = 0; i0 < M; i0 += tile_m) { /* tiled matmul */ }\n"
                "// Reuse A/B tiles across the inner accumulation loop."
            ),
            stage=Stage.DRAFTING,
        ),
        _build_seed_item(
            memory_id="seed:cpu_simd:api:normalization:stats",
            operator_family="normalization",
            summary="Normalization kernels must compute stable statistics before scaling and shifting.",
            memory_kind="backend_knowledge",
            retrieval_text=(
                "backend=cpu_simd\n"
                "memory_kind=backend_knowledge\n"
                "category=normalization\n"
                "keywords=layernorm,mean,variance,eps,float32\n"
                "summary=Compute mean and variance over the trailing axis, apply eps,"
                " then fuse gamma and beta only after normalization."
            ),
            code=(
                "float mean = ...;\n"
                "float variance = ...;\n"
                "out[i] = ((x[i] - mean) / sqrtf(variance + eps)) * gamma[i] + beta[i];"
            ),
            stage=Stage.DRAFTING,
        ),
        _build_seed_item(
            memory_id="seed:cpu_simd:hint:refinement:general",
            operator_family="general",
            summary="Refinement should preserve correctness first and optimize only one bottleneck at a time.",
            memory_kind="refinement_hint",
            retrieval_text=(
                "backend=cpu_simd\n"
                "memory_kind=refinement_hint\n"
                "category=best_practice\n"
                "keywords=refining,preserve correctness,local changes,latency\n"
                "summary=Keep the function signature stable, isolate one optimization,"
                " and retain a scalar fallback while iterating on latency."
            ),
            code=(
                "// Preserve correctness gates.\n"
                "// Apply one targeted optimization per refinement attempt."
            ),
            stage=Stage.REFINING,
        ),
    ]
    return items


def _build_seed_item(
    *,
    memory_id: str,
    operator_family: str,
    summary: str,
    memory_kind: str,
    retrieval_text: str,
    code: str,
    stage: Stage,
) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        task_id="__seed__",
        backend_id="cpu_simd",
        operator_family=operator_family,
        stage=stage,
        code=code,
        summary=summary,
        context_summary="seed memory",
        memory_kind=memory_kind,
        reward=0.0,
        is_feasible=True,
        became_start_point=False,
        verifier_outcome=VerificationOutcome(
            anti_hack_passed=True,
            compile_passed=True,
            correctness_passed=True,
            latency_ms=None,
            error_category=None,
            feedback_summary="seed knowledge",
        ),
        retrieval_text=retrieval_text,
    )
