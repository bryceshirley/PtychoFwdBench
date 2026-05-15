import logging
import time

import numpy as np

from ptycho_fwd_bench.dim_3.generators_3d import (
    generate_3d_blob_phantom,
    get_2d_airy_probe,
)
from ptycho_fwd_bench.dim_3.plotters_3d import plot_comparison, plot_phantom_slices
from ptycho_fwd_bench.dim_3.solvers_3d.batched_multislice import (
    StandardMultisliceSolver,
)
from ptycho_fwd_bench.dim_3.solvers_3d.batched_parallel_multislice import (
    ParallelSTEMSolver,
)

# =============================================================================
# LOGGING SETUP
# =============================================================================
# 1. Set global logging to INFO
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
# 2. Activate lower-level (DEBUG) logs specifically for your solver package
logging.getLogger("ptycho_fwd_bench").setLevel(logging.DEBUG)

# 3. Create a logger for this specific benchmark script
logger = logging.getLogger("benchmark_3d")

# Use cupy for direct GPU memory management in benchmark
try:
    import cupy as cp

    HAS_GPU = True
    logger.info("CuPy detected. Running in GPU mode.")
except ImportError:
    HAS_GPU = False
    cp = None
    logger.warning("CuPy not found. Falling back to CPU mode.")


# =============================================================================
# BENCHMARK EXECUTION
# =============================================================================


def run_benchmark():
    logger.info("Initializing 3D Benchmark Configuration...")

    # --- CONFIGURATION (Soft X-ray Parameters) ---
    n_physical = 200
    n_padding = 56
    total_n = n_physical + n_padding

    # Realistic Soft X-ray values (nm)
    physical_width = 200.0  # Increased FOV for 1.75nm wavelength
    wavelength = 1.756  # 706 eV
    probe_dia = 15.0  # Realistic FZP focus size
    probe_focus = -50.0  # Defocus upstream
    thickness = 600.0  # Standard thick sample

    dx = physical_width / n_physical
    nz_depth = 100  # Standard slice count for thick samples
    dz = thickness / nz_depth

    calc_shape = (total_n, total_n, nz_depth)
    full_map_shape = (512, 512, nz_depth)

    logger.info(
        f"Calculated Grid Shape (Padded): {calc_shape} | Full Map: {full_map_shape}"
    )
    logger.info(f"Step Sizes: dx={dx:.3f} nm, dz={dz:.3f} nm")

    # --- PRECISION & BATCHING ---
    n_scan = 40
    batch_size = 1
    n_iters = 2
    alpha = 1e-8  # Minimal wrap-around error for double precision

    margin = total_n // 2
    scan_min = margin
    scan_max = full_map_shape[0] - margin

    positions = [
        (r, c)
        for r in np.linspace(scan_min, scan_max, n_scan).astype(int)
        for c in np.linspace(scan_min, scan_max, n_scan).astype(int)
    ]
    logger.info(f"Generated {len(positions)} scan positions.")

    # --- PREPARE DATA ---
    logger.info("Generating 3D blob phantom...")
    n_background = 1.0 + 0j
    delta = -1.0e-4
    beta = 2.0e-5
    n_blobs = 600
    full_potential_map = generate_3d_blob_phantom(
        *full_map_shape,
        n_background=n_background,
        delta_n=delta,
        beta_n=beta,
        n_blobs=n_blobs,
    )
    if HAS_GPU:
        full_potential_map = cp.asarray(full_potential_map)
        n_mean = cp.mean(full_potential_map)
    else:
        n_mean = np.mean(full_potential_map)

    logger.info(f"Phantom mean refractive index: {n_mean:.6e}")

    logger.info("Generating 2D Airy probe...")
    probe = get_2d_airy_probe(total_n, total_n, dx, probe_dia, probe_focus, wavelength)

    # 1. Parallel Solver Setup (complex128/float64)
    logger.info("Setting up ParallelSTEMSolver...")
    p_solver = ParallelSTEMSolver(
        calc_shape, dx, dz, wavelength, n_mean=n_mean, alpha=alpha, gpu_id=0
    )
    p_solver.precompute_probe(probe)  # Restored precomputation

    # 2. Standard Solver Setup
    logger.info("Setting up StandardMultisliceSolver...")
    s_solver = StandardMultisliceSolver(
        calc_shape, dx, dz, wavelength, n_mean=n_mean, gpu_id=0
    )
    s_solver.precompute_probe(probe)

    # --- WARM-UP ---
    logger.info("Running warm-up passes to compile kernels...")
    warmup_pos = positions[:batch_size]
    # Signature updated: removed 'probe' as it's now precomputed
    _ = p_solver.run_scan(
        full_potential_map, warmup_pos, batch_size=batch_size, n_iter=n_iters
    )
    _ = s_solver.run_scan(full_potential_map, warmup_pos, batch_size=batch_size)
    if HAS_GPU:
        cp.cuda.Stream.null.synchronize()
    logger.info("Warm-up complete.")

    # --- TIMING PARALLEL ---
    logger.info("Starting Parallel Solver timing run...")
    if HAS_GPU:
        cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()

    p_exit_waves = p_solver.run_scan(
        full_potential_map, positions, batch_size=batch_size, n_iter=n_iters
    )

    if HAS_GPU:
        cp.cuda.Stream.null.synchronize()
    p_solve_time = time.perf_counter() - t0
    logger.info(f"Parallel Solver finished in {p_solve_time:.4f} seconds.")

    # --- TIMING STANDARD ---
    logger.info("Starting Standard Solver timing run...")
    if HAS_GPU:
        cp.get_default_memory_pool().free_all_blocks()
    if HAS_GPU:
        cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()

    s_exit_waves = s_solver.run_scan(
        full_potential_map, positions, batch_size=batch_size
    )

    if HAS_GPU:
        cp.cuda.Stream.null.synchronize()
    s_solve_time = time.perf_counter() - t1
    logger.info(f"Standard Solver finished in {s_solve_time:.4f} seconds.")

    # --- RESULTS ---
    p_rate = len(positions) / p_solve_time
    s_rate = len(positions) / s_solve_time

    logger.info("=========================================")
    logger.info("           BENCHMARK RESULTS             ")
    logger.info("=========================================")
    logger.info(f"Positions Scanned : {len(positions)}")
    logger.info(f"Parallel Speed    : {p_rate:.2f} pos/sec")
    logger.info(f"Standard Speed    : {s_rate:.2f} pos/sec")
    logger.info(f"Speedup           : {p_rate / s_rate:.2f}x")
    logger.info("=========================================")

    logger.info("Generating comparison plots...")
    p_res = cp.asnumpy(p_exit_waves[0]) if HAS_GPU else p_exit_waves[0]
    s_res = cp.asnumpy(s_exit_waves[0]) if HAS_GPU else s_exit_waves[0]
    plot_comparison(probe, p_res, s_res)
    plot_phantom_slices(
        cp.asnumpy(full_potential_map) if HAS_GPU else full_potential_map
    )
    logger.info("Benchmark execution complete.")


if __name__ == "__main__":
    run_benchmark()
