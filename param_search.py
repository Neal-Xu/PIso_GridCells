import numpy as np
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


def run_train_and_analyze_on_gpu(rho, size, run_id, gpu_id, train_script, analyze_script):
    """
    Run one train + analyze subprocess on the assigned GPU.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Bind this process to one GPU

    print(f"\n[GPU {gpu_id}] Running: rho={rho}, size={size}, run={run_id}")

    train_cmd = [
        "python", train_script,
        "--rho", str(rho),
        "--size", str(size),
        "--run", str(run_id)
    ]
    print("Training:", " ".join(train_cmd))
    subprocess.run(train_cmd, env=env)

    analyze_cmd = [
        "python", analyze_script,
        "--rho", str(rho),
        "--size", str(size),
        "--run", str(run_id)
    ]
    print("Analyzing:", " ".join(analyze_cmd))
    subprocess.run(analyze_cmd, env=env)

def grid_search_and_analyze(
        rho_values,
        size_values,
        train_script="train_model.py",
        analyze_script="cell_statistics.py",
        repeat=1,
        max_workers=6,
        gpus=(0, 1)
):
    """
    Run training and analysis over a grid of rho and size values in parallel.

    Parameters:
    - rho_values: list or np.ndarray of rho values to try
    - size_values: list or np.ndarray of size values to try
    - train_script: training script name
    - analyze_script: script that runs analysis and saves JSON
    - repeat: how many times to repeat each experiment
    - max_workers: number of parallel processes
    - gpus: tuple of GPU ids to assign tasks
    """
    tasks = []
    gpu_count = len(gpus)

    for i, (rho, size, run_id) in enumerate(
        ( (rho, size, run_id)
          for rho in rho_values
          for size in size_values
          for run_id in range(repeat) )
    ):
        gpu_id = gpus[i % gpu_count]  # Round-robin GPU allocation
        tasks.append((rho, size, run_id, gpu_id))

    print(f"\nTotal tasks: {len(tasks)} (running with {max_workers} parallel workers)\n")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_train_and_analyze_on_gpu, rho, size, run_id, gpu_id, train_script, analyze_script)
            for rho, size, run_id, gpu_id in tasks
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print("Task failed:", e)

if __name__ == "__main__":
    # Sweep 1: vary rho
    rho_values = np.round(np.arange(0.8, 1.6 + 0.05, 0.1), 3)  # 0.7-0.9,0.8 twice,0.7 none
    size_values_fixed = [0.84]  # 0.995

    # Sweep 2: vary size
    size_values = np.round(np.arange(0.21, 0.45 + 0.005, 0.04), 3)  # (0.21, 0.45 + 0.005, 0.04) (0.49, 0.85 + 0.005, 0.04)
    rho_values_fixed = [0.8]

    # Run with 6 parallel workers across 2 GPUs
    print("### Running grid search (varying rho, fixed size)")
    grid_search_and_analyze(
        rho_values, size_values_fixed,
        repeat=30,
        max_workers=6,
        gpus=(0, 1)
    )

    print("### Running grid search (varying size, fixed rho)")
    grid_search_and_analyze(
        rho_values_fixed, size_values,
        repeat=30,
        max_workers=6,
        gpus=(0, 1)
    )
