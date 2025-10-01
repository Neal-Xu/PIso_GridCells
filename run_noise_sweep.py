# run_noise_sweep.py
# --------------------------------------------------------------------------------------
# Run a noise-std sweep for your model WITHOUT changing train.py / representation.py.
# - Keeps rho=0.8 and size=0.5 fixed (you can tweak defaults below).
# - Does NOT overload "run id" to encode noise. Run id still means repetition index.
# - For each run, we iterate over std values SEQUENTIALLY (to avoid file overwrite),
#   and after each train+analyze round we rename:
#       data/saved-models/model_rho{...}_size{...}_run{...}.pth
#           -> data/saved-models/model_rho{...}_size{...}_run{...}_std{...}.pth
#       data/results/rho{...}_size{...}_run{...}.json
#           -> data/results/rho{...}_size{...}_run{...}_std{...}.json
#
# How it works:
# - We set env["STD"] for each subprocess so your model reads the noise std from
#   the environment (using the small change you added in build_encoder()).
# - We parallelize ACROSS runs (ProcessPoolExecutor); inside each run we loop all stds
#   synchronously to avoid clobbering files that share the same (rho,size,run) tag.
# - Only .pth and .json are renamed with the noise suffix; other artifacts are ignored.
#
# Usage:
#   python run_noise_sweep.py
# (Optionally edit DEFAULTS at the bottom or pass CLI flags.)
# --------------------------------------------------------------------------------------

import os
import shutil
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, Sequence

# ----------------------------- Utilities -----------------------------

def tag_from(rho: float, size: float, run: int) -> str:
    """Match train.py's file tag format: replace '.' with 'p'."""
    return f"rho{rho}_size{size}_run{run}".replace('.', 'p')

def std_str(std: float) -> str:
    """Noise std string compatible with tag format."""
    return str(std).replace('.', 'p')

def default_paths_for_tag(tag: str) -> dict:
    """Paths produced by train.py / representation.py BEFORE renaming."""
    base = os.path.dirname(os.path.abspath(__file__))
    return {
        "pth":  os.path.join(base, "data", "saved-models", f"model_{tag}.pth"),
        "json": os.path.join(base, "data", "results",       f"{tag}.json"),
    }

def noise_paths_for_tag(tag: str, std: float) -> dict:
    """Desired paths AFTER renaming, encoding std in filenames."""
    base = os.path.dirname(os.path.abspath(__file__))
    tag_std = f"{tag}_std{std_str(std)}"
    return {
        "pth":  os.path.join(base, "data", "saved-models", f"model_{tag_std}.pth"),
        "json": os.path.join(base, "data", "results",       f"{tag_std}.json"),
    }

def safe_move(src: str, dst: str):
    """Move if exists; create parent dirs; print action."""
    if not os.path.exists(src):
        print(f"[WARN] File not found, skip: {src}")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    print(f"[ARCHIVE] {os.path.basename(src)} -> {os.path.basename(dst)}")

# ----------------------------- Core runners -----------------------------

def run_train_and_analyze(rho: float, size: float, run: int, std: float, gpu_id: int,
                          train_script: str, analyze_script: str):
    """
    Launch train.py and cell_statistics.py as subprocesses with a given STD on a given GPU.
    Then rename ONLY the .pth and .json files to include the noise suffix.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # bind to a GPU
    env["STD"] = str(std)                      # the model reads STD from env

    tag = tag_from(rho, size, run)
    print(f"\n[GPU {gpu_id}] rho={rho}, size={size}, run={run}, std={std}  (tag={tag})")

    # Train
    train_cmd = ["python", train_script, "--rho", str(rho), "--size", str(size), "--run", str(run)]
    print("[TRAIN ]", " ".join(train_cmd))
    subprocess.run(train_cmd, env=env, check=True)

    # Analyze (this creates/overwrites data/results/{tag}.json)
    analyze_cmd = ["python", analyze_script, "--rho", str(rho), "--size", str(size), "--run", str(run), "--grid-threshold", "0.7"]
    print("[ANALYZE]", " ".join(analyze_cmd))
    subprocess.run(analyze_cmd, env=env, check=True)

    # Rename/archive outputs to include std
    src = default_paths_for_tag(tag)
    dst = noise_paths_for_tag(tag, std)

    # Only rename .pth and .json as requested
    safe_move(src["pth"],  dst["pth"])
    safe_move(src["json"], dst["json"])

def worker_one_run(run: int, std_values: Sequence[float], rho: float, size: float,
                   gpu_id: int, train_script: str, analyze_script: str):
    """
    For a single run id, iterate std values SEQUENTIALLY to avoid file clobbering.
    """
    for std in std_values:
        run_train_and_analyze(
            rho=rho, size=size, run=run, std=std, gpu_id=gpu_id,
            train_script=train_script, analyze_script=analyze_script
        )

def launch_noise_sweep(std_values: Sequence[float],
                       repeats: int,
                       rho: float,
                       size: float,
                       gpus: Sequence[int],
                       max_workers: int,
                       train_script: str,
                       analyze_script: str):
    """
    Parallelize ACROSS runs (each run gets its own worker/GPU), and within each run
    iterate ALL std values serially. This preserves the meaning of run id and avoids
    filename clashes in train.py/representation.py.
    """
    gpus = list(gpus)
    gpu_count = max(1, len(gpus))

    print("\n=== Noise sweep configuration ===")
    print(f"rho={rho}, size={size}")
    print(f"std_values={list(std_values)}")
    print(f"repeats per std={repeats}")
    print(f"parallel workers={max_workers}, gpus={gpus}\n")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for run in range(repeats):
            gpu_id = gpus[run % gpu_count]
            futures.append(
                ex.submit(
                    worker_one_run, run, tuple(std_values), rho, size,
                    gpu_id, train_script, analyze_script
                )
            )
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print("[ERROR] A run failed:", e)

# ----------------------------- CLI -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Noise std sweep runner (rho and size fixed).")
    parser.add_argument("--rho", type=float, default=0.8, help="Fixed rho (default: 1.0)")
    parser.add_argument("--size", type=float, default=0.5, help="Fixed size (default: 0.84)")
    parser.add_argument("--std_list", type=str, default=  "15, 17, 19, 21, 23, 25",# "0, 5, 10, 15, 17, 19, 21, 23, 25, 30, 35, 40",
                        help="Comma-separated std values, e.g., '0.05,0.1,0.2'")
    parser.add_argument("--repeats", type=int, default=30, help="Repetitions per std (run ids)")
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated GPU ids, e.g., '0,1'")
    parser.add_argument("--max_workers", type=int, default=6, help="Max parallel workers (across runs)")
    parser.add_argument("--train_script", type=str, default="train_model.py", help="Training script filename")
    parser.add_argument("--analyze_script", type=str, default="cell_statistics.py",
                        help="Analysis script filename")
    return parser.parse_args()

# ----------------------------- Main -----------------------------

if __name__ == "__main__":
    args = parse_args()

    std_values = [float(s) for s in args.std_list.split(",") if s.strip() != ""]
    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]

    launch_noise_sweep(
        std_values=std_values,
        repeats=args.repeats,
        rho=args.rho,
        size=args.size,
        gpus=gpus,
        max_workers=args.max_workers,
        train_script=args.train_script,
        analyze_script=args.analyze_script,
    )
