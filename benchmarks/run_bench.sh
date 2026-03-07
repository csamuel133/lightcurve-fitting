#!/bin/bash
#SBATCH --job-name=lcfit-bench
#SBATCH --partition=skylake-gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=benchmarks/bench_%j.log

set -euo pipefail

module load cuda/12.8.0
module load python-scientific/3.11.5-foss-2023b

cd /home/mcoughli/lightcurve-fitting

echo "=== Node: $(hostname), GPUs: ==="
nvidia-smi --list-gpus

echo ""
echo "########## Building (release + CUDA) ##########"
cargo build --release --features cuda --test bench_throughput 2>&1 | tail -5

echo ""
echo "########## Running throughput benchmark ##########"
cargo test --release --features cuda --test bench_throughput -- --ignored --nocapture 2>&1

echo ""
echo "########## Generating plots ##########"
python3 benchmarks/plot_throughput.py

echo ""
echo "=== Done ==="
