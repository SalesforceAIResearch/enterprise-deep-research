#!/bin/bash

echo "🚀 Starting run_research.sh script..."
echo "📅 Start time: $(date)"
echo "👤 User: $(whoami)"
echo "📁 Working directory: $(pwd)"
echo "==============================================="

LOGS_DIR="logs"
mkdir -p $LOGS_DIR

## Simple test
# python -u run_research.py "what is ai?" \
#   --max-loops 2 \
#   --output sample_result.json > $LOGS_DIR/traj.log 2>&1 &

## DeepResearch Bench (DRB)
# python -u run_research_concurrent.py \
# --benchmark drb \
# --input /Users/akshara.prabhakar/Documents/deep_research/benchmarks/deep_research_bench/data/prompt_data/query.jsonl \
# --output_dir drb_steer_trajectories \
# --max_concurrent 1 \
# --task_ids 1 \
# --enable-steering \
# --parallel-search \
# --parallel-max 4 \
# --collect-traj > $LOGS_DIR/drb_traj_steer.log 2>&1 &

## DeepConsult
# python -u run_research_concurrent.py \
# --benchmark deepconsult \
# --input ydc-deep-research-evals/datasets/DeepConsult/queries.csv \
# --limit 1 \
# --output_dir deepconsult_trajectories > $LOGS_DIR/deepconsult_traj.log 2>&1 &


## LiveResearchBench
# python -u run_research_concurrent.py \
# --benchmark lrb \
# --limit 5 \
# --output_dir lrb_trajectories_5loops_steer \
# --save_md
# --max_concurrent 3 > $LOGS_DIR/lrb_traj_steer.log 2>&1 &    

# HealthBench
# python run_research_concurrent.py \
#     --benchmark healthbench \
#     --input healthbench_data/all.json \
#     --output_dir healthbench_trajectories \
#     --provider google \
#     --model gemini-2.5-pro \
#     --max_concurrent 6 \
#     --max_loops 5 \
#     --collect-traj > $LOGS_DIR/healthbench_traj_all.log 2>&1 &

# python process_healthbench.py  --input-dir healthbench_trajectories
# python evaluate_healthbench.py healthbench_results/edr_healthbench_final_run_100.jsonl --grader-model gpt-4.1-2025-04-14