#!/bin/bash

echo "🚀 Starting run_research.sh script..."
echo "📅 Start time: $(date)"
echo "👤 User: $(whoami)"
echo "📁 Working directory: $(pwd)"
echo "==============================================="

LOGS_DIR="logs"
mkdir -p $LOGS_DIR

## Simple test
python -u run_research.py "what is ai?" \
  --max-loops 2 \
  --output sample_result.json > $LOGS_DIR/traj.log 2>&1 &

## DeepResearch Bench (DRB)
# python -u run_research_concurrent.py \
# --benchmark drb \
# --input /Users/akshara.prabhakar/Documents/deep_research/benchmarks/deep_research_bench/data/prompt_data/query.jsonl \
# --output_dir drb_trajectories \
# --collect-traj \
# --task_ids 81 > $LOGS_DIR/drb_traj1.log 2>&1 &

## DeepConsult
# python -u run_research_concurrent.py \
# --benchmark deepconsult \
# --input ydc-deep-research-evals/datasets/DeepConsult/queries.csv \
# --limit 1 \
# --output_dir deepconsult_trajectories > $LOGS_DIR/deepconsult_traj.log 2>&1 &
