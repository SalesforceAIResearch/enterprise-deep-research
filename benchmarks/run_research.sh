#!/bin/bash

# Echo script start information
echo "🚀 Starting run_research.sh script..."
echo "📅 Start time: $(date)"
echo "🔢 Script PID: $$"
echo "👤 User: $(whoami)"
echo "📁 Working directory: $(pwd)"
echo "==============================================="

LOGS_DIR="logs"
mkdir -p $LOGS_DIR


# python -u run_research_concurrent.py --benchmark-type rqa --rate-limit 1.0 --max-workers 10 --provider google --model gemini-2.5-pro > $LOGS_DIR/rqa_test.log 2>&1 &

python -u run_research_concurrent.py --benchmark-type deepconsult --rate-limit 1.0 --max-workers 6 --provider google --model gemini-2.5-pro > $LOGS_DIR/deepconsult_test_10_loops.log 2>&1 &

## CONCURRENT
# python -u run_research_concurrent.py --max-workers 2 --max-loops 10  --rate-limit 3 --disable-visualizations --task-ids '1,2,3,51,52' > $LOGS_DIR/sfr_auto_report.log 2>&1 &

### ResearchQA
## SEQUENTIAL
# python run_research.py "What are the key factors affecting the compressive strength and hardening behavior of geopolymer concretes that utilize Shirasu aggregates?" --max-loops 2 --output /Users/akshara.prabhakar/Documents/deep_research/benchmarks/gaia_results/sfr_test_report.json --disable-visualizations --benchmark-mode
