# Deep Research Agent Benchmark Setup

This guide demonstrates how to use the Deep Research Agent for evaluation using the `run_research.py` and `run_research_concurrent.py` scripts.

- **Sequential Mode** (`run_research.py`): Process queries one at a time
- **Concurrent Mode** (`run_research_concurrent.py`): Process multiple queries in parallel for faster evaluation

## Quick Usage Example

Process individual queries using `run_research.py` from the benchmarks directory:

```bash
# Basic usage with default settings
python run_research.py "Your research query here" \
  --max-loops 1 \
  --disable-visualizations \
  --output /path/to/output.json

# With file analysis
python run_research.py "Analyze this research paper" \
  --file /path/to/research_paper.pdf \
  --max-loops 1 \
  --disable-visualizations \
```

#### Common Parameters

- `--max-loops`: Number of web search iterations (default: 4-5)
- `--disable-visualizations`: Skip any visualizations like charts in report
- `--provider`: LLM provider (`openai`, `anthropic`, `groq`, `google`)
- `--model`: Specific model name (e.g., `o3-mini`, `claude-3-5-sonnet`, `gemini-2.5-pro`)
- `--benchmark-mode`: Optimized settings for QA style evaluation with citations
- `--qa-mode`: Simple question-answering mode
- `--file`: Path to file for analysis (PDF, TXT, etc.)

## DeepResearchBench Evaluation

Clone the DeepResearchBench repo:
```bash
git clone https://github.com/Ayanami0730/deep_research_bench.git
```

To run DeepResearchBench evaluation:

1. **Process all 100 queries**:
```bash
python run_research_concurrent.py \
  --benchmark-type drb \
  --max-workers 4 \
  --provider google \
  --model gemini-2.5-pro \
  --max-loops 5
```

2. **Convert to benchmark format**:
```bash
python process_drb.py \
  --input-dir deep_research_bench/data/test_data/raw_data/edr_reports_gemini \
  --model-name edr_gemini
```

The processed report will be saved to `deep_research_bench/data/test_data/raw_data/edr_gemini.jsonl`

3. **Run DeepResearchBench evaluation**:
```bash
cd deep_research_bench
# Add your model name (eg. edr_gemini) to TARGET_MODELS in run_benchmark.sh
bash run_benchmark.sh
```

The results will be written to `deep_research_bench/results`


## DeepConsult Evaluation

Clone the DeepConsult repo and follow the [installation steps](https://github.com/Su-Sea/ydc-deep-research-evals?tab=readme-ov-file#installation):
```bash
git clone https://github.com/Su-Sea/ydc-deep-research-evals.git
```

To run DeepConsult evaluation:

1. **Process DeepConsult CSV queries**:
```bash
python run_research_concurrent.py \
  --benchmark-type deepconsult \
  --max-workers 4 \
  --max-loops 10 \
  --provider google \
  --model gemini-2.5-pro
```

2. **Create responses CSV for evaluation**:
```bash
python process_deepconsult.py \
  --queries-file /path/to/queries.csv \
  --baseline-file /path/to/baseline_responses.csv \
  --reports-dir /path/to/generated_reports \
  --output-file /path/to/custom_output.csv
```

This script combines:
- Questions from the original `queries.csv`
- Baseline answers from existing responses
- Your generated candidate answers from the JSON files
- Output: `responses_EDR_vs_ARI_YYYY-MM-DD.csv`

3. **Run pairwise evaluation**:
```bash
cd benchmarks/ydc-deep-research-evals/evals
export OPENAI_API_KEY=""
python deep_research_pairwise_evals.py \
    --input-data /path/to/csv/previous/step \
    --output-dir results \
    --model gpt-4.1-2025-04-14 \
    --num-workers 4 \
    --metric-num-workers 3 \
    --metric-num-trials 3
```

## Monitoring and Debugging

### Real-time Progress Monitoring

The concurrent script provides detailed progress tracking:

- **Live progress updates** every 10 seconds showing completion rate and ETA
- **Individual task logging** with timing and performance metrics
- **Comprehensive summary** with success/failure statistics

Example output:
```
📈 Progress: 45/100 completed, 2 failed, 8 in progress, ETA: 12.3min
[Task 23] ✅ SUCCESS - Completed in 45.67s
[Task 23] 📊 Metrics: 3 loops, 12 sources, 8,234 chars
[Task 23] 📈 Throughput: 180 chars/second
```
