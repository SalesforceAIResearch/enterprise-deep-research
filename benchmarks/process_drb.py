#!/usr/bin/env python3
"""
Convert individual JSON research reports to JSONL format for DeepResearchBench evaluation.

Usage:
    python process_drb.py --input-dir /path/to/json/files --model-name your_model_name
"""

import json
import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Convert individual JSON research reports to JSONL format for DeepResearchBench evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        required=True,
        help="Directory containing individual JSON report files",
    )

    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        required=True,
        help="Model name for the output JSONL filename",
    )

    args = parser.parse_args()

    # Validate input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return 1

    # Collect all JSON files in the directory
    json_files = [f for f in os.listdir(args.input_dir) if f.endswith(".json")]

    if not json_files:
        print(f"Error: No JSON files found in '{args.input_dir}'")
        return 1

    print(f"Found {len(json_files)} JSON files in '{args.input_dir}'")

    all_reports = []
    for file in json_files:
        file_path = os.path.join(args.input_dir, file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                # Extract required fields
                report = {
                    "id": data["id"],
                    "prompt": data["prompt"],
                    "article": data["article"],
                }
                all_reports.append(report)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Error processing file '{file}': {e}")
            continue
        except Exception as e:
            print(f"Warning: Unexpected error processing file '{file}': {e}")
            continue

    if not all_reports:
        print("Error: No valid reports were processed")
        return 1

    # Sort all reports by id
    all_reports.sort(key=lambda x: x["id"])

    with open(f"{args.model_name}.jsonl", "w", encoding="utf-8") as f:
        for report in all_reports:
            f.write(json.dumps(report, ensure_ascii=False) + "\n")

    print(f"Successfully processed {len(all_reports)} reports")
    print(f"Output saved to: {args.model_name}.jsonl")

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
