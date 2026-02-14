#!/usr/bin/env bash
#
# Master script to run the complete pipeline:
#   1. Data collection   -> python -m data_pipeline.main
#   2. Report generation -> python -m deepscholar_base.main
#   3. Evaluation        -> python -m eval.main
#
# Usage: ./run_all.sh [OPTIONS]

set -e  # Exit on error

# ==============================================================
# Default values — empty string means "not passed" (use YAML or
# module default). Booleans that are tri-state use "true"/"false"/"".
# ==============================================================

# --- Data Pipeline ---
FIELD=""                      # Named field shorthand (cs|bio|econ|phy|stat); repeat for multiple: "cs bio"
CATEGORIES=""                 # Explicit ArXiv categories (space-separated)
START_DATE=""
END_DATE=""
PAPER_ID=""
EXISTING_PAPERS_CSV=""
MIN_HINDEX=""                 # default in module/YAML: 20
MAX_HINDEX=""
MAX_PAPERS_PER_CATEGORY=""    # default in module/YAML: 1000
MIN_CITATIONS=""              # default in module/YAML: 5
NO_SAVE_PAPERS=false
NO_SAVE_CONTENT=false
NO_SAVE_CITATIONS=false
CONCURRENT_REQUESTS=""
REQUEST_DELAY=""
DATA_PIPELINE_CONFIG=""       # override default configs/data_pipeline.yaml

# --- DeepScholar-Base ---
MODEL=""
START_IDX=""
END_IDX=""
SEARCH_MODE=""                # "agentic" or "recursive"; empty = use YAML/default
USE_STRUCTURED_OUTPUT=""      # "true" / "false" / "" (not set)
ENABLE_WEB_SEARCH=""          # "true" / "false" / ""
USE_RESPONSES_MODEL=false     # agentic-only flag
MAX_SEARCH_RETRIES=""
PER_QUERY_MAX_SEARCH_RESULTS=""
NUM_SEARCH_STEPS=""           # recursive-only
NUM_SEARCH_QUERIES_PER_STEP="" # recursive-only
WEB_CORPUSES=""               # space-separated: "TAVILY ARXIV"
USE_SEM_FILTER=""             # "true" / "false" / ""
USE_SEM_TOPK=""               # "true" / "false" / ""
FINAL_MAX_RESULTS=""
CATEGORIZE_REFERENCES=""      # "true" / "false" / ""
GENERATE_CATEGORY_SUMMARY=""  # "true" / "false" / ""
GENERATE_INSIGHTS=""          # "true" / "false" / ""
DEEPSCHOLAR_CONFIG=""         # override default configs/deepscholar_base.yaml

# --- Evaluation ---
EVALS="all"
EVAL_MODES="deepscholar_base" # space-separated list of systems to evaluate
FILE_ID=""                    # specific file IDs to evaluate (space-separated)
REFERENCE_FOLDER=""
IMPORTANT_CITATIONS_PATH=""
NUGGET_GROUNDTRUTH_DIR_PATH=""
EVAL_MODEL=""                 # model for LLM-judge; empty = use YAML/default
EVAL_CONFIG=""                # override default configs/eval.yaml

# --- General ---
OUTPUT_BASE=""
DATASET_DIR="./dataset"   # default; override with --dataset-dir
RESULTS_DIR=""            # derived from OUTPUT_BASE if not set
EVAL_DIR=""               # derived from OUTPUT_BASE if not set
SCRAPE_NEW_DATA=false   # Data pipeline is skipped by default; pass --scrape-new-data to enable
SKIP_EVAL=false

# ==============================================================
# Usage
# ==============================================================
usage() {
  cat << EOF
Usage: $0 [OPTIONS]

Runs the complete DeepScholar-Bench pipeline:
  1. Data collection   -> python -m data_pipeline.main
  2. Report generation -> python -m deepscholar_base.main
  3. Evaluation        -> python -m eval.main

Step 2 (report generation) cannot be skipped.
If step 1 is skipped, --output-base must point to an existing run directory
that contains a dataset/ sub-folder.

=== Data Pipeline Options ===
  --field FIELD [FIELD ...]        Named field(s): cs, bio, econ, phy, stat
                                   (repeat or space-separate for multiple)
  --categories CAT [CAT ...]       Explicit ArXiv categories (e.g. cs.AI cs.LG)
  --start-date DATE                Start date YYYY-MM-DD
  --end-date DATE                  End date YYYY-MM-DD
  --paper-id ID                    Single-paper mode (ArXiv ID)
  --existing-papers-csv PATH       Skip scraping; use existing papers CSV
  --min-hindex N                   Min author h-index (default from YAML: 20)
  --max-hindex N                   Max author h-index
  --max-papers-per-category N      Max papers per category (default from YAML: 1000)
  --min-citations N                Min citations in related works (default from YAML: 5)
  --no-save-papers                 Don't save raw papers CSV
  --no-save-content                Don't save paper content CSV
  --no-save-citations              Don't save citations CSV
  --concurrent-requests N          Concurrent API requests
  --request-delay F                Delay between requests (seconds)
  --data-pipeline-config FILE      YAML config for data pipeline
                                   (default: configs/data_pipeline.yaml)

=== DeepScholar-Base Options ===
  --model MODEL                    Model name (e.g. gpt-4o)
  --start-idx N                    Start index of queries (default: 0)
  --end-idx N                      End index of queries (default: all)
  --search-mode MODE               "agentic" or "recursive"
  --max-search-retries N           Max search retries
  --enable-web-search              Enable web search
  --no-web-search                  Disable web search
  --use-responses-model            Use OpenAI responses API (agentic only)
  --per-query-max-search-results N Max results per query
  --num-search-steps N             Steps for recursive search
  --num-search-queries-per-step N  Queries per step (recursive)
  --web-corpuses CORPUS [...]      Web corpuses (e.g. TAVILY ARXIV)
  --use-sem-filter                 Enable semantic filter
  --no-sem-filter                  Disable semantic filter
  --use-sem-topk                   Enable semantic top-k
  --no-sem-topk                    Disable semantic top-k
  --final-max-results N            Final max results count
  --use-structured-output          Enable structured output
  --no-structured-output           Disable structured output
  --categorize-references          Enable reference categorization
  --no-categorize-references       Disable reference categorization
  --generate-category-summary      Enable category summary generation
  --no-generate-category-summary   Disable category summary generation
  --generate-insights              Enable insights generation
  --no-generate-insights           Disable insights generation
  --deepscholar-config FILE        YAML config for deepscholar_base
                                   (default: configs/deepscholar_base.yaml)

=== Evaluation Options ===
  --evals EVAL [...]               Metrics to run (default: all)
                                   Options: nugget_coverage, reference_coverage,
                                            organization, cite_p, all
  --eval-modes MODE [...]          Systems to evaluate (default: deepscholar_base)
  --eval-model MODEL               Judge model (e.g. gpt-4o)
  --file-id ID [...]               Specific file IDs to evaluate
  --reference-folder PATH          Ground-truth reference folder
  --important-citations-path PATH  Path to important citations CSV
  --nugget-groundtruth-dir PATH    Path to nugget ground-truth directory
  --eval-config FILE               YAML config for evaluation
                                   (default: configs/eval.yaml)

=== General Options ===
  --output-base DIR                Base output directory (default: runs/FIELD_TIMESTAMP)
  --dataset-dir DIR                Dataset directory (default: ./dataset)
  --results-dir DIR                Results directory (default: OUTPUT_BASE/results)
  --eval-dir DIR                   Evaluation directory (default: OUTPUT_BASE/evaluation)
  --scrape-new-data                Run step 1 (data pipeline); skipped by default
  --skip-eval                      Skip step 3
  -h, --help                       Show this help

Examples:
  # Run generation + evaluation on existing dataset
  $0 --output-base runs/cs_jan2025

  # Full pipeline including data collection
  $0 --scrape-new-data --field cs --start-date 2025-01-01 --output-base runs/cs_jan2025

  # Multiple fields
  $0 --field cs --field bio --start-date 2025-01-01

  # Single paper
  $0 --paper-id 2502.07374 --output-base runs/single_paper

  # Skip collection, use existing dataset
  $0 --skip-data-pipeline --output-base runs/cs_jan2025

  # Run with custom configs
  $0 --field cs --start-date 2025-01-01 \\
     --data-pipeline-config my_pipeline.yaml \\
     --deepscholar-config my_model.yaml \\
     --eval-config my_eval.yaml

EOF
}

# ==============================================================
# Parse arguments
# ==============================================================
while [[ $# -gt 0 ]]; do
  case $1 in
    # --- Data Pipeline ---
    --field)
      FIELD="$FIELD $2"
      shift 2
      ;;
    --categories)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        CATEGORIES="$CATEGORIES $1"
        shift
      done
      CATEGORIES="${CATEGORIES# }"
      ;;
    --start-date)
      START_DATE="$2"
      shift 2
      ;;
    --end-date)
      END_DATE="$2"
      shift 2
      ;;
    --paper-id)
      PAPER_ID="$2"
      shift 2
      ;;
    --existing-papers-csv)
      EXISTING_PAPERS_CSV="$2"
      shift 2
      ;;
    --min-hindex)
      MIN_HINDEX="$2"
      shift 2
      ;;
    --max-hindex)
      MAX_HINDEX="$2"
      shift 2
      ;;
    --max-papers-per-category)
      MAX_PAPERS_PER_CATEGORY="$2"
      shift 2
      ;;
    --min-citations)
      MIN_CITATIONS="$2"
      shift 2
      ;;
    --no-save-papers)
      NO_SAVE_PAPERS=true
      shift
      ;;
    --no-save-content)
      NO_SAVE_CONTENT=true
      shift
      ;;
    --no-save-citations)
      NO_SAVE_CITATIONS=true
      shift
      ;;
    --concurrent-requests)
      CONCURRENT_REQUESTS="$2"
      shift 2
      ;;
    --request-delay)
      REQUEST_DELAY="$2"
      shift 2
      ;;
    --data-pipeline-config)
      DATA_PIPELINE_CONFIG="$2"
      shift 2
      ;;
    # --- DeepScholar-Base ---
    --model)
      MODEL="$2"
      shift 2
      ;;
    --start-idx)
      START_IDX="$2"
      shift 2
      ;;
    --end-idx)
      END_IDX="$2"
      shift 2
      ;;
    --search-mode)
      SEARCH_MODE="$2"
      shift 2
      ;;
    --max-search-retries)
      MAX_SEARCH_RETRIES="$2"
      shift 2
      ;;
    --enable-web-search)
      ENABLE_WEB_SEARCH="true"
      shift
      ;;
    --no-web-search)
      ENABLE_WEB_SEARCH="false"
      shift
      ;;
    --use-responses-model)
      USE_RESPONSES_MODEL=true
      shift
      ;;
    --per-query-max-search-results)
      PER_QUERY_MAX_SEARCH_RESULTS="$2"
      shift 2
      ;;
    --num-search-steps)
      NUM_SEARCH_STEPS="$2"
      shift 2
      ;;
    --num-search-queries-per-step)
      NUM_SEARCH_QUERIES_PER_STEP="$2"
      shift 2
      ;;
    --web-corpuses)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        WEB_CORPUSES="$WEB_CORPUSES $1"
        shift
      done
      WEB_CORPUSES="${WEB_CORPUSES# }"
      ;;
    --use-sem-filter)
      USE_SEM_FILTER="true"
      shift
      ;;
    --no-sem-filter)
      USE_SEM_FILTER="false"
      shift
      ;;
    --use-sem-topk)
      USE_SEM_TOPK="true"
      shift
      ;;
    --no-sem-topk)
      USE_SEM_TOPK="false"
      shift
      ;;
    --final-max-results)
      FINAL_MAX_RESULTS="$2"
      shift 2
      ;;
    --use-structured-output)
      USE_STRUCTURED_OUTPUT="true"
      shift
      ;;
    --no-structured-output)
      USE_STRUCTURED_OUTPUT="false"
      shift
      ;;
    --categorize-references)
      CATEGORIZE_REFERENCES="true"
      shift
      ;;
    --no-categorize-references)
      CATEGORIZE_REFERENCES="false"
      shift
      ;;
    --generate-category-summary)
      GENERATE_CATEGORY_SUMMARY="true"
      shift
      ;;
    --no-generate-category-summary)
      GENERATE_CATEGORY_SUMMARY="false"
      shift
      ;;
    --generate-insights)
      GENERATE_INSIGHTS="true"
      shift
      ;;
    --no-generate-insights)
      GENERATE_INSIGHTS="false"
      shift
      ;;
    --deepscholar-config)
      DEEPSCHOLAR_CONFIG="$2"
      shift 2
      ;;
    # --- Evaluation ---
    --evals)
      shift
      EVALS=""
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        EVALS="$EVALS $1"
        shift
      done
      EVALS="${EVALS# }"
      ;;
    --eval-modes)
      shift
      EVAL_MODES=""
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        EVAL_MODES="$EVAL_MODES $1"
        shift
      done
      EVAL_MODES="${EVAL_MODES# }"
      ;;
    --eval-model)
      EVAL_MODEL="$2"
      shift 2
      ;;
    --file-id)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        FILE_ID="$FILE_ID $1"
        shift
      done
      FILE_ID="${FILE_ID# }"
      ;;
    --reference-folder)
      REFERENCE_FOLDER="$2"
      shift 2
      ;;
    --important-citations-path)
      IMPORTANT_CITATIONS_PATH="$2"
      shift 2
      ;;
    --nugget-groundtruth-dir)
      NUGGET_GROUNDTRUTH_DIR_PATH="$2"
      shift 2
      ;;
    --eval-config)
      EVAL_CONFIG="$2"
      shift 2
      ;;
    # --- General ---
    --output-base)
      OUTPUT_BASE="$2"
      shift 2
      ;;
    --dataset-dir)
      DATASET_DIR="$2"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --eval-dir)
      EVAL_DIR="$2"
      shift 2
      ;;
    --scrape-new-data)
      SCRAPE_NEW_DATA=true
      shift
      ;;
    --skip-eval)
      SKIP_EVAL=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

# Trim leading spaces from accumulated lists
FIELD="${FIELD# }"

# ==============================================================
# Derive output directories
# ==============================================================
if [[ -z "$OUTPUT_BASE" ]]; then
  if [[ -n "$PAPER_ID" ]]; then
    OUTPUT_BASE="runs/single_paper_$(date +%Y%m%d_%H%M%S)_$PAPER_ID"
  elif [[ -n "$FIELD" ]]; then
    # Use first field name in the timestamp label
    ALL_FIELDS=$(echo "$FIELD" | tr ' ' '_')
    OUTPUT_BASE="runs/${ALL_FIELDS}_$(date +%Y%m%d_%H%M%S)"
  else
    OUTPUT_BASE="runs/custom_$(date +%Y%m%d_%H%M%S)"
  fi
fi

[[ -z "$RESULTS_DIR" ]] && RESULTS_DIR="$OUTPUT_BASE/results"
[[ -z "$EVAL_DIR" ]]    && EVAL_DIR="$OUTPUT_BASE/evaluation"

mkdir -p "$OUTPUT_BASE"

# ==============================================================
# Print configuration summary
# ==============================================================
echo "=========================================="
echo "DeepScholar-Bench Pipeline"
echo "=========================================="
echo "Output Base:           $OUTPUT_BASE"
echo "Dataset Directory:     $DATASET_DIR"
echo "Results Directory:     $RESULTS_DIR"
echo "Evaluation Directory:  $EVAL_DIR"
echo
echo "Steps:"
echo "  1. Data Pipeline:    $([ "$SCRAPE_NEW_DATA" = true ] && echo "RUN" || echo "SKIP")"
echo "  2. DeepScholar-Base: RUN"
echo "  3. Evaluation:       $([ "$SKIP_EVAL" = true ] && echo "SKIP" || echo "RUN")"
echo "=========================================="
echo

# ==============================================================
# Step 1: Data Pipeline
# ==============================================================
if [ "$SCRAPE_NEW_DATA" = true ]; then
  echo "=========================================="
  echo "Step 1/3: Running Data Pipeline"
  echo "=========================================="
  echo

  DATA_PIPELINE_CMD="python -m data_pipeline.main"
  DATA_PIPELINE_CMD="$DATA_PIPELINE_CMD --output-dir \"$DATASET_DIR\""

  # Config file
  [[ -n "$DATA_PIPELINE_CONFIG" ]] && DATA_PIPELINE_CMD="$DATA_PIPELINE_CMD --config-yaml \"$DATA_PIPELINE_CONFIG\""

  # Paper selection
  if [[ -n "$PAPER_ID" ]]; then
    DATA_PIPELINE_CMD="$DATA_PIPELINE_CMD --paper-id $PAPER_ID"
  else
    if [[ -n "$FIELD" ]]; then
      for f in $FIELD; do
        DATA_PIPELINE_CMD="$DATA_PIPELINE_CMD --field $f"
      done
    fi
    if [[ -n "$CATEGORIES" ]]; then
      DATA_PIPELINE_CMD="$DATA_PIPELINE_CMD --categories $CATEGORIES"
    fi
    [[ -n "$START_DATE" ]] && DATA_PIPELINE_CMD="$DATA_PIPELINE_CMD --start-date $START_DATE"
    [[ -n "$END_DATE" ]]   && DATA_PIPELINE_CMD="$DATA_PIPELINE_CMD --end-date $END_DATE"
  fi

  [[ -n "$EXISTING_PAPERS_CSV" ]] && DATA_PIPELINE_CMD="$DATA_PIPELINE_CMD --existing-papers-csv \"$EXISTING_PAPERS_CSV\""
  [[ -n "$MIN_HINDEX" ]]           && DATA_PIPELINE_CMD="$DATA_PIPELINE_CMD --min-hindex $MIN_HINDEX"
  [[ -n "$MAX_HINDEX" ]]           && DATA_PIPELINE_CMD="$DATA_PIPELINE_CMD --max-hindex $MAX_HINDEX"
  [[ -n "$MAX_PAPERS_PER_CATEGORY" ]] && DATA_PIPELINE_CMD="$DATA_PIPELINE_CMD --max-papers-per-category $MAX_PAPERS_PER_CATEGORY"
  [[ -n "$MIN_CITATIONS" ]]        && DATA_PIPELINE_CMD="$DATA_PIPELINE_CMD --min-citations $MIN_CITATIONS"
  $NO_SAVE_PAPERS   && DATA_PIPELINE_CMD="$DATA_PIPELINE_CMD --no-save-papers"
  $NO_SAVE_CONTENT  && DATA_PIPELINE_CMD="$DATA_PIPELINE_CMD --no-save-content"
  $NO_SAVE_CITATIONS && DATA_PIPELINE_CMD="$DATA_PIPELINE_CMD --no-save-citations"
  [[ -n "$CONCURRENT_REQUESTS" ]] && DATA_PIPELINE_CMD="$DATA_PIPELINE_CMD --concurrent-requests $CONCURRENT_REQUESTS"
  [[ -n "$REQUEST_DELAY" ]]       && DATA_PIPELINE_CMD="$DATA_PIPELINE_CMD --request-delay $REQUEST_DELAY"

  echo "Running: $DATA_PIPELINE_CMD"
  eval "$DATA_PIPELINE_CMD"

  echo
  echo "Data pipeline completed"
  echo
else
  echo "Step 1/3: Data pipeline skipped (pass --scrape-new-data to enable)"
  echo "  Using existing dataset in: $DATASET_DIR"
  echo

  if [[ ! -d "$DATASET_DIR" ]]; then
    echo "ERROR: Dataset directory does not exist: $DATASET_DIR"
    echo "       Pass --output-base pointing to an existing run, or add --scrape-new-data"
    exit 1
  fi
fi

# ==============================================================
# Step 2: DeepScholar-Base (cannot be skipped)
# ==============================================================
echo "=========================================="
echo "Step 2/3: Running DeepScholar-Base"
echo "=========================================="
echo

DEEPSCHOLAR_CMD="python -m deepscholar_base.main"
DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --queries-file \"$DATASET_DIR/queries.csv\""
DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --output-folder \"$RESULTS_DIR\""
[[ -n "$DEEPSCHOLAR_CONFIG" ]]            && DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --config-yaml \"$DEEPSCHOLAR_CONFIG\""
[[ -n "$MODEL" ]]                         && DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --model $MODEL"
[[ -n "$START_IDX" ]]                     && DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --start-idx $START_IDX"
[[ -n "$END_IDX" ]]                       && DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --end-idx $END_IDX"
[[ -n "$SEARCH_MODE" ]]                   && DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --search-mode $SEARCH_MODE"
[[ -n "$MAX_SEARCH_RETRIES" ]]            && DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --max-search-retries $MAX_SEARCH_RETRIES"
[[ -n "$PER_QUERY_MAX_SEARCH_RESULTS" ]]  && DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --per-query-max-search-results $PER_QUERY_MAX_SEARCH_RESULTS"
[[ -n "$NUM_SEARCH_STEPS" ]]              && DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --num-search-steps $NUM_SEARCH_STEPS"
[[ -n "$NUM_SEARCH_QUERIES_PER_STEP" ]]   && DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --num-search-queries-per-step $NUM_SEARCH_QUERIES_PER_STEP"
[[ -n "$FINAL_MAX_RESULTS" ]]             && DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --final-max-results $FINAL_MAX_RESULTS"
[[ -n "$WEB_CORPUSES" ]]                  && DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --web-corpuses $WEB_CORPUSES"
$USE_RESPONSES_MODEL                      && DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --use-responses-model"

# Tri-state booleans: "true" -> --flag, "false" -> --no-flag, "" -> not passed
[[ "$ENABLE_WEB_SEARCH" == "true" ]]           && DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --enable-web-search"
[[ "$ENABLE_WEB_SEARCH" == "false" ]]          && DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --no-web-search"
[[ "$USE_STRUCTURED_OUTPUT" == "true" ]]       && DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --use-structured-output"
[[ "$USE_STRUCTURED_OUTPUT" == "false" ]]      && DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --no-structured-output"
[[ "$USE_SEM_FILTER" == "true" ]]              && DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --use-sem-filter"
[[ "$USE_SEM_FILTER" == "false" ]]             && DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --no-sem-filter"
[[ "$USE_SEM_TOPK" == "true" ]]                && DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --use-sem-topk"
[[ "$USE_SEM_TOPK" == "false" ]]               && DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --no-sem-topk"
[[ "$CATEGORIZE_REFERENCES" == "true" ]]       && DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --categorize-references"
[[ "$CATEGORIZE_REFERENCES" == "false" ]]      && DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --no-categorize-references"
[[ "$GENERATE_CATEGORY_SUMMARY" == "true" ]]   && DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --generate-category-summary"
[[ "$GENERATE_CATEGORY_SUMMARY" == "false" ]]  && DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --no-generate-category-summary"
[[ "$GENERATE_INSIGHTS" == "true" ]]           && DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --generate-insights"
[[ "$GENERATE_INSIGHTS" == "false" ]]          && DEEPSCHOLAR_CMD="$DEEPSCHOLAR_CMD --no-generate-insights"

echo "Running: $DEEPSCHOLAR_CMD"
eval "$DEEPSCHOLAR_CMD"

echo
echo "DeepScholar-Base completed"
echo

# ==============================================================
# Step 3: Evaluation
# ==============================================================
if [ "$SKIP_EVAL" = false ]; then
  echo "=========================================="
  echo "Step 3/3: Running Evaluation"
  echo "=========================================="
  echo

  EVAL_CMD="python -m eval.main"
  EVAL_CMD="$EVAL_CMD --modes $EVAL_MODES"
  EVAL_CMD="$EVAL_CMD --evals $EVALS"
  EVAL_CMD="$EVAL_CMD --input-folder \"$RESULTS_DIR\""
  EVAL_CMD="$EVAL_CMD --output-folder \"$EVAL_DIR\""
  EVAL_CMD="$EVAL_CMD --dataset-path \"$DATASET_DIR/papers_with_related_works.csv\""

  [[ -n "$EVAL_CONFIG" ]]                  && EVAL_CMD="$EVAL_CMD --config-yaml \"$EVAL_CONFIG\""
  [[ -n "$EVAL_MODEL" ]]                   && EVAL_CMD="$EVAL_CMD --model $EVAL_MODEL"
  [[ -n "$FILE_ID" ]]                      && EVAL_CMD="$EVAL_CMD --file-id $FILE_ID"
  [[ -n "$REFERENCE_FOLDER" ]]             && EVAL_CMD="$EVAL_CMD --reference-folder \"$REFERENCE_FOLDER\""
  [[ -n "$IMPORTANT_CITATIONS_PATH" ]]     && EVAL_CMD="$EVAL_CMD --important-citations-path \"$IMPORTANT_CITATIONS_PATH\""
  [[ -n "$NUGGET_GROUNDTRUTH_DIR_PATH" ]]  && EVAL_CMD="$EVAL_CMD --nugget-groundtruth-dir-path \"$NUGGET_GROUNDTRUTH_DIR_PATH\""

  echo "Running: $EVAL_CMD"
  eval "$EVAL_CMD"

  echo
  echo "Evaluation completed"
  echo
else
  echo "Skipping evaluation step"
  echo
fi

# ==============================================================
# Summary
# ==============================================================
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Output Directory: $OUTPUT_BASE"
echo
echo "Generated Artifacts:"
[ "$SCRAPE_NEW_DATA" = true ] && echo "  Dataset:     $DATASET_DIR/"
echo "  Results:     $RESULTS_DIR/"
[ "$SKIP_EVAL" = false ]     && echo "  Evaluation:  $EVAL_DIR/"
echo
echo "=========================================="
