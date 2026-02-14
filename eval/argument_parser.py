import argparse
import os
import sys
import logging

try:
    from parsers import ParserType
    from eval.evaluator import EvaluationFunction
except ImportError:
    from .parsers import ParserType
    from .evaluator import EvaluationFunction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_ALL_EVALS = [e.value for e in EvaluationFunction]
_ALL_MODES = [m.value for m in ParserType]

_DEFAULT_CONFIG_YAML = "configs/eval.yaml"


def _load_yaml(path: str) -> dict:
    """Load a YAML file and return its contents as a dict. Returns {} on any error."""
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            print(f"Warning: {path} is not a YAML mapping; ignoring.", file=sys.stderr)
            return {}
        return data
    except Exception as e:
        print(f"Warning: Could not load config {path}: {e}", file=sys.stderr)
        return {}


def parse_args():
    """Parse command line arguments"""
    # Pre-parse to get --config-yaml before building the main parser
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-yaml", default=_DEFAULT_CONFIG_YAML)
    pre_args, _ = pre_parser.parse_known_args()

    yaml_values = _load_yaml(pre_args.config_yaml)

    parser = argparse.ArgumentParser(description="DeepScholar-Bench Evaluation Suite")

    parser.add_argument(
        "--config-yaml",
        type=str,
        default=pre_args.config_yaml,
        help=(
            f"Path to YAML config file (default: {_DEFAULT_CONFIG_YAML}). "
            "Provides base values; any explicitly-passed CLI arg overrides the YAML."
        ),
    )

    parser.add_argument(
        "--modes",
        nargs="+",
        default=["deepscholar_base"],
        help=(
            "System(s) to evaluate. "
            f"Choices: {', '.join(_ALL_MODES)}. "
            "One input folder must be provided per mode."
        ),
    )

    parser.add_argument(
        "--evals",
        nargs="+",
        default=_ALL_EVALS,
        help=(
            "Evaluation metric(s) to run. "
            f"Choices: {', '.join(_ALL_EVALS)}, all. "
            "Pass 'all' to run every metric."
        ),
    )

    parser.add_argument(
        "--file-id",
        dest="file_id",
        nargs="+",
        type=str,
        help="Specific file IDs to process. If not provided, processes all files in the input folder.",
    )

    parser.add_argument(
        "--input-folder",
        dest="input_folder",
        nargs="+",
        type=str,
        default=None,
        help="Input folder(s) containing system outputs. One folder per mode.",
    )

    parser.add_argument(
        "--output-folder",
        dest="output_folder",
        type=str,
        default="results",
        help="Output directory for evaluation results (default: results)",
    )

    parser.add_argument(
        "--dataset-path",
        dest="dataset_path",
        type=str,
        default="dataset/papers_with_related_works.csv",
        help="Path to the dataset CSV file",
    )

    parser.add_argument(
        "--model-name", "--model",
        dest="model_name",
        type=str,
        default="gpt-4o",
        help="Model to use for LLM-based evaluations (default: gpt-4o)",
    )

    parser.add_argument(
        "--reference-folder",
        dest="reference_folder",
        type=str,
        default="test/baselines_results/openscholar",
        help="Reference folder for ground-truth file IDs",
    )

    parser.add_argument(
        "--important-citations-path",
        dest="important_citations_path",
        type=str,
        default="dataset/important_citations.csv",
        help="Path to the important citations CSV file",
    )

    parser.add_argument(
        "--nugget-groundtruth-dir-path",
        dest="nugget_groundtruth_dir_path",
        type=str,
        default="dataset/gt_nuggets_outputs",
        help="Path to the nugget ground-truth directory",
    )

    # Apply YAML as defaults so CLI args naturally override them
    if yaml_values:
        parser.set_defaults(**yaml_values)

    args = parser.parse_args()

    # Validate input_folder (not required in parser because it can come from YAML)
    if not args.input_folder:
        parser.error("--input-folder is required (or set 'input_folder' in the config YAML)")

    # Expand "all" in --evals to every evaluation function
    if isinstance(args.evals, list) and "all" in args.evals:
        args.evals = _ALL_EVALS

    # Validate and convert evals
    invalid_evals = [e for e in args.evals if e not in _ALL_EVALS]
    if invalid_evals:
        parser.error(f"Unknown eval(s): {invalid_evals}. Choices: {_ALL_EVALS + ['all']}")
    args.evals = [EvaluationFunction(e) for e in args.evals]

    # Validate and convert modes
    invalid_modes = [m for m in args.modes if m not in _ALL_MODES]
    if invalid_modes:
        parser.error(f"Unknown mode(s): {invalid_modes}. Choices: {_ALL_MODES}")
    args.modes = [ParserType(m) for m in args.modes]

    if len(args.input_folder) != len(args.modes):
        parser.error(
            f"Number of input folders ({len(args.input_folder)}) "
            f"must match number of modes ({len(args.modes)})"
        )

    logger.info(f"Processing modes: {args.modes}")
    logger.info(f"Input folder: {args.input_folder}")
    logger.info(f"Output directory: {args.output_folder}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Evals: {args.evals}")

    if args.file_id:
        logger.info(f"Processing specific file IDs: {args.file_id}")
    else:
        logger.info("Processing all files in input folders")

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, "tokens"), exist_ok=True)

    return args
