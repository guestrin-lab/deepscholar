"""Ground-truth nugget folder i must hold the nuggets of dataset row i."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "validate_gt_nuggets", ROOT / "scripts" / "validate_gt_nuggets.py"
)
validate_gt_nuggets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validate_gt_nuggets)


def test_gt_nugget_folders_match_dataset_rows():
    errors = validate_gt_nuggets.validate(
        str(ROOT / "dataset" / "papers_with_related_works.csv"),
        str(ROOT / "dataset" / "gt_nuggets_outputs"),
    )
    assert errors == [], "\n".join(errors)
