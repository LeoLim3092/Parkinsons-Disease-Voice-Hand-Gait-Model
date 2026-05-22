#!/usr/bin/env python
"""Train and apply calibration-only models for Parkinson's probabilities.

This module keeps only the Platt-scaling part of the workflow:

1. Fit one calibrator per model output (`gait`, `voice`, `tapping`, `ensemble`)
2. Save the fitted calibration parameters to JSON
3. Apply the saved calibrators to each new prediction

Unlike the larger workflow, this file does not do prior correction or
deployment-threshold selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


EPS = 1e-6
MODEL_COLUMNS = ["gait", "voice", "tapping", "ensemble"]
PUBLIC_INPUT_ORDER = ["gait", "voice", "hand", "ensemble"]
INPUT_TO_MODEL = {
    "gait": "gait",
    "voice": "voice",
    "hand": "tapping",
    "tapping": "tapping",
    "ensemble": "ensemble",
}


def to_probability(values: pd.Series | np.ndarray | Sequence[float]) -> np.ndarray:
    """Normalize values to 0-1 probabilities.

    The source files often store percentages in 0-100 scale, while inference
    may already be 0-1. This helper supports both.
    """

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    if float(np.nanmax(array)) > 1.0 + EPS:
        array = array / 100.0
    return np.clip(array, EPS, 1.0 - EPS)


def logit(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(probabilities, EPS, 1.0 - EPS)
    return np.log(probabilities / (1.0 - probabilities))


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def detect_id_column(df: pd.DataFrame) -> str:
    for column in df.columns:
        if "app" in str(column).strip().lower():
            return column
    return str(df.columns[0])


def load_workbook(path: Path, label: int, sheet_name: str | int = 0) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name)
    id_column = detect_id_column(df)
    required_columns = [id_column] + MODEL_COLUMNS
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{path.name} is missing required columns: {', '.join(missing_columns)}"
        )

    df = df[required_columns].copy()
    df = df.rename(columns={id_column: "record_id"})
    for column in MODEL_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=MODEL_COLUMNS).copy()
    df["record_id"] = df["record_id"].fillna("").astype(str).str.strip()
    df["label"] = int(label)
    return df


def load_previous_test_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = ["PD"] + MODEL_COLUMNS
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Previous test CSV is missing required columns: {', '.join(missing_columns)}"
        )

    df = df.copy()
    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=required_columns).copy()
    if "Unnamed: 0" in df.columns:
        df["record_id"] = df["Unnamed: 0"].astype(str).str.strip()
    else:
        df["record_id"] = np.arange(len(df)).astype(str)
    df["label"] = df["PD"].astype(int)
    return df[["record_id"] + MODEL_COLUMNS + ["label"]]


def build_calibration_dataset(
    pd_file: str | Path,
    elder_file: str | Path,
    previous_test_file: str | Path | None = None,
    sheet_name: str | int = 0,
) -> pd.DataFrame:
    """Build a labeled dataset for calibration training."""

    pd_df = load_workbook(Path(pd_file), label=1, sheet_name=sheet_name)
    elder_df = load_workbook(Path(elder_file), label=0, sheet_name=sheet_name)
    frames = [pd_df, elder_df]

    if previous_test_file:
        previous_path = Path(previous_test_file)
        if previous_path.exists():
            frames.append(load_previous_test_csv(previous_path))

    return pd.concat(frames, ignore_index=True)


def fit_platt_calibrator(probabilities: Sequence[float], labels: Sequence[int]) -> Dict[str, float]:
    """Fit a single Platt scaler on one model output."""

    x = logit(to_probability(probabilities)).reshape(-1, 1)
    y = np.asarray(labels, dtype=int)

    model = LogisticRegression(max_iter=1000)
    model.fit(x, y)

    return {
        "coef": float(model.coef_[0][0]),
        "intercept": float(model.intercept_[0]),
    }


def apply_platt_calibrator(
    probabilities: Sequence[float] | float, calibrator: Mapping[str, float]
) -> np.ndarray:
    """Apply a saved Platt scaler to one or more probabilities."""

    probs = np.atleast_1d(to_probability(probabilities))
    logits = logit(probs)
    calibrated_logits = (float(calibrator["coef"]) * logits) + float(calibrator["intercept"])
    return np.clip(sigmoid(calibrated_logits), EPS, 1.0 - EPS)


def train_calibration_bundle(
    pd_file: str | Path,
    elder_file: str | Path,
    previous_test_file: str | Path | None = None,
    sheet_name: str | int = 0,
) -> Dict[str, object]:
    """Train and return all calibration parameters as a serializable bundle.

    For reproducibility, previous-test data is included only when explicitly
    passed in.
    """

    dataset = build_calibration_dataset(
        pd_file=pd_file,
        elder_file=elder_file,
        previous_test_file=previous_test_file,
        sheet_name=sheet_name,
    )

    bundle: Dict[str, object] = {
        "metadata": {
            "pd_file": str(pd_file),
            "elder_file": str(elder_file),
            "previous_test_file": str(previous_test_file) if previous_test_file else None,
            "sheet_name": sheet_name,
            "row_count": int(len(dataset)),
            "models": MODEL_COLUMNS,
        },
        "calibrators": {},
    }

    for model_name in MODEL_COLUMNS:
        bundle["calibrators"][model_name] = fit_platt_calibrator(
            probabilities=dataset[model_name].to_numpy(),
            labels=dataset["label"].to_numpy(),
        )

    return bundle


def save_calibration_bundle(bundle: Mapping[str, object], output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(bundle, indent=2), encoding="utf-8")


def load_calibration_bundle(path: str | Path) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalize_new_prediction_input(
    new_predictions: Mapping[str, float] | Sequence[float],
) -> Dict[str, float]:
    """Accept either a dict or [gait, voice, hand, ensemble] list/tuple."""

    if isinstance(new_predictions, Mapping):
        normalized: Dict[str, float] = {}
        for input_name, value in new_predictions.items():
            key = str(input_name).strip().lower()
            if key not in INPUT_TO_MODEL:
                raise ValueError(
                    f"Unsupported prediction key '{input_name}'. "
                    f"Use one of: {', '.join(PUBLIC_INPUT_ORDER)}"
                )
            normalized[INPUT_TO_MODEL[key]] = float(value)
        return normalized

    if len(new_predictions) != 4:
        raise ValueError(
            "Sequence input must be [gait, voice, hand, ensemble] in exactly that order."
        )

    gait, voice, hand, ensemble = [float(value) for value in new_predictions]
    return {
        "gait": gait,
        "voice": voice,
        "tapping": hand,
        "ensemble": ensemble,
    }


def calibrate_new_predictions(
    new_predictions: Mapping[str, float] | Sequence[float],
    calibration_bundle: Mapping[str, object] | str | Path,
) -> Dict[str, float]:
    """Calibrate a single new prediction.

    Input:
        - dict like {"gait": 0.61, "voice": 0.44, "hand": 0.55, "ensemble": 0.58}
        - or list/tuple in this order: [gait, voice, hand, ensemble]

    Output:
        - calibrated probabilities in 0-1 scale
    """

    if isinstance(calibration_bundle, (str, Path)):
        calibration_bundle = load_calibration_bundle(calibration_bundle)

    calibrators = calibration_bundle["calibrators"]
    raw_predictions = _normalize_new_prediction_input(new_predictions)

    calibrated = {
        "gait": float(apply_platt_calibrator(raw_predictions["gait"], calibrators["gait"])[0]),
        "voice": float(
            apply_platt_calibrator(raw_predictions["voice"], calibrators["voice"])[0]
        ),
        "hand": float(
            apply_platt_calibrator(raw_predictions["tapping"], calibrators["tapping"])[0]
        ),
        "ensemble": float(
            apply_platt_calibrator(raw_predictions["ensemble"], calibrators["ensemble"])[0]
        ),
    }
    return calibrated


def calibrate_batch(
    rows: Sequence[Mapping[str, float] | Sequence[float]],
    calibration_bundle: Mapping[str, object] | str | Path,
) -> List[Dict[str, float]]:
    """Calibrate a batch of new predictions."""

    return [calibrate_new_predictions(row, calibration_bundle) for row in rows]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibration-only workflow for PD predictions.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="Fit and save calibration parameters."
    )
    train_parser.add_argument("--pd-file", required=True, help="PD workbook path.")
    train_parser.add_argument("--elder-file", required=True, help="Healthy elder workbook path.")
    train_parser.add_argument(
        "--previous-test-file",
        default=None,
        help="Optional previous labeled probability CSV. Omit to avoid using it.",
    )
    train_parser.add_argument(
        "--sheet-name", default=0, help="Sheet name or index for both workbooks."
    )
    train_parser.add_argument(
        "--output",
        default="pd_calibrators.json",
        help="Where to save the calibration bundle.",
    )

    apply_parser = subparsers.add_parser(
        "apply", help="Apply saved calibrators to one new prediction."
    )
    apply_parser.add_argument(
        "--calibrators",
        required=True,
        help="Path to the saved calibration JSON.",
    )
    apply_parser.add_argument("--gait", type=float, required=True)
    apply_parser.add_argument("--voice", type=float, required=True)
    apply_parser.add_argument(
        "--hand",
        type=float,
        required=True,
        help="Hand/tapping raw prediction.",
    )
    apply_parser.add_argument("--ensemble", type=float, required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "train":
        bundle = train_calibration_bundle(
            pd_file=args.pd_file,
            elder_file=args.elder_file,
            previous_test_file=args.previous_test_file,
            sheet_name=args.sheet_name,
        )
        save_calibration_bundle(bundle, args.output)
        print(f"Saved calibration bundle to: {Path(args.output).resolve()}")
        return

    if args.command == "apply":
        result = calibrate_new_predictions(
            {
                "gait": args.gait,
                "voice": args.voice,
                "hand": args.hand,
                "ensemble": args.ensemble,
            },
            args.calibrators,
        )
        print(json.dumps(result, indent=2))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
