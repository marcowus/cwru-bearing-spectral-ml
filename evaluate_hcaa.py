#!/usr/bin/env python3
"""Hybrid CNN + multimodal LLM arbitration pipeline.

This script reproduces the plan discussed in the design conversation:
1. Load the pretrained multi-class CNN as the baseline predictor.
2. Build deterministic validation/test splits from the processed spectrogram
   windows so that every split keeps the same tensor layout that the CNN
   expects (N x win_width).
3. For each sample, extract lightweight statistical descriptors and (when not
   running in ``--dry-run`` mode) render a 2x2 diagnostic panel that will be
   shared with the multimodal LLM.
4. Query SiliconFlow's OpenAI-compatible endpoint with the baseline CNN's top
   guess, the structured features and the optional image to obtain a calibrated
   probability distribution for the same four bearing classes.
5. Fuse both sources with a simple product-of-experts head followed by
   temperature scaling (HCAA). The calibration parameters are fitted on the
   validation split by minimising log-loss.
6. Report Accuracy / NLL / ECE / AURC on the held-out test split for the CNN,
   the LLM, their naïve average and the calibrated HCAA fusion.

Run ``python evaluate_hcaa.py --help`` for configuration options.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss

# ``matplotlib`` and ``pandas`` are optional at import time. The heavy work only
# happens in helper functions, so we guard those imports where needed to keep
# the startup time small.

# Repository-local imports
from models import ClassifierCNN_multi_class

FAULT_CLASSES: Tuple[str, ...] = (
    "normal",
    "ball_fault",
    "inner_race_fault",
    "outer_race_fault",
)
LABEL_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(FAULT_CLASSES)}

MIN_NORM = -90.0
MAX_NORM = 46.0


def _infer_label_from_filename(filename: str) -> str:
    stem = os.path.basename(filename)
    if stem.startswith("Time_Normal"):
        return "normal"
    if stem.startswith("B"):
        return "ball_fault"
    if stem.startswith("IR"):
        return "inner_race_fault"
    if stem.startswith("OR"):
        return "outer_race_fault"
    raise ValueError(f"Unable to infer label from file name: {filename}")


def _normalise(array: np.ndarray) -> np.ndarray:
    # Match the preprocessing applied during training.
    scaled = (array - MIN_NORM) / (MAX_NORM - MIN_NORM)
    scaled = (scaled - 0.5) * 2.0
    return scaled.astype(np.float32, copy=False)


@dataclass
class Sample:
    data: np.ndarray  # shape: (segment_length, win_width)
    label_idx: int
    label_name: str
    source_file: str
    segment_index: int


def load_split_samples(
    directory: str,
    segment_length: int,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[Sample], List[Sample]]:
    """Load spectrogram windows and deterministically split them.

    The processed arrays are (num_windows, win_width). We stack ``segment_length``
    consecutive rows to re-create the training-time tensor blocks and then carve
    them into disjoint validation / test subsets.
    """

    val_samples: List[Sample] = []
    test_samples: List[Sample] = []

    files = sorted(
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    )
    if not files:
        raise FileNotFoundError(f"No .npy files found under {directory!r}.")

    for filename in files:
        path = os.path.join(directory, filename)
        label_name = _infer_label_from_filename(filename)
        label_idx = LABEL_TO_INDEX[label_name]

        array = np.load(path)
        array = _normalise(array)

        num_segments = array.shape[0] // segment_length
        if num_segments <= 1:
            continue

        val_count = max(1, int(num_segments * val_ratio))
        test_count = max(1, int(num_segments * test_ratio))
        if val_count + test_count > num_segments:
            overflow = val_count + test_count - num_segments
            # Prefer shrinking the test set to keep at least one validation block.
            if test_count > overflow:
                test_count -= overflow
            else:
                val_count = max(1, val_count - overflow)
                test_count = 0

        for split, count, offset in ((val_samples, val_count, 0), (test_samples, test_count, val_count)):
            for idx in range(count):
                global_idx = offset + idx
                if global_idx >= num_segments:
                    break
                start = global_idx * segment_length
                end = start + segment_length
                block = array[start:end]
                split.append(
                    Sample(
                        data=block,
                        label_idx=label_idx,
                        label_name=label_name,
                        source_file=filename,
                        segment_index=global_idx,
                    )
                )

    if not val_samples or not test_samples:
        raise RuntimeError(
            "Validation or test split is empty. Adjust segment length / ratios so "
            "that every class contributes at least one block."
        )

    return val_samples, test_samples


def extract_features(sample: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute lightweight descriptive statistics for the LLM."""

    flat = sample.flatten()
    rms = float(np.sqrt(np.mean(np.square(flat))))
    row_rms = np.sqrt(np.mean(np.square(sample), axis=1))
    col_rms = np.sqrt(np.mean(np.square(sample), axis=0))
    col_energy = float(np.sum(col_rms)) + 1e-9

    bands = {}
    splits = np.array_split(col_rms, 4)
    for i, chunk in enumerate(splits, start=1):
        bands[f"band_{i}_energy_ratio"] = float(np.sum(chunk) / col_energy)

    percentile_values = np.percentile(flat, [5, 25, 50, 75, 95])

    return {
        "global": {
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "rms": rms,
            "energy": float(np.sum(np.square(flat))),
            "p05": float(percentile_values[0]),
            "p25": float(percentile_values[1]),
            "p50": float(percentile_values[2]),
            "p75": float(percentile_values[3]),
            "p95": float(percentile_values[4]),
        },
        "row_profile": {
            "mean_rms": float(np.mean(row_rms)),
            "std_rms": float(np.std(row_rms)),
            "max_rms": float(np.max(row_rms)),
            "dominant_row_index": int(np.argmax(row_rms)),
        },
        "column_profile": {
            "mean_rms": float(np.mean(col_rms)),
            "std_rms": float(np.std(col_rms)),
            "max_rms": float(np.max(col_rms)),
            "dominant_bin_index": int(np.argmax(col_rms)),
            **bands,
        },
    }


def create_diagnostic_panel(sample: np.ndarray, title: str) -> str:
    """Render a 2x2 diagnostic panel and return it as a base64 PNG string."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_rms = np.sqrt(np.mean(np.square(sample), axis=1))
    col_rms = np.sqrt(np.mean(np.square(sample), axis=0))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=110, constrained_layout=False)
    fig.suptitle(title)
    fig.subplots_adjust(top=0.92)

    ax = axes[0, 0]
    im = ax.imshow(sample, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title("Spectrogram window")
    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("Frame index")
    fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    ax.plot(row_rms)
    ax.set_title("Row RMS profile")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("RMS amplitude")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(col_rms)
    ax.set_title("Column RMS profile")
    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("RMS amplitude")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist(sample.flatten(), bins=30, color="tab:blue", alpha=0.8)
    ax.set_title("Value distribution")
    ax.set_xlabel("Normalised magnitude")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


class SiliconFlowLLMArbiter:
    """Wrapper around SiliconFlow's OpenAI-compatible multimodal endpoint."""

    def __init__(self, api_key: str | None, model_name: str, dry_run: bool = False):
        self.api_key = api_key
        self.model_name = model_name
        self.dry_run = dry_run or not api_key

        if self.dry_run:
            self.client = None
        else:
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RuntimeError(
                    "openai package is required for live SiliconFlow calls. Install it with 'pip install openai'."
                ) from exc

            self.client = OpenAI(base_url="https://api.siliconflow.cn/v1", api_key=api_key)

    def score_sample(
        self,
        features: Dict[str, Dict[str, float]],
        baseline_probs: Sequence[float],
        image_base64: str | None,
    ) -> np.ndarray:
        if self.dry_run or not self.client:
            return np.full(len(FAULT_CLASSES), 1.0 / len(FAULT_CLASSES), dtype=np.float32)

        feature_summary = json.dumps(features, indent=2)
        baseline_top = FAULT_CLASSES[int(np.argmax(baseline_probs))]

        prompt_header = (
            "You are a rotating machinery fault diagnosis expert. Treat this as a NEW, stand-alone session. "
            "Ignore any prior conversation.\n\n"
            "## Task: Cognitive Arbitration\n"
            f"- Baseline CNN suggestion: '{baseline_top.replace('_', ' ').title()}'\n"
            "- Use the structured statistics (and image if provided) to output a probability distribution over the fault classes.\n\n"
            "## Numerical features (JSON):\n"
            f"{feature_summary}\n\n"
            "## Output format (STRICT):\n"
            "Return ONLY a JSON object with a single key \"fault_probs\" whose value maps each class name to its probability."
            " Probabilities must sum to 1.0.\n\n"
            f"## Fault Class List: {json.dumps(FAULT_CLASSES)}"
        )

        if image_base64:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_header},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    ],
                }
            ]
        else:
            messages = [{"role": "user", "content": prompt_header}]

        try:
            response = self.client.chat.completions.create(  # type: ignore[union-attr]
                model=self.model_name,
                messages=messages,
                temperature=0.1,
            )
            content = response.choices[0].message.content  # type: ignore[index]
            json_str = content[content.find("{") : content.rfind("}") + 1]
            data = json.loads(json_str)
            probs_dict = data.get("fault_probs", {})
            result = np.array([float(probs_dict.get(name, 0.0)) for name in FAULT_CLASSES], dtype=np.float64)
            total = float(np.sum(result))
            if total <= 0:
                raise ValueError("Received degenerate probability vector from LLM.")
            return (result / total).astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            print(f"[LLM] Falling back to uniform distribution due to error: {exc}", file=sys.stderr)
            return np.full(len(FAULT_CLASSES), 1.0 / len(FAULT_CLASSES), dtype=np.float32)


class HCAAModel:
    def __init__(self) -> None:
        self.alpha = 1.0
        self.beta = 1.0
        self.temperature = 1.0

    @staticmethod
    def _poe(bn: np.ndarray, llm: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        eps = 1e-9
        fused = np.power(np.clip(bn, eps, 1.0), alpha) * np.power(np.clip(llm, eps, 1.0), beta)
        fused /= np.sum(fused, axis=1, keepdims=True)
        return fused

    @staticmethod
    def _temperature_scale(probs: np.ndarray, temperature: float) -> np.ndarray:
        eps = 1e-9
        logits = np.log(np.clip(probs, eps, 1.0))
        scaled = np.exp(logits / max(temperature, eps))
        scaled /= np.sum(scaled, axis=1, keepdims=True)
        return scaled

    def fit(self, bn: np.ndarray, llm: np.ndarray, labels: np.ndarray) -> None:
        from scipy.optimize import minimize

        def objective(params: np.ndarray) -> float:
            alpha, beta, temperature = params
            fused = self._poe(bn, llm, alpha, beta)
            calibrated = self._temperature_scale(fused, temperature)
            return log_loss(labels, calibrated, labels=range(len(FAULT_CLASSES)))

        result = minimize(
            objective,
            x0=np.array([1.0, 1.0, 1.0], dtype=np.float64),
            method="L-BFGS-B",
            bounds=[(0.1, 5.0), (0.1, 5.0), (0.2, 5.0)],
        )
        self.alpha, self.beta, self.temperature = result.x

    def predict(self, bn: np.ndarray, llm: np.ndarray, calibrated: bool = True) -> np.ndarray:
        fused = self._poe(bn, llm, self.alpha, self.beta)
        return self._temperature_scale(fused, self.temperature) if calibrated else fused


def calculate_ece(labels: np.ndarray, probs: np.ndarray, num_bins: int = 15) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    ece = 0.0
    bin_edges = np.linspace(0, 1, num_bins + 1)
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > start) & (confidences <= end)
        if not np.any(mask):
            continue
        bin_conf = float(np.mean(confidences[mask]))
        bin_acc = float(np.mean(accuracies[mask]))
        bin_prob_mass = float(np.mean(mask))
        ece += abs(bin_conf - bin_acc) * bin_prob_mass
    return float(ece)


def calculate_aurc(labels: np.ndarray, probs: np.ndarray) -> float:
    confidences = np.max(probs, axis=1)
    order = np.argsort(-confidences)
    sorted_labels = labels[order]
    sorted_preds = np.argmax(probs[order], axis=1)
    errors = (sorted_labels != sorted_preds).astype(np.float64)
    cumulative_risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    coverage = np.arange(1, len(errors) + 1) / len(errors)
    return float(np.trapezoid(1 - cumulative_risk, coverage))


def run_baseline_predictions(
    model: ClassifierCNN_multi_class,
    samples: Sequence[Sample],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for sample in samples:
            tensor = torch.from_numpy(sample.data).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            outputs.append(probs)
    return np.vstack(outputs)


def run_llm_predictions(
    arbiter: SiliconFlowLLMArbiter,
    samples: Sequence[Sample],
    baseline_probs: np.ndarray,
    skip_images: bool,
) -> np.ndarray:
    predictions: List[np.ndarray] = []
    for idx, sample in enumerate(samples):
        features = extract_features(sample.data)
        image_b64 = None if skip_images else create_diagnostic_panel(
            sample.data, f"{sample.label_name} | segment {sample.segment_index}"
        )
        pred = arbiter.score_sample(features, baseline_probs[idx], image_b64)
        predictions.append(pred)
    return np.vstack(predictions)


def tabulate_results(results: Dict[str, Dict[str, float]]) -> str:
    import pandas as pd

    df = pd.DataFrame.from_dict(results, orient="index")
    df = df[["Accuracy", "NLL", "ECE", "AURC"]]
    return df.to_string(float_format=lambda x: f"{x:.3f}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the HCAA multimodal arbiter.")
    parser.add_argument("--data-root", default="processed", help="Root folder containing the processed spectrogram windows.")
    parser.add_argument("--win-width", type=int, default=524, help="Spectral window width used during preprocessing.")
    parser.add_argument("--slide", type=int, default=128, help="Slide used in the preprocessing directory name.")
    parser.add_argument("--segment-length", type=int, default=32, help="Number of frames per CNN sample (N).")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of segments per file allocated to validation.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Fraction of segments per file allocated to the test set.")
    parser.add_argument("--model-path", default="models/model.pth", help="Path to the trained CNN weights (state_dict).")
    parser.add_argument("--device", default="cpu", help="Torch device for inference (default: cpu).")
    parser.add_argument("--siliconflow-model", default="Qwen/Qwen2.5-VL-32B-Instruct", help="Multimodal model identifier.")
    parser.add_argument("--dry-run", action="store_true", help="Skip SiliconFlow calls and use uniform LLM predictions.")
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Do not generate diagnostic panels (useful together with --dry-run).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the aggregated metrics as JSON.",
    )

    args = parser.parse_args(argv)

    dataset_dir = os.path.join(args.data_root, f"{args.win_width}_{args.slide}")
    print(f"Loading samples from {dataset_dir} ...")
    val_samples, test_samples = load_split_samples(
        dataset_dir, args.segment_length, args.val_ratio, args.test_ratio
    )
    print(f"Validation samples: {len(val_samples)}, Test samples: {len(test_samples)}")

    device = torch.device(args.device)
    model = ClassifierCNN_multi_class(N=args.segment_length, win_width=args.win_width)
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)

    print("Running baseline CNN ...")
    val_bn = run_baseline_predictions(model, val_samples, device)
    test_bn = run_baseline_predictions(model, test_samples, device)

    api_key = os.getenv("SILICON_FLOW_API_KEY")
    if not api_key and not args.dry_run:
        raise RuntimeError(
            "SILICON_FLOW_API_KEY environment variable is not set. Use --dry-run to bypass LLM calls."
        )

    arbiter = SiliconFlowLLMArbiter(api_key, args.siliconflow_model, dry_run=args.dry_run)

    print("Running multimodal LLM arbitration ...")
    val_llm = run_llm_predictions(arbiter, val_samples, val_bn, skip_images=args.skip_images)
    test_llm = run_llm_predictions(arbiter, test_samples, test_bn, skip_images=args.skip_images)

    val_labels = np.array([sample.label_idx for sample in val_samples], dtype=np.int32)
    test_labels = np.array([sample.label_idx for sample in test_samples], dtype=np.int32)

    print("Fitting HCAA fusion on validation set ...")
    hcaa = HCAAModel()
    hcaa.fit(val_bn, val_llm, val_labels)

    print("Evaluating on test set ...")
    models = {
        "CNN": test_bn,
        "LLM": test_llm,
        "Average": 0.5 * (test_bn + test_llm),
        "HCAA (uncalibrated)": hcaa.predict(test_bn, test_llm, calibrated=False),
        "HCAA (calibrated)": hcaa.predict(test_bn, test_llm, calibrated=True),
    }

    results: Dict[str, Dict[str, float]] = {}
    for name, probs in models.items():
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        acc = accuracy_score(test_labels, np.argmax(probs, axis=1))
        nll = log_loss(test_labels, probs, labels=range(len(FAULT_CLASSES)))
        ece = calculate_ece(test_labels, probs)
        aurc = calculate_aurc(test_labels, probs)
        results[name] = {
            "Accuracy": float(acc),
            "NLL": float(nll),
            "ECE": float(ece),
            "AURC": float(aurc),
        }

    print("\nFinal performance comparison:\n")
    print(tabulate_results(results))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"Metrics saved to {args.output}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
