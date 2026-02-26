#!/usr/bin/env python3
"""
Metrics analysis and visualisation for the 4DGS synthetic benchmark.

Ingests per-iteration training metrics from three reconstruction methods:
  - 3DGS             (3D Gaussian Splatting)
  - 4DGS             (4D Gaussian Splatting)
  - Deformable 3DGS

and generates two families of plots:
  1. Training curves  — PSNR / SSIM / LPIPS vs iteration, one figure per scene
                        (10 figures × 3 subplots = 30 plots)
  2. Eval vs complexity — final test-set metric for each method plotted against
                          scene complexity rank (3 figures)

Expected metrics directory layout:
    metrics/
    ├── 3dgs/        scene1.json  …  scene10.json
    ├── 4dgs/        scene1.json  …  scene10.json
    └── deformable/  scene1.json  …  scene10.json

Each JSON file is either a JSON array or newline-delimited JSON objects:
    [
        {"iteration": 1000,  "psnr": 25.3, "ssim": 0.85, "lpips": 0.12},
        {"iteration": 7000,  "psnr": 28.1, "ssim": 0.91, "lpips": 0.08},
        ...
    ]

Usage:
    python scripts/metrics_analysis.py
    python scripts/metrics_analysis.py --metrics_dir my_metrics --output_dir figs
    python scripts/metrics_analysis.py --complexity_only
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Scene metadata
# ---------------------------------------------------------------------------

SCENE_NAMES: Dict[int, str] = {
    1:  "Close Proximity\n(Diff. Colors)",
    2:  "Close Proximity\n(Identical)",
    3:  "Three-Body\nCollision",
    4:  "Occlusion &\nDis-occlusion",
    5:  "Rapid Direction\nChanges",
    6:  "Extreme Scale\nChange",
    7:  "Deformable vs\nRigid Collision",
    8:  "Thin Structure\nTracking",
    9:  "Topology\nChange",
    10: "High-Freq.\nTexture + Motion",
}

METHOD_KEYS   = ["3dgs", "4dgs", "deformable"]
METHOD_LABELS = {"3dgs": "3DGS", "4dgs": "4DGS", "deformable": "Deformable 3DGS"}
METHOD_COLORS = {"3dgs": "#2196F3", "4dgs": "#4CAF50", "deformable": "#FF5722"}
METHOD_MARKERS = {"3dgs": "o", "4dgs": "s", "deformable": "^"}

METRICS       = ["psnr", "ssim", "lpips"]
METRIC_LABELS = {"psnr": "PSNR (dB)", "ssim": "SSIM", "lpips": "LPIPS"}
METRIC_BETTER = {"psnr": "higher is better", "ssim": "higher is better", "lpips": "lower is better"}


# ---------------------------------------------------------------------------
# Scene complexity scoring
# ---------------------------------------------------------------------------
# Ten factors, each scored [0, 1], combined via a weighted sum C = Σ wᵢ · fᵢ.
# Weights reflect how strongly each factor challenges 4DGS-class methods.
#
# Factor definitions
# ------------------
# motion_speed       Peak object speed relative to scene bounds.
#                    Fast motion violates temporal-continuity priors in k-planes.
# num_objects        Number of independently moving objects (normalised to [0,1]
#                    with 3+ objects = 1.0). More objects require more Gaussians.
# occlusion          Degree of complete or prolonged occlusion.
#                    Full occlusion causes hallucination and dis-occlusion artefacts.
# proximity          Closest inter-object separation normalised by object diameter.
#                    Touching/overlapping = 1.0. Gaussians from nearby objects blur.
# deformation        Presence and magnitude of non-rigid shape change.
#                    4DGS assumes near-rigid deformation fields.
# topology           Presence of topology change events (split / merge).
#                    Fixed Gaussian count cannot represent object count changes.
# thin_geometry      Aspect ratio of the thinnest object (radius / half-length).
#                    Very thin structures require many tiny anisotropic Gaussians.
# scale_variation    Ratio of apparent object size at closest vs farthest point.
#                    Extreme scale changes challenge level-of-detail representation.
# direction_changes  Number and sharpness of direction reversals.
#                    Sharp turns violate smooth motion priors in the deformation field.
# texture_complexity High-frequency surface texture under rotation/translation.
#                    Appearance and geometry become entangled in Gaussian features.

FACTOR_WEIGHTS: Dict[str, float] = {
    "motion_speed":       0.12,
    "num_objects":        0.08,
    "occlusion":          0.13,
    "proximity":          0.10,
    "deformation":        0.13,
    "topology":           0.18,
    "thin_geometry":      0.09,
    "scale_variation":    0.05,
    "direction_changes":  0.07,
    "texture_complexity": 0.05,
}
assert abs(sum(FACTOR_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1"

# fmt: off
_FACTOR_SCORES: Dict[int, Dict[str, float]] = {
    1: dict(motion_speed=0.3, num_objects=0.5, occlusion=0.0, proximity=0.9,
            deformation=0.0, topology=0.0, thin_geometry=0.0, scale_variation=0.0,
            direction_changes=0.0, texture_complexity=0.0),
    2: dict(motion_speed=0.3, num_objects=0.5, occlusion=0.0, proximity=1.0,
            deformation=0.0, topology=0.0, thin_geometry=0.0, scale_variation=0.0,
            direction_changes=0.0, texture_complexity=0.0),
    3: dict(motion_speed=0.9, num_objects=1.0, occlusion=0.4, proximity=1.0,
            deformation=0.0, topology=0.0, thin_geometry=0.0, scale_variation=0.0,
            direction_changes=0.2, texture_complexity=0.0),
    4: dict(motion_speed=0.4, num_objects=0.3, occlusion=1.0, proximity=0.1,
            deformation=0.0, topology=0.0, thin_geometry=0.2, scale_variation=0.3,
            direction_changes=0.0, texture_complexity=0.0),
    5: dict(motion_speed=0.8, num_objects=0.0, occlusion=0.0, proximity=0.0,
            deformation=0.0, topology=0.0, thin_geometry=0.0, scale_variation=0.1,
            direction_changes=1.0, texture_complexity=0.0),
    6: dict(motion_speed=0.5, num_objects=0.0, occlusion=0.0, proximity=0.0,
            deformation=0.0, topology=0.0, thin_geometry=0.0, scale_variation=1.0,
            direction_changes=0.1, texture_complexity=0.0),
    7: dict(motion_speed=0.5, num_objects=0.3, occlusion=0.0, proximity=0.8,
            deformation=1.0, topology=0.0, thin_geometry=0.0, scale_variation=0.0,
            direction_changes=0.2, texture_complexity=0.0),
    8: dict(motion_speed=0.6, num_objects=0.5, occlusion=0.0, proximity=0.1,
            deformation=0.0, topology=0.0, thin_geometry=1.0, scale_variation=0.0,
            direction_changes=0.2, texture_complexity=0.0),
    9: dict(motion_speed=0.2, num_objects=0.5, occlusion=0.0, proximity=0.9,
            deformation=0.5, topology=1.0, thin_geometry=0.0, scale_variation=0.0,
            direction_changes=0.0, texture_complexity=0.0),
   10: dict(motion_speed=0.5, num_objects=0.0, occlusion=0.0, proximity=0.0,
            deformation=0.0, topology=0.0, thin_geometry=0.0, scale_variation=0.3,
            direction_changes=0.0, texture_complexity=1.0),
}
# fmt: on


def compute_complexity_scores() -> Dict[int, float]:
    """Return the weighted complexity score C ∈ [0, 1] for every scene."""
    return {
        sid: round(sum(FACTOR_WEIGHTS[k] * v for k, v in factors.items()), 3)
        for sid, factors in _FACTOR_SCORES.items()
    }


def compute_complexity_ranks(scores: Dict[int, float]) -> Dict[int, int]:
    """Return complexity rank (1 = most complex) for every scene."""
    sorted_scenes = sorted(scores, key=lambda s: scores[s], reverse=True)
    return {scene: rank + 1 for rank, scene in enumerate(sorted_scenes)}


COMPLEXITY_SCORES: Dict[int, float] = compute_complexity_scores()
COMPLEXITY_RANKS:  Dict[int, int]   = compute_complexity_ranks(COMPLEXITY_SCORES)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SceneMetrics:
    method:     str
    scene:      int
    iterations: List[int]   = field(default_factory=list)
    psnr:       List[float] = field(default_factory=list)
    ssim:       List[float] = field(default_factory=list)
    lpips:      List[float] = field(default_factory=list)

    def final(self, metric: str) -> Optional[float]:
        """Last recorded value for a metric (proxy for end-of-training eval)."""
        values: list = getattr(self, metric)
        return values[-1] if values else None

    def is_empty(self) -> bool:
        return len(self.iterations) == 0


# ---------------------------------------------------------------------------
# Per-method placeholder parsers
# ---------------------------------------------------------------------------
# Each function receives the raw list of record dicts loaded from the JSON
# file and returns a populated SceneMetrics object.
#
# Fill in the body of each function to handle method-specific quirks:
#   - Field renaming        (e.g. "PSNR" → "psnr")
#   - Unit conversion       (e.g. SSIM stored as percentage)
#   - Filtering checkpoints (e.g. drop validation-only rows)
#   - Aggregating sub-keys  (e.g. per-camera metrics → mean)


def parse_3dgs(records: List[Dict[str, Any]], scene: int) -> SceneMetrics:
    """
    Parse 3DGS metrics records → SceneMetrics.

    TODO: remap 3DGS-specific field names / apply unit conversions here.
    Currently passes records through unchanged assuming the standard schema:
        {"iteration": int, "psnr": float, "ssim": float, "lpips": float}
    """
    m = SceneMetrics(method="3dgs", scene=scene)
    for r in records:
        m.iterations.append(int(r["iteration"]))
        m.psnr.append(float(r["psnr"]))
        m.ssim.append(float(r["ssim"]))
        m.lpips.append(float(r["lpips"]))
    return m


def parse_4dgs(records: List[Dict[str, Any]], scene: int) -> SceneMetrics:
    """
    Parse 4DGS metrics records → SceneMetrics.

    TODO: remap 4DGS-specific field names / apply unit conversions here.
    Currently passes records through unchanged assuming the standard schema:
        {"iteration": int, "psnr": float, "ssim": float, "lpips": float}
    """
    m = SceneMetrics(method="4dgs", scene=scene)
    for r in records:
        m.iterations.append(int(r["iteration"]))
        m.psnr.append(float(r["psnr"]))
        m.ssim.append(float(r["ssim"]))
        m.lpips.append(float(r["lpips"]))
    return m


def parse_deformable(records: List[Dict[str, Any]], scene: int) -> SceneMetrics:
    """
    Parse Deformable 3DGS metrics records → SceneMetrics.

    TODO: remap Deformable-3DGS-specific field names / apply unit conversions here.
    Currently passes records through unchanged assuming the standard schema:
        {"iteration": int, "psnr": float, "ssim": float, "lpips": float}
    """
    m = SceneMetrics(method="deformable", scene=scene)
    for r in records:
        m.iterations.append(int(r["iteration"]))
        m.psnr.append(float(r["psnr"]))
        m.ssim.append(float(r["ssim"]))
        m.lpips.append(float(r["lpips"]))
    return m


PARSERS = {
    "3dgs":       parse_3dgs,
    "4dgs":       parse_4dgs,
    "deformable": parse_deformable,
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json_records(path: Path) -> List[Dict[str, Any]]:
    """
    Load a metrics JSON file.  Accepts:
      - JSON array:       [{"iteration": 1000, ...}, ...]
      - NDJSON:           one JSON object per line
      - Concatenated JSON: multiple top-level objects in one file
    """
    text = path.read_text().strip()
    if not text:
        return []
    # Try standard JSON first (array or single object)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass
    # Fall back to NDJSON
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def load_all_metrics(metrics_dir: Path) -> Dict[str, Dict[int, SceneMetrics]]:
    """
    Walk metrics_dir and load every scene file found.

    Returns:  {method_key: {scene_id: SceneMetrics}}
    """
    all_data: Dict[str, Dict[int, SceneMetrics]] = {m: {} for m in METHOD_KEYS}

    method_dirs = {
        "3dgs":       metrics_dir / "3dgs",
        "4dgs":       metrics_dir / "4dgs",
        "deformable": metrics_dir / "deformable",
    }

    for method, directory in method_dirs.items():
        if not directory.exists():
            print(f"  [warn] directory not found: {directory}")
            continue
        for scene_id in range(1, 11):
            json_path = directory / f"scene{scene_id}.json"
            if not json_path.exists():
                continue
            try:
                records = load_json_records(json_path)
                parsed  = PARSERS[method](records, scene_id)
                all_data[method][scene_id] = parsed
                print(f"  loaded {method}/scene{scene_id}  ({len(records)} records)")
            except Exception as exc:
                print(f"  [warn] failed to load {json_path}: {exc}")

    return all_data


# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------

def _apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi":        150,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "legend.fontsize":   9,
    })


# ---------------------------------------------------------------------------
# Graph type 1 — training curves per scene
# ---------------------------------------------------------------------------

def plot_training_curves(
    all_data: Dict[str, Dict[int, SceneMetrics]],
    output_dir: Path,
) -> None:
    """
    For each scene produce one figure with 3 subplots (PSNR / SSIM / LPIPS)
    showing all three methods' values over training iterations.

    Output: output_dir/training_curves/scene{N:02d}_training_curves.png
            10 figures × 3 subplots = 30 plots total.
    """
    _apply_style()
    out = output_dir / "training_curves"
    out.mkdir(parents=True, exist_ok=True)

    for scene_id in range(1, 11):
        score = COMPLEXITY_SCORES[scene_id]
        rank  = COMPLEXITY_RANKS[scene_id]
        title = (
            f"Scene {scene_id}: {SCENE_NAMES[scene_id].replace(chr(10), ' ')}  "
            f"| complexity {score:.3f}  rank {rank}/10"
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle(title, fontsize=12, y=1.01)

        has_data = False
        for method in METHOD_KEYS:
            sm = all_data.get(method, {}).get(scene_id)
            if sm is None or sm.is_empty():
                continue
            has_data = True
            for ax, metric in zip(axes, METRICS):
                ax.plot(
                    sm.iterations,
                    getattr(sm, metric),
                    label=METHOD_LABELS[method],
                    color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method],
                    markersize=3,
                    linewidth=1.8,
                    markevery=max(1, len(sm.iterations) // 10),
                )

        for ax, metric in zip(axes, METRICS):
            ax.set_xlabel("Iteration")
            ax.set_ylabel(METRIC_LABELS[metric])
            ax.set_title(f"{METRIC_LABELS[metric]}  ({METRIC_BETTER[metric]})")
            ax.legend(loc="best")

        if not has_data:
            for ax in axes:
                ax.text(
                    0.5, 0.5, "No data available",
                    ha="center", va="center",
                    transform=ax.transAxes, color="gray", fontsize=11,
                )

        plt.tight_layout()
        dest = out / f"scene{scene_id:02d}_training_curves.png"
        fig.savefig(dest, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {dest}")


# ---------------------------------------------------------------------------
# Graph type 2 — eval metrics vs scene complexity rank
# ---------------------------------------------------------------------------

def plot_eval_vs_complexity(
    all_data: Dict[str, Dict[int, SceneMetrics]],
    output_dir: Path,
) -> None:
    """
    For each metric produce one figure with all three methods plotted as lines,
    x-axis = complexity rank (1 = most complex), y-axis = final eval metric.

    Output: output_dir/eval_vs_complexity/eval_{metric}.png
            3 figures total.
    """
    _apply_style()
    out = output_dir / "eval_vs_complexity"
    out.mkdir(parents=True, exist_ok=True)

    # Scenes ordered from most to least complex
    ranked_scenes = sorted(range(1, 11), key=lambda s: COMPLEXITY_RANKS[s])
    x_pos = np.arange(len(ranked_scenes))
    x_labels = [
        f"Rank {COMPLEXITY_RANKS[s]}\nS{s} ({COMPLEXITY_SCORES[s]:.2f})"
        for s in ranked_scenes
    ]

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(13, 5))

        for method in METHOD_KEYS:
            y_vals: List[Optional[float]] = []
            for scene_id in ranked_scenes:
                sm  = all_data.get(method, {}).get(scene_id)
                val = sm.final(metric) if sm else None
                y_vals.append(val)

            valid_xy = [(x, y) for x, y in zip(x_pos, y_vals) if y is not None]
            if not valid_xy:
                continue
            xs, ys = zip(*valid_xy)
            ax.plot(
                xs, ys,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=8,
                linewidth=2.0,
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_xlabel("Scene ordered by complexity (rank 1 = most complex)")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(
            f"{METRIC_LABELS[metric]} at End of Training vs Scene Complexity\n"
            f"({METRIC_BETTER[metric]})"
        )
        ax.legend(loc="best")

        plt.tight_layout()
        dest = out / f"eval_{metric}.png"
        fig.savefig(dest, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {dest}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_complexity_table() -> None:
    print("\nScene Complexity Scores")
    print("=" * 62)
    print(f"{'#':<4} {'Scene':<38} {'Score':>6}  {'Rank':>4}")
    print("-" * 62)
    for sid in range(1, 11):
        name  = SCENE_NAMES[sid].replace("\n", " ")
        score = COMPLEXITY_SCORES[sid]
        rank  = COMPLEXITY_RANKS[sid]
        print(f"  {sid:<3} {name:<38} {score:>6.3f}  {rank:>4}")
    print("=" * 62)
    print(
        "\nFormula:  C = "
        + " + ".join(f"{w}·{k[:4]}" for k, w in FACTOR_WEIGHTS.items())
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate metrics analysis plots for 4DGS benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--metrics_dir", default="metrics",
        help="Root directory with 3dgs/, 4dgs/, deformable/ sub-dirs (default: metrics)",
    )
    parser.add_argument(
        "--output_dir", default="figures",
        help="Directory to save generated figures (default: figures)",
    )
    parser.add_argument(
        "--complexity_only", action="store_true",
        help="Print the complexity table and exit without generating plots",
    )
    args = parser.parse_args()

    print_complexity_table()

    if args.complexity_only:
        return

    metrics_dir = Path(args.metrics_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading metrics from: {metrics_dir}")
    all_data = load_all_metrics(metrics_dir)

    print("\nGenerating training curve plots  (10 figures × 3 subplots)…")
    plot_training_curves(all_data, output_dir)

    print("\nGenerating eval vs complexity plots  (3 figures)…")
    plot_eval_vs_complexity(all_data, output_dir)

    print(f"\nAll figures saved to: {output_dir}/")
    print("  training_curves/      — scene*_training_curves.png")
    print("  eval_vs_complexity/   — eval_psnr.png  eval_ssim.png  eval_lpips.png")


if __name__ == "__main__":
    main()
