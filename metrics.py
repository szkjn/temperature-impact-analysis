"""Run the temperature-vs-metrics analysis on saved gpt-4.1 sweeps.

Mirrors the pipeline in metrics.ipynb: loads each results/*.json, computes
per-sample metrics, aggregates by temperature, and renders the figure.

Usage:
    uv run python metrics.py
    uv run python metrics.py --language en --save-figures --dark
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer

from text_analysis import (
    calculate_temperature_stats,
    export_temperature_stats_to_csv,
    plot_single_model_metrics,
    process_model_data,
)

FIGURES_DIR = Path("figures")

# (results_path, figure_filename, error_type, robust_center)
SWEEPS = [
    ("results/results_openai.json",
     "gpt-4.1_full_sweep.png", "iqr", "median"),
    ("results/results_gpt-4.1_temp_0.6-1.6_step_0.05_tokens_200.json",
     "gpt-4.1_0.6-1.6_step_0.05.png", "iqr", "median"),
    ("results/results_gpt-4.1_temp_1-1.4_step_0.02_tokens_200.json",
     "gpt-4.1_1.0-1.4_step_0.02.png", "iqr", "median"),
    ("results/results_gpt-4.1_temp_1-1.5_step_0.025_tokens_200.json",
     "gpt-4.1_1.0-1.5_step_0.025_mad.png", "mad", "mean"),
    ("results/results_gpt-4.1_temp_1-1.5_step_0.025_tokens_200.json",
     "gpt-4.1_1.0-1.5_step_0.025_iqr.png", "iqr", "median"),
]


def load_and_analyze(path, sbert_model, label="gpt-4.1"):
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    df = pd.DataFrame(raw_data)
    analysis_df = process_model_data(df, sbert_model, label)
    return calculate_temperature_stats(analysis_df)


def render(temp_stats, model_name, filename, language, save_figures,
           error_type, robust_center):
    fig, _ = plot_single_model_metrics(
        temp_stats, model_name,
        error_type=error_type, robust_center=robust_center,
        show_zones=True, language=language,
    )
    if save_figures:
        FIGURES_DIR.mkdir(exist_ok=True)
        fig.savefig(FIGURES_DIR / filename, dpi=150, transparent=True,
                    bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--language", choices=["fr", "en"], default="fr",
                        help="Plot label language (default: fr)")
    parser.add_argument("--save-figures", action="store_true",
                        help="Save rendered figures to figures/")
    parser.add_argument("--dark", action="store_true",
                        help="Use matplotlib dark_background style")
    parser.add_argument("--export-csv", action="store_true",
                        help="Export aggregated stats for the full sweep to CSV")
    args = parser.parse_args()

    if args.dark:
        plt.style.use("dark_background")

    sbert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    cached = {}
    for path, filename, error_type, robust_center in SWEEPS:
        print(f"\n=== {path} ({error_type}/{robust_center}) ===")
        if path not in cached:
            cached[path] = load_and_analyze(path, sbert_model)
        render(cached[path], "gpt-4.1", filename,
               args.language, args.save_figures,
               error_type, robust_center)

    if args.export_csv:
        export_temperature_stats_to_csv(
            cached["results/results_openai.json"], "gpt-4.1"
        )


if __name__ == "__main__":
    main()
