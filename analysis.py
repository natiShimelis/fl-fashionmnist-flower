"""Visualize and compare results from multiple federated learning runs.

Usage:
    python analysis.py
"""

import json
import pathlib

import matplotlib.pyplot as plt
from prettytable import PrettyTable


class ResultsVisualizer:
    """Load, display, and plot results from federated learning runs."""

    def __init__(self):
        # Maps run name -> list of round dicts loaded from JSON
        self.runs = {}

    def add_run(self, name, file_path):
        """Load a results JSON file and store it under the given name."""
        with open(file_path) as f:
            self.runs[name] = json.load(f)
        print(f"Loaded run '{name}' from {file_path} ({len(self.runs[name])} rounds)")

    def print_run_summary_table(self):
        """Print a table showing final-round metrics for each run."""
        table = PrettyTable()
        table.field_names = ["Run", "train_loss", "train_acc", "eval_loss", "eval_acc"]

        for name, rounds in self.runs.items():
            last = rounds[-1]  # final round results
            table.add_row([
                name,
                f"{last['train_loss']:.4f}",
                f"{last['train_acc']:.4f}",
                f"{last['eval_loss']:.4f}",
                f"{last['eval_acc']:.4f}",
            ])

        print(table)

    def plot_metric(self, metric, fig_directory):
        """Plot a single metric across rounds for all runs, save as PNG."""
        plt.figure()

        for name, rounds in self.runs.items():
            x = [r["round"] for r in rounds]
            y = [r[metric] for r in rounds]
            plt.plot(x, y, marker="o", label=name)

        plt.xlabel("Round")
        plt.ylabel(metric)
        plt.title(f"{metric} vs Round")
        plt.legend()
        plt.grid(True)

        out_dir = pathlib.Path(fig_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{metric}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Saved plot: {out_path}")

    def plot_all(self, fig_directory="figures"):
        """Plot all four standard metrics."""
        for metric in ["train_loss", "train_acc", "eval_loss", "eval_acc"]:
            self.plot_metric(metric, fig_directory)


def main():
    visualizer = ResultsVisualizer()

    # Load all result files from the three experiment subdirectories
    for subdir in ["vary_alpha", "vary_clients", "vary_fraction", "vary_optimizer"]:
        for path in sorted((pathlib.Path("results") / subdir).glob("*.json")):
            name = f"{subdir}/{path.stem}"
            visualizer.add_run(name, path)

    if not visualizer.runs:
        print("No result files found in results/. Run a simulation first.")
        return

    visualizer.print_run_summary_table()
    for subdir in ["vary_alpha", "vary_clients", "vary_fraction", "vary_optimizer"]:
        runs_in_subdir = {k: v for k, v in visualizer.runs.items() if k.startswith(subdir)}
        if runs_in_subdir:
            sub_viz = ResultsVisualizer()
            sub_viz.runs = runs_in_subdir
            sub_viz.plot_all(fig_directory=f"figures/{subdir}")


if __name__ == "__main__":
    main()
