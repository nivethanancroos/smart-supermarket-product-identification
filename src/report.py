"""Module 4: Statistical analysis and report generation.

Turns a list of per-item detections into the summary the assignment asks
for: total count, category-wise counts, distribution percentages, and
bar/pie charts.
"""

import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # we only save figures, never show them; avoids GUI/Qt backend issues
import matplotlib.pyplot as plt

from src import config


def generate_report(detections: list[dict], output_dir: Path = config.OUTPUT_DIR) -> dict:
    """detections: list of {"label": str, "confidence": float, "bbox": [...]}."""
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [d["label"] for d in detections]
    counts = Counter(labels)
    total = len(labels)

    summary = {
        "total_products": total,
        "category_counts": dict(counts),
        "category_percentages": {
            label: round(count / total * 100, 2) for label, count in counts.items()
        } if total else {},
    }

    _print_console_report(summary)
    _save_csv(summary, output_dir / "report.csv")
    if total:
        _save_bar_chart(summary, output_dir / "category_bar_chart.png")
        _save_pie_chart(summary, output_dir / "category_pie_chart.png")

    return summary


def _print_console_report(summary: dict):
    print("\n===== PRODUCT DETECTION SUMMARY =====")
    print(f"Total products detected: {summary['total_products']}")
    print(f"{'Category':<20}{'Count':<10}{'Percentage':<10}")
    print("-" * 40)
    for label, count in sorted(summary["category_counts"].items(), key=lambda kv: -kv[1]):
        pct = summary["category_percentages"][label]
        print(f"{label:<20}{count:<10}{pct:<10.2f}")
    print("=" * 40)


def _save_csv(summary: dict, path: Path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "count", "percentage"])
        for label, count in summary["category_counts"].items():
            writer.writerow([label, count, summary["category_percentages"][label]])


def _save_bar_chart(summary: dict, path: Path):
    items = sorted(summary["category_counts"].items(), key=lambda kv: -kv[1])
    labels, counts = zip(*items)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), 4))
    ax.bar(labels, counts, color="#4C72B0")
    ax.set_ylabel("Count")
    ax.set_title("Products Detected per Category")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path)


def _save_pie_chart(summary: dict, path: Path):
    items = sorted(summary["category_counts"].items(), key=lambda kv: -kv[1])
    labels, counts = zip(*items)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Product Distribution")
    fig.tight_layout()
    fig.savefig(path)
