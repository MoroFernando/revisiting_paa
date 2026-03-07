# Revisiting PAA

This repository contains the experimental framework for the research project **"Revisiting PAA"**. The study re-evaluates the classic Piecewise Aggregate Approximation (PAA) method by replacing the traditional "mean" operator with 17 different statistical aggregators across multiple time series classification and structural preservation benchmarks.

## 📌 Project Overview

The goal of this research is to analyze how different local aggregation strategies impact:
1.  **Classification Performance:** Accuracy and computational efficiency across state-of-the-art classifiers.
2.  **Structural Preservation:** How well the reduced series preserves the original neighborhood relationships ($P@k$ and Trustworthiness).

## 📂 Project Structure

The project follows a modular architecture designed for reproducibility in high-performance computing (HPC) environments:

```text
Revisiting_PAA/
├── src/
│   ├── aggregators.py   # Implementation of the 17 aggregation operators
│   ├── metrics.py       # Neighborhood preservation metrics (P@5, Trustworthiness)
│   ├── models.py        # Wrapper for classifiers (Rocket, QUANT, LITE, 1NN-DTW)
│   └── data_utils.py    # Dataset loading and Z-normalization logic
├── experiments/
│   ├── bench_classification.py  # Script for classification benchmarks
│   └── bench_neighborhood.py    # Script for structural preservation benchmarks
├── results/             # CSV files with raw experimental data
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation