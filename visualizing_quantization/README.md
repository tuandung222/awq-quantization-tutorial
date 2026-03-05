# Visualizing Quantization 🎨

Welcome to the beginner-friendly, visual guide to Neural Network Quantization!

This folder is designed specifically for Computer Science students (or anyone new to the field) who want to build a strong, intuitive understanding of how Quantization—and specifically the **AWQ (Activation-aware Weight Quantization)** algorithm—works under the hood, without getting bogged down by heavy matrix calculus.

## What's Inside?

### 1. The Intuition Guide
Start here! Read **[`02_intuition_guide.md`](02_intuition_guide.md)** for a conceptual breakdown.
- Uses relatable analogies (like compressing a high-color photo to a GIF).
- Explains the "Outlier / Screaming Student" problem in standard quantization.
- Uses clear Mermaid diagrams to show how the math flows through the hardware.

### 2. The Interactive Visualization Notebook
Next, open **[`01_visualize_basics.ipynb`](01_visualize_basics.ipynb)**.
This interactive Jupyter Notebook uses standard Python libraries (`numpy`, `matplotlib`, `seaborn`) to simulate quantization from scratch. 

You will visually see:
- Continuous sine waves being forced into discrete INT8, INT4, and INT2 "steps."
- A bar chart demonstrating what happens to a group of 128 weights when *one* massive outlier is present.
- A **Heatmap comparison** showing why standard Round-to-Nearest (RTN) fails, and how AWQ's simple division trick miraculously preserves the precision of the network.

## How to Run the Notebook

If you haven't already set up the environment in the main folder:

```bash
# Install the visualization requirements
pip install numpy matplotlib seaborn jupyter

# Launch Jupyter Notebook
jupyter notebook 01_visualize_basics.ipynb
```

---
*Back to the [Main Project README](../README.md)*
