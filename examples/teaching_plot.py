"""Plot the actual first-analysis simulation, including its nonsignificant age contrast."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from conlens import plot_null_distribution, plot_running_sum
from examples.teaching_workflow import first_analysis


def main():
    example = first_analysis()
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for row, name in enumerate(["patient_vs_control", "age"]):
        result = example["fit"][name]
        item = result.get("A--A")
        plot_running_sum(result, "A--A", ax=axes[row, 0])
        plot_null_distribution(result, "A--A", ax=axes[row, 1])
        axes[row, 0].set_title(f"{name}: A--A (q={item.q_value:.4f})")
    figure.suptitle("Simulated teaching data | 120 subjects | 399 FL permutations")
    output = Path("website/public/figures/teaching-analysis.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
