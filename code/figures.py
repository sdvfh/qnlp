import csv
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(results_dir: Path) -> pd.DataFrame:
    """
    Loads metric results from pickle files located in the specified directory.

    Each pickle file should contain at least the following keys:
    'level', 'model', 'n_layers', 'false_positive_rate', 'true_positive_rate'.

    Parameters:
    -----------
    results_dir : Path
        Directory where the pickle result files are stored.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing all loaded results. The 'model' column is extracted
        from the file name by splitting on underscores.
    """
    results = []
    for file in results_dir.rglob("*.pkl"):
        with open(file, "rb") as f:
            metrics: Dict[str, Any] = pickle.load(f)
            metrics["file"] = file.stem
            results.append(metrics)
    df = pd.DataFrame(results)
    # Extract the model name from the file name (e.g., "AnsatzRot_...").
    df["model"] = df["file"].apply(lambda x: x.split("_")[0])
    return df


def compute_average_roc(
    df: pd.DataFrame,
) -> Dict[str, Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]]]:
    """
    Computes the average ROC curves for each (level, model, n_layers) group.

    The function performs the following steps:
    1. Groups the DataFrame by 'level', 'model', and 'n_layers'.
    2. Ensures that each individual ROC curve starts at (0, 0) and ends at (1, 1).
    3. Interpolates all curves onto a common grid of FPR (False Positive Rate) values
       spanning from 0 to 1 in 1000 equally spaced points.
    4. Calculates the average TPR (True Positive Rate) across the interpolated curves.

    Parameters:
    -----------
    df : pd.DataFrame
        Must contain the columns 'level', 'model', 'n_layers', 'false_positive_rate',
        and 'true_positive_rate'.

    Returns:
    --------
    Dict[str, Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]]]
        A nested dictionary structured as follows:
        {
            level: {
                model: {
                    n_layers: (mean_fpr, mean_tpr)
                }
            }
        }
        where 'mean_fpr' is the common FPR grid and 'mean_tpr' is the average TPR.
    """
    roc_avg: Dict[str, Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]]] = {}
    grouped = df.groupby(["level", "model", "n_layers"])

    # Define a common grid of FPR values from 0 to 1 (1000 points) for interpolation.
    common_fpr = np.linspace(0, 1, 1000)

    for (level, model, n_layers), group in grouped:
        fpr_list = group["false_positive_rate"].tolist()
        tpr_list = group["true_positive_rate"].tolist()

        interpolated_tprs = []
        for fpr, tpr in zip(fpr_list, tpr_list, strict=True):
            fpr = np.array(fpr, dtype=float)
            tpr = np.array(tpr, dtype=float)
            # Sort the points to ensure the curve progresses in ascending order of FPR.
            sorted_indices = np.argsort(fpr)
            fpr = fpr[sorted_indices]
            tpr = tpr[sorted_indices]

            # Ensure the curve starts at (0, 0) if needed.
            if fpr[0] > 0:
                fpr = np.insert(fpr, 0, 0.0)
                tpr = np.insert(tpr, 0, 0.0)
            # Ensure the curve ends at (1, 1) if needed.
            if fpr[-1] < 1:
                fpr = np.append(fpr, 1.0)
                tpr = np.append(tpr, 1.0)

            # Interpolate the TPR values onto the common FPR grid.
            interp_tpr = np.interp(common_fpr, fpr, tpr)
            interpolated_tprs.append(interp_tpr)

        mean_tpr = np.mean(interpolated_tprs, axis=0)

        if level not in roc_avg:
            roc_avg[level] = {}
        if model not in roc_avg[level]:
            roc_avg[level][model] = {}

        # Store the (common_fpr, mean_tpr) tuple.
        roc_avg[level][model][n_layers] = (common_fpr, mean_tpr)

    return roc_avg


def plot_roc_curves(
    roc_avg: Dict[str, Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]]],
    output_dir: Path,
) -> None:
    """
    Plots ROC curves for each difficulty level.

    The function creates one plot per 'level' and includes the following:
    - A dashed diagonal line representing the random chance (from (0,0) to (1,1)).
    - Solid lines for n_layers = 1, shown in a darker color.
    - Dashed lines for n_layers = 10, shown in a lighter version of the same color.
    - The same base color is used for the same model, varying only by line style
      and brightness to differentiate the number of layers.

    Parameters:
    -----------
    roc_avg : Dict[str, Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]]]
        Nested dictionary of average ROC curves.
    output_dir : Path
        Directory in which to save the resulting ROC plot images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create one plot per difficulty level.
    for level, models_data in roc_avg.items():
        plt.figure(figsize=(8, 6))

        # Plot the diagonal line representing a random classifier.
        plt.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random Chance")

        # Build a color map to ensure the same base color is used for the same model.
        unique_models = sorted(models_data.keys())
        color_cycle = plt.cm.tab10(np.linspace(0, 1, len(unique_models)))
        model_to_color = dict(zip(unique_models, color_cycle, strict=True))

        for model, layers_data in models_data.items():
            color = model_to_color[model]

            # Plot the ROC for n_layers = 1 with a solid line (darker color).
            if 1 in layers_data:
                mean_fpr, mean_tpr = layers_data[1]
                plt.plot(
                    mean_fpr,
                    mean_tpr,
                    label=f"{model} (1 layer)",
                    linewidth=3,
                    color=color,
                    linestyle="-",
                    marker="o",
                    markevery=50,  # place a marker every 50 points, for instance
                )

            # Plot the ROC for n_layers = 10 with a dashed line (lighter color).
            if 10 in layers_data:
                mean_fpr, mean_tpr = layers_data[10]
                plt.plot(
                    mean_fpr,
                    mean_tpr,
                    label=f"{model} (10 layers)",
                    linewidth=1.5,
                    color=color,
                    linestyle="--",
                    marker="^",
                    markevery=50,
                    alpha=0.8,
                )

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for Difficulty Level: {level}")
        plt.legend(loc="lower right")
        plt.grid(True)

        # Save the plot to a PNG file.
        output_file = output_dir / f"roc_curve_{level}.png"
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()


def main() -> None:
    """
    Main function that:
    1. Loads the results from pickle files.
    2. Computes the average ROC curves.
    3. Plots and saves the resulting ROC curves to disk.
    """
    # Adjust the paths.
    root_path: Path = Path(__file__).parent.parent.resolve()
    results_path: Path = root_path / "results"

    # Load the results into a DataFrame.
    data_df: pd.DataFrame = load_results(results_path)

    # Verify that all required columns exist.
    required_columns = [
        "level",
        "model",
        "n_layers",
        "false_positive_rate",
        "true_positive_rate",
    ]
    if not all(col in data_df.columns for col in required_columns):
        raise ValueError(
            f"The DataFrame must contain the following columns: {required_columns}"
        )

    # Compute the average ROC curves.
    roc_avg = compute_average_roc(data_df)

    # Plot and save the ROC curves.
    plot_roc_curves(roc_avg, results_path)


if __name__ == "__main__":
    main()
