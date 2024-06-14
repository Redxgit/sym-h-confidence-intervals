# metrics.py
# Deep Neural Networks for Geomagnetic Indices Forecasting
# European Space Agency Contract No. 4000137421/22/NL/GLC/my
# WP3000 - DNNs
# Author: Armando Collado
# Last update date: 05-12-2023
# Version: 1.0
# Metrics to evaluate the forecasting models


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

from sklearn.metrics import mean_squared_error as msem
from sklearn.metrics import r2_score as r2m

from matplotlib.colors import LinearSegmentedColormap


BINWIDTH = 10

SYM_H_THRESHOLD_LOW = -90
SYM_H_THRESHOLD_MODERATE = -130
SYM_H_THRESHOLD_INTENSE = -230
SYM_H_THRESHOLD_SUPERINTENSE = -390

COLOR_SUPERINTENSE = "darkmagenta"
COLOR_INTENSE = "firebrick"
COLOR_MODERATE = "goldenrod"
COLOR_LOW = "yellow"
COLOR_INACTIVE = "olivedrab"


def rmse(y_true, y_pred):
    """
    Wrapper
    """
    return msem(y_true, y_pred, squared=False)


def roundup(x):
    return int(np.ceil(x / 10.0)) * 10


def rounddown(x):
    return int(np.floor(x / 10.0)) * 10


def calculate_BFE(labels, preds, binwidth=10):
    df = pd.DataFrame({"labels": labels, "preds": preds})
    df["diff"] = df["labels"] - df["preds"]
    df["diff"] = df["diff"].abs()
    min_val = rounddown(df["labels"].min())
    max_val = roundup(df["labels"].max())
    bins = np.arange(min_val, max_val + binwidth, binwidth)
    df["labels_bins"] = pd.cut(df["labels"], bins=bins, right=False)
    df["labels_bins"] = df["labels_bins"].apply(lambda x: x.left)
    bfe = df.groupby("labels_bins", observed=True)["diff"].mean()
    bfe.index = bfe.index.astype(int)
    return bfe.mean()


def calculate_interval_stats(labels, quantile_start, quantile_end, binwidth=10):
    df = pd.DataFrame(
        {
            "labels": labels,
            "quantile_start": quantile_start,
            "quantile_end": quantile_end,
        }
    )
    df["covered"] = (
        (df["quantile_start"] <= df["labels"]) & (df["labels"] <= df["quantile_end"])
    ).astype(int)

    picp = df["covered"].mean()

    df["interval_width"] = df["quantile_end"] - df["quantile_start"]

    width_mean = df["interval_width"].mean()

    min_val = rounddown(df["labels"].min())
    max_val = roundup(df["labels"].max())
    bins = np.arange(min_val, max_val + binwidth, binwidth)
    df["labels_bins"] = pd.cut(df["labels"], bins=bins, right=False)
    df["labels_bins"] = df["labels_bins"].apply(lambda x: x.left)
    biw = df.groupby("labels_bins", observed=True)["interval_width"].mean()
    biw.index = biw.index.astype(int)
    return picp, width_mean, biw.mean()


def get_BFE(labels, preds, binwidth=10):
    df = pd.DataFrame({"labels": labels, "preds": preds})
    df["diff"] = df["labels"] - df["preds"]
    df["diff"] = df["diff"].abs()
    min_val = rounddown(df["labels"].min())
    max_val = roundup(df["labels"].max())
    bins = np.arange(min_val, max_val + binwidth, binwidth)
    df["labels_bins"] = pd.cut(df["labels"], bins=bins, right=False)
    df["labels_bins"] = df["labels_bins"].apply(lambda x: x.left)
    bfe = df.groupby("labels_bins", observed=True)["diff"].mean()
    bfe.index = bfe.index.astype(int)
    return bfe


def get_BFE_count(labels, binwidth=10):
    df = pd.DataFrame({"labels": labels})
    min_val = rounddown(df["labels"].min())
    max_val = roundup(df["labels"].max())
    bins = np.arange(min_val, max_val + binwidth, binwidth)
    df["labels_bins"] = pd.cut(df["labels"], bins=bins, right=False)
    df["labels_bins"] = df["labels_bins"].apply(lambda x: x.left)
    bfe_count = df.groupby("labels_bins", observed=True)["labels"].count()
    bfe_count.index = bfe_count.index.astype(int)
    return bfe_count


def get_BFE_bins(labels, binwidth=10):
    df = pd.DataFrame({"labels": labels})
    min_val = rounddown(df["labels"].min())
    max_val = roundup(df["labels"].max())
    bins = np.arange(min_val, max_val + binwidth, binwidth)
    return bins


def plot_comparison_bfe(
    df,
    df_to_compare,
    title,
    title_to_compare,
    ax=None,
    plot_sym_bars=False,
    xlabel_title=None,
):
    all_preds = df.copy()
    comparison = df_to_compare.copy()
    index_column_name = df.columns[0]
    comparison["diff_comparison"] = (
        comparison[comparison.columns[0]] - comparison[comparison.columns[1]]
    )
    all_preds["diff_base"] = (
        all_preds[all_preds.columns[0]] - all_preds[all_preds.columns[1]]
    )
    all_preds["diff_comparison"] = comparison["diff_comparison"]

    storm_plot = all_preds.reset_index()
    storm_plot = storm_plot.sort_values(index_column_name)
    storm_plot["diff_base_abs"] = np.abs(storm_plot["diff_base"])
    storm_plot["diff_comparison_abs"] = np.abs(storm_plot["diff_comparison"])
    storm_plot["diff_base_squared"] = storm_plot["diff_base"] * storm_plot["diff_base"]
    storm_plot["diff_comparison_squared"] = (
        storm_plot["diff_comparison"] * storm_plot["diff_comparison"]
    )

    min_index = rounddown(df[all_preds.columns[0]].min())
    max_index = roundup(df[all_preds.columns[0]].max())
    index_range = np.abs(max_index - min_index)

    bins = np.arange(min_index - BINWIDTH, max_index + BINWIDTH, BINWIDTH)

    storm_plot["index_bins"] = pd.cut(
        storm_plot[index_column_name], bins=bins, right=False
    )
    storm_plot["index_bins_mid"] = storm_plot["index_bins"].apply(lambda x: x.mid)
    mean_diff_to_plot_by_bin = storm_plot.groupby("index_bins_mid", observed=True)[
        "diff_base_abs"
    ].mean()
    mean_diff_comparison_to_plot_by_bin = storm_plot.groupby(
        "index_bins_mid", observed=True
    )["diff_comparison_abs"].mean()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(13, 6))
    ax.plot(
        mean_diff_to_plot_by_bin.index,
        mean_diff_to_plot_by_bin.values,
        linestyle="-",
        color="blue",
        label=title,
    )
    ax.plot(
        mean_diff_to_plot_by_bin.index,
        mean_diff_comparison_to_plot_by_bin.values,
        linestyle="-",
        color="gray",
        label=title_to_compare,
    )
    # ax.fill_between(mean_diff_to_plot_by_bin.index, mean_diff_to_plot_by_bin.values, 0, alpha = 0.2, color = 'blue')
    ax.plot(0, 0, label="Bin count", color="green")
    ax.fill_between(
        mean_diff_to_plot_by_bin.index,
        mean_diff_to_plot_by_bin.values,
        mean_diff_comparison_to_plot_by_bin.values,
        where=mean_diff_to_plot_by_bin.values
        > mean_diff_comparison_to_plot_by_bin.values,
        alpha=0.2,
        interpolate=True,
        color="gray",
    )

    ax.fill_between(
        mean_diff_to_plot_by_bin.index,
        mean_diff_to_plot_by_bin.values,
        mean_diff_comparison_to_plot_by_bin.values,
        where=mean_diff_to_plot_by_bin.values
        < mean_diff_comparison_to_plot_by_bin.values,
        alpha=0.2,
        interpolate=True,
        color="blue",
    )

    ax.set_ylim(0, ax.get_ylim()[1])
    ax.tick_params(axis="both", which="major", labelsize=14, width=2, length=10)
    ax.set_xlim(
        mean_diff_to_plot_by_bin.index.min(), mean_diff_to_plot_by_bin.index.max()
    )
    twin = ax.twinx()
    twin.hist(storm_plot[index_column_name], bins=bins, alpha=0.15, color="green")
    twin.set_yscale("log")
    twin.grid(linestyle="--")
    if xlabel_title is None:
        ax.set_xlabel(f"{index_column_name} (nT)", fontsize=18)
    else:
        ax.set_xlabel(xlabel_title, fontsize=18)
    ax.set_ylabel("Mean Absolute Difference (nT)", fontsize=18)

    if plot_sym_bars:
        min_sym = rounddown(df[index_column_name].min())
        if min_sym <= SYM_H_THRESHOLD_LOW:
            ax.axvline(SYM_H_THRESHOLD_LOW, linestyle="--", color=COLOR_LOW)
            if min_sym <= SYM_H_THRESHOLD_MODERATE:
                ax.axvline(
                    SYM_H_THRESHOLD_MODERATE,
                    linestyle="--",
                    color=COLOR_MODERATE,
                )
                if min_sym <= SYM_H_THRESHOLD_INTENSE:
                    ax.axvline(
                        SYM_H_THRESHOLD_INTENSE,
                        linestyle="--",
                        color=COLOR_INTENSE,
                    )
                    if min_sym <= SYM_H_THRESHOLD_SUPERINTENSE:
                        ax.axvline(
                            SYM_H_THRESHOLD_SUPERINTENSE,
                            linestyle="--",
                            color=COLOR_SUPERINTENSE,
                        )

    twin.set_ylabel("Bin count", fontsize=18)
    twin.tick_params(axis="y", which="major", labelsize=14, width=2, length=10)
    twin.tick_params(axis="y", which="minor", width=1, length=5)

    ax.set_title(
        f"Diff BFE ({title_to_compare} - {title}): {mean_diff_comparison_to_plot_by_bin.mean() - mean_diff_to_plot_by_bin.mean():.3f}",
        fontsize=18,
    )

    leg = ax.legend(
        bbox_to_anchor=(0.5, 1.2),
        loc="upper center",
        ncol=3,
        fancybox=True,
        prop={"size": 14},
    )
    leg.get_frame().set_alpha(None)
    leg.get_frame().set_facecolor((0, 0, 0, 0))
    leg.get_lines()[-1].set_linewidth(12.0)
    leg.get_lines()[-1].set_alpha(0.15)
    leg.get_frame().set_edgecolor("black")
    plt.setp(ax.spines.values(), lw=2, color="black", alpha=1)
    twin.grid(False)
    ax.grid(True)

    print(f"Diff BFE ({title_to_compare} - {title}):")
    print(
        f"Total: {mean_diff_comparison_to_plot_by_bin.mean() - mean_diff_to_plot_by_bin.mean():.3f}"
    )

    return ax


def plot_evaluation_bfe(df, title, ax=None, plot_sym_bars=False, xlabel_title=None):
    storm_plot = df.copy()
    original_column = storm_plot.columns[0]
    predicted_column = storm_plot.columns[1]
    storm_plot["diff"] = storm_plot[original_column] - storm_plot[predicted_column]
    storm_plot = storm_plot.reset_index()
    storm_plot = storm_plot.sort_values(original_column)
    storm_plot["diff_abs"] = np.abs(storm_plot["diff"])
    storm_plot["diff_squared"] = storm_plot["diff"] * storm_plot["diff"]

    min_index = rounddown(df[original_column].min())
    max_index = roundup(df[original_column].max())

    bins = np.arange(
        min_index - BINWIDTH, max_index + BINWIDTH, BINWIDTH
    )  # You can adjust the bin size and range as needed

    # Cut SYM_H into bins and calculate mean diff_abs within each bin
    storm_plot["index_bins"] = pd.cut(
        storm_plot[original_column], bins=bins, right=False
    )
    storm_plot["index_bins_mid"] = storm_plot["index_bins"].apply(lambda x: x.left)
    mean_value_to_plot = storm_plot.groupby("index_bins_mid", observed=True)[
        "diff_abs"
    ].mean()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(13, 6))

    ax.plot(
        mean_value_to_plot.index,
        mean_value_to_plot.values,
        linestyle="-",
        label="Evaluated predictions",
    )
    ax.fill_between(
        mean_value_to_plot.index,
        mean_value_to_plot.values,
        0,
        alpha=0.2,
        color="blue",
    )
    ax.plot(0, 0, color="green", label="Bin count")
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.tick_params(axis="both", which="major", labelsize=14, width=2, length=10)
    ax.set_xlim(
        mean_value_to_plot.index.min(),
        mean_value_to_plot.index.max(),
    )
    twin = ax.twinx()
    twin.hist(storm_plot[original_column], bins=bins, alpha=0.15, color="green")
    twin.set_yscale("log")
    twin.grid(linestyle="--")
    if xlabel_title is None:
        ax.set_xlabel(f"{original_column}", fontsize=18)
    else:
        ax.set_xlabel(xlabel_title, fontsize=18)
    ax.set_ylabel("Mean Absolute Difference (nT)", fontsize=18)
    twin.set_ylabel("Bin count", fontsize=18)
    twin.tick_params(axis="y", which="major", labelsize=14, width=2, length=10)
    twin.tick_params(axis="y", which="minor", width=1, length=5)

    mean_value_to_plot.index = mean_value_to_plot.index.astype(int)
    bfe = calculate_BFE(df[original_column].values, df[predicted_column].values)
    ax.set_title(f"{title} | BFE: {bfe:.3f}", fontsize=18)

    if plot_sym_bars:
        min_sym = rounddown(df[original_column].min())
        if min_sym <= SYM_H_THRESHOLD_LOW:
            ax.axvline(SYM_H_THRESHOLD_LOW, linestyle="--", color=COLOR_LOW)
            if min_sym <= SYM_H_THRESHOLD_MODERATE:
                ax.axvline(
                    SYM_H_THRESHOLD_MODERATE,
                    linestyle="--",
                    color=COLOR_MODERATE,
                )
                if min_sym <= SYM_H_THRESHOLD_INTENSE:
                    ax.axvline(
                        SYM_H_THRESHOLD_INTENSE,
                        linestyle="--",
                        color=COLOR_INTENSE,
                    )
                    if min_sym <= SYM_H_THRESHOLD_SUPERINTENSE:
                        ax.axvline(
                            SYM_H_THRESHOLD_SUPERINTENSE,
                            linestyle="--",
                            color=COLOR_SUPERINTENSE,
                        )

    leg = ax.legend(
        ncol=4,
        fancybox=True,
        prop={"size": 14},
        bbox_to_anchor=(0.5, 1.2),
        loc="upper center",
    )
    leg.get_frame().set_alpha(None)
    leg.get_frame().set_facecolor((0, 0, 0, 0))
    leg.get_lines()[-1].set_linewidth(12.0)
    leg.get_lines()[-1].set_alpha(0.15)
    leg.get_frame().set_edgecolor("black")
    plt.setp(ax.spines.values(), lw=2, color="black", alpha=1)
    twin.grid(False)
    ax.grid(True)

    return ax


def plot_forecast(
    dfx,
    title,
    ax=None,
    plot_sym_bars=False,
    ylabel_title="SYM-H (nT)",
    plot_prediction_error=True,
    observed_color="blue",
    predicted_color="red",
    prediction_error_color="black",
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(13, 6))

    df = dfx.copy()

    observed_column = df.columns[0]
    predicted_column = df.columns[1]

    month = df.index[len(df) // 2].month_name()
    year = df.index[len(df) // 2].year

    df["prediction_error"] = df[observed_column] - df[predicted_column]

    ax.plot(df.index, df["mse_predicted_SYM_H"], color=observed_color, alpha=0.8)

    ax.plot(df.index, df["SYM_H"], color=predicted_color, alpha=0.8)

    if plot_prediction_error:
        ax.plot(
            df.index, df["prediction_error"], color=prediction_error_color, alpha=0.5
        )

    rmse_val = rmse(df[observed_column], df[predicted_column])
    r2_val = r2m(df[observed_column], df[predicted_column])
    bfe_val = calculate_BFE(df[observed_column], df[predicted_column])

    ax.set_xlabel(f"{month} of {year}", fontsize=18)
    ax.set_ylabel(ylabel_title, fontsize=18)

    ax.set_title(
        f"{title}\nBFE: {bfe_val:.3f} | RMSE: {rmse_val:.3f} | R2: {r2_val:.3f}",
        fontsize=18,
    )

    if plot_sym_bars:
        min_sym = rounddown(df[observed_column].min())
        if min_sym <= SYM_H_THRESHOLD_LOW:
            ax.axhline(SYM_H_THRESHOLD_LOW, linestyle="--", color=COLOR_LOW)
            if min_sym <= SYM_H_THRESHOLD_MODERATE:
                ax.axhline(
                    SYM_H_THRESHOLD_MODERATE,
                    linestyle="--",
                    color=COLOR_MODERATE,
                )
                if min_sym <= SYM_H_THRESHOLD_INTENSE:
                    ax.axhline(
                        SYM_H_THRESHOLD_INTENSE,
                        linestyle="--",
                        color=COLOR_INTENSE,
                    )
                    if min_sym <= SYM_H_THRESHOLD_SUPERINTENSE:
                        ax.axhline(
                            SYM_H_THRESHOLD_SUPERINTENSE,
                            linestyle="--",
                            color=COLOR_SUPERINTENSE,
                        )

    plt.setp(ax.spines.values(), lw=2, color="black", alpha=1)

    ax.xaxis_date()

    ax.xaxis.set_major_locator(
        mdates.DayLocator(interval=1)
    )  # Adjust interval as needed
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    # Adjust the tick parameters for better visibility
    ax.tick_params(axis="x", which="major", labelsize=14, width=2, length=10)
    ax.tick_params(axis="y", which="major", labelsize=14, width=2, length=10)

    # Ensure that the grid is enabled and properly configured
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.5)

    ax.set_xlim(df.index[0], df.index[-1])

    if plot_prediction_error:
        custom_lines = [
            plt.Line2D([0], [0], color=observed_color, alpha=0.8),
            plt.Line2D([0], [0], color=predicted_color, alpha=0.8),
            plt.Line2D([0], [0], color=prediction_error_color, alpha=0.5),
        ]
    else:
        custom_lines = [
            plt.Line2D([0], [0], color=observed_color),
            plt.Line2D([0], [0], color=predicted_color),
        ]

    handles = []
    labels = []

    handles.extend(custom_lines)

    labels.extend(["Observed SYM-H", "Predicted SYM-H"])
    if plot_prediction_error:
        labels.extend(["Prediction Error"])

    leg = ax.legend(
        handles=handles,
        labels=labels,
        ncol=len(handles),
        fancybox=True,
        prop={"size": 14},
        bbox_to_anchor=(0.5, 1.25),
        loc="upper center",
        edgecolor="black",
    )
    leg.get_frame().set_alpha(None)
    leg.get_frame().set_facecolor((0, 0, 0, 0))
    return fig, ax


def plot_forecast_quantile(
    dfx,
    title,
    ax=None,
    plot_sym_bars=False,
    ylabel_title="SYM-H (nT)",
    plot_prediction_error=True,
    observed_color="blue",
    predicted_color="red",
    prediction_error_color="black",
    small_quantile_color=[180 / 255, 225 / 255, 188 / 255, 1],
    big_quantile_color=[180 / 255, 225 / 255, 250 / 255, 1],
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(13, 6))

    df = dfx.copy()

    observed_column = df.columns[0]
    predicted_column = df.columns[1]
    if len(df.columns) > 4:
        multiple_q = True
        big_quantile_column_start = df.columns[2]
        small_quantile_column_start = df.columns[3]
        small_quantile_column_end = df.columns[4]
        big_quantile_column_end = df.columns[5]
    else:
        multiple_q = False
        quantile_column_start = df.columns[2]
        quantile_column_end = df.columns[3]

    month = df.index[len(df) // 2].month_name()
    year = df.index[len(df) // 2].year

    df["prediction_error"] = df[observed_column] - df[predicted_column]

    ax.plot(df.index, df["SYM_H"], color=observed_color, alpha=0.8)

    ax.plot(df.index, df["mse_predicted_SYM_H"], color=predicted_color, alpha=0.8)

    if plot_prediction_error:
        ax.plot(
            df.index, df["prediction_error"], color=prediction_error_color, alpha=0.5
        )

    if multiple_q:
        ax.fill_between(
            df.index,
            df[big_quantile_column_start],
            df[small_quantile_column_start],
            alpha=0.8,
            interpolate=True,
            color=big_quantile_color,
        )

        ax.fill_between(
            df.index,
            df[small_quantile_column_start],
            df[small_quantile_column_end],
            alpha=0.8,
            interpolate=True,
            color=small_quantile_color,
        )

        ax.fill_between(
            df.index,
            df[small_quantile_column_end],
            df[big_quantile_column_end],
            alpha=0.8,
            interpolate=True,
            color=big_quantile_color,
        )
    else:
        ax.fill_between(
            df.index,
            df[quantile_column_start],
            df[quantile_column_end],
            alpha=0.8,
            interpolate=True,
            color=big_quantile_color,
        )

    rmse_val = rmse(df[observed_column], df[predicted_column])
    r2_val = r2m(df[observed_column], df[predicted_column])
    bfe_val = calculate_BFE(df[observed_column], df[predicted_column])

    ax.set_xlabel(f"{month} of {year}", fontsize=18)
    ax.set_ylabel(ylabel_title, fontsize=18)

    ax.set_title(
        f"{title}\nBFE: {bfe_val:.3f} | RMSE: {rmse_val:.3f} | R2: {r2_val:.3f}",
        fontsize=18,
    )

    if plot_sym_bars:
        min_sym = rounddown(df[observed_column].min())
        if min_sym <= SYM_H_THRESHOLD_LOW:
            ax.axhline(SYM_H_THRESHOLD_LOW, linestyle="--", color=COLOR_LOW)
            if min_sym <= SYM_H_THRESHOLD_MODERATE:
                ax.axhline(
                    SYM_H_THRESHOLD_MODERATE,
                    linestyle="--",
                    color=COLOR_MODERATE,
                )
                if min_sym <= SYM_H_THRESHOLD_INTENSE:
                    ax.axhline(
                        SYM_H_THRESHOLD_INTENSE,
                        linestyle="--",
                        color=COLOR_INTENSE,
                    )
                    if min_sym <= SYM_H_THRESHOLD_SUPERINTENSE:
                        ax.axhline(
                            SYM_H_THRESHOLD_SUPERINTENSE,
                            linestyle="--",
                            color=COLOR_SUPERINTENSE,
                        )

    plt.setp(ax.spines.values(), lw=2, color="black", alpha=1)

    ax.xaxis_date()

    ax.xaxis.set_major_locator(
        mdates.DayLocator(interval=1)
    )  # Adjust interval as needed
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    # Adjust the tick parameters for better visibility
    ax.tick_params(axis="x", which="major", labelsize=14, width=2, length=10)
    ax.tick_params(axis="y", which="major", labelsize=14, width=2, length=10)

    # Ensure that the grid is enabled and properly configured
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.5)

    ax.set_xlim(df.index[0], df.index[-1])

    if plot_prediction_error:
        custom_lines = [
            plt.Line2D([0], [0], color=observed_color, alpha=0.8),
            plt.Line2D([0], [0], color=predicted_color, alpha=0.8),
            plt.Line2D([0], [0], color=prediction_error_color, alpha=0.5),
        ]
    else:
        custom_lines = [
            plt.Line2D([0], [0], color=observed_color),
            plt.Line2D([0], [0], color=predicted_color),
        ]

    handles = []
    labels = []

    if multiple_q:
        patch_qsmall = mpatches.Patch(
            color=small_quantile_color, label=f"50% confidence"
        )
        handles.append(patch_qsmall)

    patch_qbig = mpatches.Patch(color=big_quantile_color, label=f"90% confidence")
    handles.extend(custom_lines)

    handles.append(patch_qbig)

    labels.extend(["Observed SYM-H", "Predicted SYM-H"])
    if plot_prediction_error:
        labels.extend(["Prediction Error"])
    if multiple_q:
        labels.extend([p.get_label() for p in [patch_qsmall, patch_qbig]])
    else:
        labels.extend([p.get_label() for p in [patch_qbig]])

    leg = ax.legend(
        handles=handles,
        labels=labels,
        ncol=len(handles),
        fancybox=True,
        prop={"size": 14},
        bbox_to_anchor=(0.5, 1.25),
        loc="upper center",
        edgecolor="black",
    )
    leg.get_frame().set_alpha(None)
    leg.get_frame().set_facecolor((0, 0, 0, 0))
    return fig, ax


def plot_evaluation_bfe_quantile(
    df,
    title,
    ax=None,
    plot_sym_bars=False,
    xlabel_title=None,
    upscaling=10,
    false_color=[1.0, 0.5, 0.5, 1.0],  # Lighter red color
    true_color=[0.5, 1.0, 0.5, 1.0],  # Lighter green color
):
    df_plot = df.copy()
    original_column = df_plot.columns[0]
    predicted_column = df_plot.columns[1]
    quantile_column = df_plot.columns[2]
    df_plot["diff"] = df_plot[original_column] - df_plot[predicted_column]
    df_plot = df_plot.reset_index()
    df_plot = df_plot.sort_values(original_column)
    df_plot["diff_abs"] = np.abs(df_plot["diff"])

    min_index = rounddown(df_plot[original_column].min())
    max_index = roundup(df_plot[original_column].max())

    bins = np.arange(min_index - BINWIDTH, max_index + BINWIDTH, BINWIDTH)

    df_plot["index_bins"] = pd.cut(df_plot[original_column], bins=bins, right=False)
    df_plot["index_bins_left"] = df_plot["index_bins"].apply(lambda x: x.left)
    mean_value_to_plot = df_plot.groupby("index_bins_left", observed=True)[
        "diff_abs"
    ].mean()

    if ax is None:
        fig, (ax, ax_color) = plt.subplots(
            2, 1, figsize=(13, 6), gridspec_kw={"height_ratios": [25, 1], "hspace": 0}
        )

    ln1 = ax.plot(
        mean_value_to_plot.index,
        mean_value_to_plot.values,
        linestyle="-",
        label="Evaluated predictions",
    )
    ax.fill_between(
        mean_value_to_plot.index,
        mean_value_to_plot.values,
        0,
        alpha=0.2,
        color="blue",
    )

    ax.set_ylim(0, ax.get_ylim()[1])
    ax.tick_params(axis="y", which="major", labelsize=14, width=2, length=10)
    ax.set_xlim(mean_value_to_plot.index.min(), mean_value_to_plot.index.max())

    mean_value_to_plot.index = mean_value_to_plot.index.astype(int)
    bfe = mean_value_to_plot.mean()
    ax.set_title(f"{title} | BFE: {bfe:.3f}", fontsize=18)

    if plot_sym_bars:
        min_sym = rounddown(df_plot[original_column].min())
        if min_sym <= SYM_H_THRESHOLD_LOW:
            ax.axvline(SYM_H_THRESHOLD_LOW, linestyle="--", color=COLOR_LOW)
            if min_sym <= SYM_H_THRESHOLD_MODERATE:
                ax.axvline(
                    SYM_H_THRESHOLD_MODERATE,
                    linestyle="--",
                    color=COLOR_MODERATE,
                )
                if min_sym <= SYM_H_THRESHOLD_INTENSE:
                    ax.axvline(
                        SYM_H_THRESHOLD_INTENSE,
                        linestyle="--",
                        color=COLOR_INTENSE,
                    )
                    if min_sym <= SYM_H_THRESHOLD_SUPERINTENSE:
                        ax.axvline(
                            SYM_H_THRESHOLD_SUPERINTENSE,
                            linestyle="--",
                            color=COLOR_SUPERINTENSE,
                        )

    twin = ax.twinx()
    twin.hist(df_plot[original_column], bins=bins, alpha=0.15, color="green")
    twin.set_yscale("log")
    twin.grid(linestyle="--")
    ax.set_ylabel("Mean Absolute Difference (nT)", fontsize=18)
    twin.set_ylabel("Bin count", fontsize=18)
    twin.tick_params(axis="y", which="major", labelsize=14, width=2, length=10)
    twin.tick_params(axis="y", which="minor", width=1, length=5)
    # After plotting the main graph, add the gradient colormap below
    cmap_light = LinearSegmentedColormap.from_list(
        "rg_light", [false_color, true_color]
    )
    # Upscale bfe for a smoother gradient
    quantile_inside = df_plot.groupby("index_bins_left", observed=True)[
        quantile_column
    ].mean()
    quantile_inside.index = quantile_inside.index.astype(int)
    upscaled_index = np.linspace(
        quantile_inside.index.min(),
        quantile_inside.index.max(),
        len(quantile_inside) * upscaling,
    )
    upscaled_values = np.interp(upscaled_index, quantile_inside.index, quantile_inside)

    # Display the gradient using the new upscaled colors
    ax_color.imshow(
        [cmap_light(upscaled_values)],
        aspect="auto",
        extent=[upscaled_index.min(), upscaled_index.max(), 0, 1],
    )

    twin.grid(False)
    ax.grid(True)

    ax_color.set_yticks([])
    ax_color.set_xticks([])
    ax.tick_params(
        axis="x",
        which="both",
        labelsize=14,
        width=2,
        length=18,
        pad=8,
        direction="out",
    )
    ax_color.set_xlim(ax.get_xlim())  # Ensure alignment with the plot above
    if xlabel_title is None:
        ax.set_xlabel(f"{original_column}", fontsize=18)
        ax.xaxis.set_label_coords(0.5, -0.13)
    else:
        ax.set_xlabel(xlabel_title, fontsize=18)
        ax.xaxis.set_label_coords(0.5, -0.13)
    ax_color.grid(False)
    plt.setp(ax_color.spines.values(), lw=2, color="black", alpha=1)
    plt.setp(ax.spines.values(), lw=2, color="black", alpha=1)
    plt.setp(twin.spines.values(), lw=2, color="black", alpha=1)

    # where some data has already been plotted to ax
    handles, labels = ax.get_legend_handles_labels()

    # manually define a new patch
    patch_bin = mpatches.Patch(color="green", alpha=0.15, label="Bin count")
    patch_false = mpatches.Patch(color=false_color, alpha=1, label="Outside 90%")
    patch_true = mpatches.Patch(color=true_color, alpha=1, label="Inside 90%")
    # line = Line2D([0], [0], label='manual line', color='k')
    # handles is a list, so append manual patch
    handles.append(patch_bin)
    handles.append(patch_false)
    handles.append(patch_true)
    # plot the legend
    leg = ax.legend(
        handles=handles,
        ncol=len(handles),
        fancybox=True,
        prop={"size": 14},
        bbox_to_anchor=(0.5, 1.19),
        loc="upper center",
        edgecolor="black",
    )
    leg.get_frame().set_alpha(None)
    leg.get_frame().set_facecolor((0, 0, 0, 0))
    return fig, (ax, ax_color)


def plot_evaluation_quantile(
    df,
    title,
    ax=None,
    plot_sym_bars=False,
    xlabel_title=None,
    upscaling=10,
    false_color=[1.0, 0.5, 0.5, 1.0],  # Lighter red color
    true_color=[0.5, 1.0, 0.5, 1.0],  # Lighter green color
):
    df_plot = df.copy()
    original_column = df_plot.columns[0]
    quantile_column_start = df.columns[1]
    quantile_column_end = df.columns[2]

    df_plot["covered"] = (
        (df_plot[quantile_column_start] <= df_plot[original_column])
        & (df_plot[original_column] <= df_plot[quantile_column_end])
    ).astype(int)

    # Calculate Prediction Interval Coverage Probability (PICP)
    picp = df_plot["covered"].mean()

    # Calculate Prediction Interval Widths (W_i)
    df_plot["interval_width"] = (
        df_plot[quantile_column_end] - df_plot[quantile_column_start]
    )

    # Calculate Average Interval Width (W_bar)
    average_width = df_plot["interval_width"].mean()

    min_index = rounddown(df_plot[original_column].min())
    max_index = roundup(df_plot[original_column].max())

    bins = np.arange(min_index - BINWIDTH, max_index + BINWIDTH, BINWIDTH)

    df_plot["index_bins"] = pd.cut(df_plot[original_column], bins=bins, right=False)
    df_plot["index_bins_left"] = df_plot["index_bins"].apply(lambda x: x.left)
    mean_value_to_plot = df_plot.groupby("index_bins_left", observed=True)[
        "interval_width"
    ].mean()

    if ax is None:
        fig, (ax, ax_color) = plt.subplots(
            2, 1, figsize=(13, 6), gridspec_kw={"height_ratios": [25, 1], "hspace": 0}
        )

    ln1 = ax.plot(
        mean_value_to_plot.index,
        mean_value_to_plot.values,
        linestyle="-",
        label="Interval width",
    )
    ax.fill_between(
        mean_value_to_plot.index,
        mean_value_to_plot.values,
        0,
        alpha=0.2,
        color="blue",
    )

    ax.set_ylim(0, ax.get_ylim()[1])
    ax.tick_params(axis="y", which="major", labelsize=14, width=2, length=10)
    ax.set_xlim(mean_value_to_plot.index.min(), mean_value_to_plot.index.max())

    mean_value_to_plot.index = mean_value_to_plot.index.astype(int)
    binned_interval_width = mean_value_to_plot.mean()
    ax.set_title(
        f"{title} \nInterval coverage: {picp:.3f} | Average Interval Width: {average_width:.3f} | Binned Interval Width: {binned_interval_width:.3f}",
        fontsize=18,
    )

    if plot_sym_bars:
        min_sym = rounddown(df_plot[original_column].min())
        if min_sym <= SYM_H_THRESHOLD_LOW:
            ax.axvline(SYM_H_THRESHOLD_LOW, linestyle="--", color=COLOR_LOW)
            if min_sym <= SYM_H_THRESHOLD_MODERATE:
                ax.axvline(
                    SYM_H_THRESHOLD_MODERATE,
                    linestyle="--",
                    color=COLOR_MODERATE,
                )
                if min_sym <= SYM_H_THRESHOLD_INTENSE:
                    ax.axvline(
                        SYM_H_THRESHOLD_INTENSE,
                        linestyle="--",
                        color=COLOR_INTENSE,
                    )
                    if min_sym <= SYM_H_THRESHOLD_SUPERINTENSE:
                        ax.axvline(
                            SYM_H_THRESHOLD_SUPERINTENSE,
                            linestyle="--",
                            color=COLOR_SUPERINTENSE,
                        )

    twin = ax.twinx()
    twin.hist(df_plot[original_column], bins=bins, alpha=0.15, color="green")
    twin.set_yscale("log")
    twin.grid(linestyle="--")
    ax.set_ylabel("Interval width (nT)", fontsize=18)
    twin.set_ylabel("Bin count", fontsize=18)
    twin.tick_params(axis="y", which="major", labelsize=14, width=2, length=10)
    twin.tick_params(axis="y", which="minor", width=1, length=5)
    # After plotting the main graph, add the gradient colormap below
    cmap_light = LinearSegmentedColormap.from_list(
        "rg_light", [false_color, true_color]
    )
    # Upscale bfe for a smoother gradient
    quantile_inside = df_plot.groupby("index_bins_left", observed=True)[
        "covered"
    ].mean()
    quantile_inside.index = quantile_inside.index.astype(int)
    upscaled_index = np.linspace(
        quantile_inside.index.min(),
        quantile_inside.index.max(),
        len(quantile_inside) * upscaling,
    )
    upscaled_values = np.interp(upscaled_index, quantile_inside.index, quantile_inside)

    # Display the gradient using the new upscaled colors
    ax_color.imshow(
        [cmap_light(upscaled_values)],
        aspect="auto",
        extent=[upscaled_index.min(), upscaled_index.max(), 0, 1],
    )

    twin.grid(False)
    ax.grid(True)

    ax_color.set_yticks([])
    ax_color.set_xticks([])
    ax.tick_params(
        axis="x",
        which="both",
        labelsize=14,
        width=2,
        length=18,
        pad=8,
        direction="out",
    )
    ax_color.set_xlim(ax.get_xlim())  # Ensure alignment with the plot above
    if xlabel_title is None:
        ax.set_xlabel(f"{original_column}", fontsize=18)
        ax.xaxis.set_label_coords(0.5, -0.13)
    else:
        ax.set_xlabel(xlabel_title, fontsize=18)
        ax.xaxis.set_label_coords(0.5, -0.13)
    ax_color.grid(False)
    plt.setp(ax_color.spines.values(), lw=2, color="black", alpha=1)
    plt.setp(ax.spines.values(), lw=2, color="black", alpha=1)
    plt.setp(twin.spines.values(), lw=2, color="black", alpha=1)

    # where some data has already been plotted to ax
    handles, labels = ax.get_legend_handles_labels()

    # manually define a new patch
    patch_bin = mpatches.Patch(color="green", alpha=0.15, label="Bin count")
    patch_false = mpatches.Patch(color=false_color, alpha=1, label="Outside 90%")
    patch_true = mpatches.Patch(color=true_color, alpha=1, label="Inside 90%")
    # line = Line2D([0], [0], label='manual line', color='k')
    # handles is a list, so append manual patch
    handles.append(patch_bin)
    handles.append(patch_false)
    handles.append(patch_true)
    # plot the legend
    leg = ax.legend(
        handles=handles,
        ncol=len(handles),
        fancybox=True,
        prop={"size": 14},
        bbox_to_anchor=(0.5, 1.25),
        loc="upper center",
        edgecolor="black",
    )
    leg.get_frame().set_alpha(None)
    leg.get_frame().set_facecolor((0, 0, 0, 0))
    return fig, (ax, ax_color)
