"""
Fig5_WRFout_hourly_means_and_timeseries.py

Refactored (minimal changes) from `plot_wrf_mean_timeseries_OPT_REF.py`.
- Kept original functions and logic intact.
- Removed duplicate imports and stray tokens.
- Use non-interactive Matplotlib backend (`Agg`) for headless runs.
"""

from __future__ import annotations

import matplotlib

# use non-interactive backend for headless environments
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# Variable labels for scientific LaTeX-style plotting
var_labels = {
    "GPP": "GPP",
    "RECO": r"R$_{\text{eco}}$",
    "NEE": "NEE",
    "T2": r"T$_{\text{2m}}$",
    "SWDOWN": r"S$_\downarrow$",
}


def compute_nee(df: pd.DataFrame, resolutions: list) -> None:
    for res in resolutions:
        df[f"NEE_{res}"] = -df[f"GPP_{res}"] + df[f"RECO_{res}"]


def preprocess_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df["datetime"] = pd.to_datetime(df["Unnamed: 0"], format="%Y-%m-%d %H:%M:%S")
    df.set_index("datetime", inplace=True)
    df["hour"] = df.index.hour
    return df


def group_hourly_average(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = df.select_dtypes(include=["number"]).columns
    return df[numeric_columns].groupby("hour").mean()


def plot_timeseries_differences(
    df: pd.DataFrame,
    df_ref: pd.DataFrame,
    column: str,
    unit: str,
    resolutions: list,
    outfolder: str,
    ref_sim: bool,
    resolution_colors: dict,
    STD_TOPO: int,
    start_date: str,
    end_date: str,
    sim_type: str,
):
    plt.figure(figsize=(15, 7))
    xticks, xticklabels = [], []

    for res in resolutions:
        if res in ["CAMS", "1km"]:
            continue

        color = resolution_colors[res]
        series_col = f"{column}_{res}"
        baseline_col = f"{column}_1km"

        if series_col not in df or baseline_col not in df:
            continue

        # compute difference wrt 1 km
        diff = (df[series_col] - df[baseline_col]).dropna()
        grouped = diff.groupby(diff.index.date)
        valid_days = [
            (date, group) for date, group in grouped if not group.dropna().eq(0).all()
        ]

        current_x = 0
        for i, (date, group) in enumerate(valid_days):
            x = np.arange(len(group)) + current_x
            plt.plot(x, group.values, linestyle="-", linewidth=1.5, color=color)
            # collect xticks only once
            if res == resolutions[0]:
                xticks.append(current_x)
                xticklabels.append(str(date))
            # gray separator between days
            if i < len(valid_days) - 1:
                gap = len(group) + 1
                plt.axvspan(
                    current_x + len(group),
                    current_x + gap,
                    color="lightgray",
                    alpha=0.5,
                )
            current_x += len(group) + 1

        plt.plot(
            [],
            [],
            label=f"{var_labels[column]} ({res}-1km, ALPS)",
            linestyle="-",
            color=color,
        )

        # ----- reference simulation -----
        if ref_sim:
            if series_col in df_ref and baseline_col in df_ref:
                diff_ref = (df_ref[series_col] - df_ref[baseline_col]).dropna()
                grouped_ref = diff_ref.groupby(diff_ref.index.date)
                valid_days_ref = [
                    (date, group)
                    for date, group in grouped_ref
                    if not group.dropna().eq(0).all()
                ]

                current_x = 0
                for i, (date, group) in enumerate(valid_days_ref):
                    x = np.arange(len(group)) + current_x
                    plt.plot(
                        x, group.values, linestyle="--", linewidth=1.5, color=color
                    )
                    current_x += len(group) + 1
                plt.plot(
                    [],
                    [],
                    label=f"{var_labels[column]} ({res}-1km, REF)",
                    linestyle="--",
                    color=color,
                )

    # identical axis formatting as in plot_timeseries_by_resolution
    plt.xticks(xticks, xticklabels, ha="left")
    plt.xlabel("Date", fontsize=20)
    plt.ylabel(r"$\Delta_\text{res}$" + f"{var_labels[column]} {unit}", fontsize=20)
    plt.tick_params(axis="x", labelsize=16, labelrotation=90)
    plt.tick_params(axis="y", labelsize=18)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=20)
    plt.tight_layout()

    plt.savefig(
        f"{outfolder}timeseries_diff_of_resolutions_{column}_domain_averaged_std_topo_{STD_TOPO}{sim_type}_{start_date}_{end_date}.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_timeseries_by_resolution(
    df: pd.DataFrame,
    df_ref: pd.DataFrame,
    column: str,
    unit: str,
    resolutions: list,
    outfolder: str,
    ref_sim: bool,
    resolution_colors: dict,
    STD_TOPO: int,
    start_date: str,
    end_date: str,
    sim_type: str,
):
    plt.figure(figsize=(15, 7))
    xticks, xticklabels = [], []

    for res in resolutions:
        color = resolution_colors[res]
        series_col = f"{column}_{res}"

        if res == "CAMS":
            df[series_col] = df[series_col].resample("h").interpolate("linear")
            label_opt = f"{var_labels[column]} ({res})"
        else:
            label_opt = f"{var_labels[column]} ({res}, ALPS)"

        y = df[series_col].dropna()
        grouped = y.groupby(y.index.date)
        valid_days = [
            (date, group) for date, group in grouped if not group.dropna().eq(0).all()
        ]

        current_x = 0
        for i, (date, group) in enumerate(valid_days):
            x = np.arange(len(group)) + current_x
            plt.plot(x, group.values, linestyle="-", linewidth=1.5, color=color)
            if res == resolutions[0]:
                xticks.append(current_x)
                xticklabels.append(str(date))
            if i < len(valid_days) - 1:
                gap = len(group) + 1
                plt.axvspan(
                    current_x + len(group),
                    current_x + gap,
                    color="lightgray",
                    alpha=0.5,
                )
            current_x += len(group) + 1
        plt.plot([], [], label=label_opt, linestyle="-", color=color)

        y_ref = df_ref[series_col].dropna() if ref_sim and res != "CAMS" else None
        if res == "CAMS":
            df[series_col] = df[series_col].resample("h").interpolate("linear")
            label_opt = f"{var_labels[column]} ({res})"
        else:
            label_opt = f"{var_labels[column]} ({res}, REF)"
        if y_ref is not None:
            if column != "T2" and column != "SWDOWN":
                label_ref = f"{label_opt}"
                grouped_ref = y_ref.groupby(y_ref.index.date)
                valid_days_ref = [
                    (date, group)
                    for date, group in grouped_ref
                    if not group.dropna().eq(0).all()
                ]

                current_x = 0
                for i, (date, group) in enumerate(valid_days_ref):
                    x = np.arange(len(group)) + current_x
                    plt.plot(
                        x, group.values, linestyle="--", linewidth=1.5, color=color
                    )
                    current_x += len(group) + 1
                plt.plot([], [], label=label_ref, linestyle="--", color=color)

    plt.xticks(xticks, xticklabels, ha="left")
    plt.xlabel("Date", fontsize=20)
    plt.ylabel(f"{var_labels[column]} {unit}", fontsize=20)
    plt.tick_params(axis="x", labelsize=16, labelrotation=90)
    plt.tick_params(axis="y", labelsize=18)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(
        f"{outfolder}timeseries_{column}_domain_averaged_std_topo_{STD_TOPO}{sim_type}_{start_date}_{end_date}.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_hourly_averages(
    hourly_avg: pd.DataFrame,
    hourly_avg_ref: pd.DataFrame,
    column: str,
    unit: str,
    resolutions: list,
    outfolder: str,
    ref_sim: bool,
    resolution_colors: dict,
    STD_TOPO: int,
    start_date: str,
    end_date: str,
    sim_type: str,
):
    plt.figure(figsize=(10, 6))
    for res in resolutions:
        series = hourly_avg[f"{column}_{res}"].dropna()
        if res == "CAMS":
            label_opt = f"{var_labels[column]} ({res})"
        else:
            label_opt = f"{var_labels[column]} ({res}, ALPS)"
        plt.plot(
            series.index,
            series,
            label=label_opt,
            linestyle="-",
            color=resolution_colors[res],
        )

        if ref_sim and res != "CAMS":
            if column != "T2" and column != "SWDOWN":
                series_ref = hourly_avg_ref[f"{column}_{res}"].dropna()
                plt.plot(
                    series_ref.index,
                    series_ref,
                    label=f"{var_labels[column]} ({res}, REF)",
                    linestyle="--",
                    color=resolution_colors[res],
                )

    plt.xlabel("UTC [h]", fontsize=30)
    plt.ylabel(f"{var_labels[column]} {unit}", fontsize=30)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks([0, 6, 12, 18, 24])
    plt.tick_params(axis="x", labelsize=30)
    plt.tick_params(axis="y", labelsize=30)
    plt.legend(fontsize=24, frameon=True, framealpha=0.4)
    plt.tight_layout()
    plt.savefig(
        f"{outfolder}timeseries_hourly_{column}_domain_averaged_std_topo_{STD_TOPO}{sim_type}_{start_date}_{end_date}.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_hourly_differences(
    hourly_avg: pd.DataFrame,
    hourly_avg_ref: pd.DataFrame,
    column: str,
    unit: str,
    resolutions_diff: list,
    outfolder: str,
    ref_sim: bool,
    resolution_colors: dict,
    STD_TOPO: int,
    start_date: str,
    end_date: str,
    sim_type: str,
):
    plt.figure(figsize=(10, 6))
    for res in resolutions_diff + ["CAMS"]:
        if res == "CAMS":
            continue

        diff_opt = hourly_avg[f"{column}_{res}"] - hourly_avg[f"{column}_1km"]
        plt.plot(
            diff_opt.index,
            diff_opt,
            label=f"{var_labels[column]} ({res}-1km, ALPS)",
            linestyle="-",
            color=resolution_colors[res],
        )

        if ref_sim:
            diff_ref = (
                hourly_avg_ref[f"{column}_{res}"] - hourly_avg_ref[f"{column}_1km"]
            )
            plt.plot(
                diff_ref.index,
                diff_ref,
                label=f"{var_labels[column]} ({res}-1km, REF)",
                linestyle="--",
                color=resolution_colors[res],
            )

    plt.xlabel("UTC [h]", fontsize=30)
    plt.ylabel(r"$\Delta_\text{res}$" + f"{var_labels[column]} {unit}", fontsize=30)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=24, frameon=True, framealpha=0.4)
    plt.xticks([0, 6, 12, 18, 24])
    plt.tick_params(axis="x", labelsize=30)
    plt.tick_params(axis="y", labelsize=30)
    plt.tight_layout()
    plt.savefig(
        f"{outfolder}timeseries_hourly_diff_of_54km_{column}_domain_averaged_std_topo_{STD_TOPO}{sim_type}_{start_date}_{end_date}.pdf",
        bbox_inches="tight",
    )
    plt.close()


def compute_hourly_means_and_differences_reshaped(
    hourly_avg: pd.DataFrame,
    hourly_avg_ref: pd.DataFrame,
    columns: list,
    resolutions: list,
    ref_sim: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records_means = []
    records_diffs = []
    baseline = "1km"

    for col in columns:
        for res in resolutions:
            key = f"{col}_{res}"
            base_key = f"{col}_{baseline}"

            if key in hourly_avg.columns:
                mean_opt = hourly_avg[key].mean()
                records_means.append(
                    {
                        "variable": col,
                        "type": "ALPS",
                        "resolution": res,
                        "mean": mean_opt,
                    }
                )

                if res != baseline and base_key in hourly_avg.columns:
                    diff_opt = (hourly_avg[key] - hourly_avg[base_key]).mean()
                    records_diffs.append(
                        {
                            "variable": col,
                            "type": "ALPS_DIFF",
                            "resolution": f"{res}-{baseline}",
                            "mean": diff_opt,
                        }
                    )

            if ref_sim and res != "CAMS" and key in hourly_avg_ref.columns:
                mean_ref = hourly_avg_ref[key].mean()
                records_means.append(
                    {
                        "variable": col,
                        "type": "REF",
                        "resolution": res,
                        "mean": mean_ref,
                    }
                )

                if res != baseline and base_key in hourly_avg_ref.columns:
                    diff_ref = (hourly_avg_ref[key] - hourly_avg_ref[base_key]).mean()
                    records_diffs.append(
                        {
                            "variable": col,
                            "type": "REF_DIFF",
                            "resolution": f"{res}-{baseline}",
                            "mean": diff_ref,
                        }
                    )

    df_means = pd.DataFrame(records_means).pivot(
        index="variable", columns=["type", "resolution"], values="mean"
    )
    df_diffs = pd.DataFrame(records_diffs).pivot(
        index="variable", columns=["type", "resolution"], values="mean"
    )

    df_pct = pd.DataFrame()
    for col in df_diffs.columns:
        diff_type, diff_res = col
        if diff_type == "ALPS_DIFF":
            base_res = diff_res.split("-")[1]
            ref_col = ("ALPS", base_res)
        elif diff_type == "REF_DIFF":
            base_res = diff_res.split("-")[1]
            ref_col = ("REF", base_res)
        else:
            continue

        if ref_col in df_means.columns:
            pct_col = (f"{diff_type}_PCT", diff_res)
            df_pct[pct_col] = (df_diffs[col] / df_means[ref_col]) * 100

    return df_means, df_diffs, df_pct


def write_domain_average_table(
    means,
    diff_mean,
    diff_pct,
    outfile="domain_averaged_2012.tex",
):
    """
    Write domain-averaged LaTeX table using dataframes.

    Parameters:
    -----------
    means : DataFrame with MultiIndex columns (sim_type, resolution)
    diff_mean : DataFrame with tuple columns
    diff_pct : DataFrame with tuple columns
    """
    variables = [
        ("T2", r"$T_\text{2m}$ [\textdegree C]"),
        ("SWDOWN", r"S$\downarrow$ [W m$^{-2}$]"),
        ("GPP", r"GPP [$\mu$mol m$^{-2}$ s$^{-1}$]"),
        ("RECO", r"R$_{\text{eco}}$ [$\mu$mol m$^{-2}$ s$^{-1}$]"),
        ("NEE", r"NEE [$\mu$mol m$^{-2}$ s$^{-1}$]"),
    ]

    def fmt(val):
        return f"{val:.2f}"

    with open(outfile, "w") as f:
        f.write(
            "\\begin{table}\n"
            "\\centering\n"
            "\\scriptsize\n"
            "\\begin{tabular}{l|rr|rr|rr|r}\n"
            "\\toprule\n"
            "& \\multicolumn{2}{c|}{1km}"
            " & \\multicolumn{2}{c|}{9km}"
            " & \\multicolumn{2}{c|}{54km}"
            " & CAMS \\\\\n"
            "& ALPS & REF & ALPS & REF & ALPS & REF & \\\\\n"
            "\\midrule\n"
        )

        for var, label in variables:
            # Get 1km values from means
            alps_1km = means.loc[var, ("ALPS", "1km")]
            ref_1km = means.loc[var, ("REF", "1km")]

            row = f"{label} & {fmt(alps_1km)}"

            if var in ["GPP", "RECO", "NEE"]:
                pct = abs(ref_1km) / abs(alps_1km) * 100
                row += f" & {fmt(ref_1km)} [{pct:.0f}\\%]"
            else:
                row += f" & {fmt(ref_1km)}"

            nee_diffs = []

            # 9km and 54km columns
            for res in ["9km", "54km"]:
                for sim in ["ALPS", "REF"]:
                    val = means.loc[var, (sim, res)]

                    # Access using the tuple column directly
                    dval = diff_mean.loc[var, (f"{sim}_DIFF", f"{res}-1km")]
                    dpct = diff_pct.loc[var, (f"{sim}_DIFF_PCT", f"{res}-1km")]

                    if var == "NEE":
                        nee_diffs.append((res, sim, abs(dval), abs(dpct)))

                    row += f" & {fmt(val)} ({fmt(dval)} [{abs(dpct):.0f}\\%])"

            # CAMS column
            cams_val = means.loc[var, ("ALPS", "CAMS")]
            dval = diff_mean.loc[var, ("ALPS_DIFF", "CAMS-1km")]
            dpct = diff_pct.loc[var, ("ALPS_DIFF_PCT", "CAMS-1km")]
            row += f" & {fmt(cams_val)} ({fmt(dval)} [{abs(dpct):.0f}\\%]) \\\\\n"

            # Apply bolding for NEE max differences
            if var == "NEE" and nee_diffs:
                max_abs = max(v[2] for v in nee_diffs)
                max_pct = max(v[3] for v in nee_diffs)
                # Find and bold the matching value
                for res, sim, abs_val, pct_val in nee_diffs:
                    if abs_val == max_abs and pct_val == max_pct:
                        old_str = f"({fmt(abs_val)} [{pct_val:.0f}\\%])"
                        new_str = f"\\textbf{{({fmt(abs_val)} [{pct_val:.0f}\\%])}}"
                        row = row.replace(old_str, new_str, 1)
                        break

            f.write(row)

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"LaTeX table written to {outfile}")


def main():
    csv_folder = "/scratch/c7071034/DATA/WRFOUT/csv/"
    outfolder = "/home/c707/c7071034/Github/WRF_VPRM_post/plots/"
    start_date, end_date = "2012-01-01 00:00:00", "2012-12-31 00:00:00"
    STD_TOPO = 200
    ref_sim = True
    sim_type = ""  # "" or "_cloudy"

    columns = ["GPP", "RECO", "NEE", "T2", "SWDOWN"]
    units = [
        " [µmol m² s⁻¹]",
        " [µmol m² s⁻¹]",
        " [µmol m² s⁻¹]",
        " [°C]",
        r" [$\frac{\text{W}}{\text{m}^2}$]",
    ]
    resolutions = ["1km", "9km", "54km", "CAMS"]
    resolutions_diff = ["54km", "9km"]

    merged_df_gt = pd.read_csv(
        f"{csv_folder}timeseries_domain_averaged_std_topo_gt_{STD_TOPO}{sim_type}_{start_date}_{end_date}.csv"
    )
    if ref_sim:
        merged_df_gt_ref = pd.read_csv(
            f"{csv_folder}timeseries_domain_averaged_REF_std_topo_gt_{STD_TOPO}{sim_type}_{start_date}_{end_date}.csv"
        )

    compute_nee(merged_df_gt, resolutions)
    compute_nee(merged_df_gt_ref, resolutions) if ref_sim else None
    merged_df_gt = preprocess_datetime(merged_df_gt)
    merged_df_gt_ref = preprocess_datetime(merged_df_gt_ref) if ref_sim else None

    hourly_avg = group_hourly_average(merged_df_gt)
    hourly_avg_ref = group_hourly_average(merged_df_gt_ref) if ref_sim else None

    df_means, df_diffs, df_pct = compute_hourly_means_and_differences_reshaped(
        hourly_avg, hourly_avg_ref, columns, resolutions, ref_sim
    )

    # Convert columns to proper MultiIndex
    df_diffs.columns = pd.MultiIndex.from_tuples(df_diffs.columns)
    df_pct.columns = pd.MultiIndex.from_tuples(df_pct.columns)

    write_domain_average_table(
        df_means,
        df_diffs,
        df_pct,
        outfile=f"{outfolder}Table_1_domain_averaged_2012{sim_type}.tex",
    )

    resolution_colors = {
        "1km": "black",
        "9km": "blue",
        "54km": "red",
        "CAMS": "orange",
    }

    for column, unit in zip(columns, units):
        plot_timeseries_by_resolution(
            merged_df_gt,
            merged_df_gt_ref,
            column,
            unit,
            resolutions,
            outfolder,
            ref_sim,
            resolution_colors,
            STD_TOPO,
            start_date,
            end_date,
            sim_type,
        )
        plot_timeseries_differences(
            merged_df_gt,
            merged_df_gt_ref,
            column,
            unit,
            resolutions,
            outfolder,
            ref_sim,
            resolution_colors,
            STD_TOPO,
            start_date,
            end_date,
            sim_type,
        )
        plot_hourly_averages(
            hourly_avg,
            hourly_avg_ref,
            column,
            unit,
            resolutions,
            outfolder,
            ref_sim,
            resolution_colors,
            STD_TOPO,
            start_date,
            end_date,
            sim_type,
        )
        plot_hourly_differences(
            hourly_avg,
            hourly_avg_ref,
            column,
            unit,
            resolutions_diff,
            outfolder,
            ref_sim,
            resolution_colors,
            STD_TOPO,
            start_date,
            end_date,
            sim_type,
        )


if __name__ == "__main__":
    main()
