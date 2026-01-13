import matplotlib

# use non-interactive backend for headless environments
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

########### Settings ###########
csv_folder = "/scratch/c7071034/DATA/WRFOUT/csv/"
outfolder = "/home/c707/c7071034/Github/WRF_VPRM_post/plots/"
start_date = "2012-01-01 00:00:00"
end_date = "2012-12-31 00:00:00"
ref_tag = "_1km"
sim_type = "_cloudy"  # "", "_parm_err" or "_cloudy"
plot_cloudy_and_clear = True  # If True, plots both cloudy and clear simulations
STD_TOPO = 200
resolutions = ["54km", "9km"]
variable_groups = {
    "dT": "ΔT [°C]",
    "dGPP": r"ΔGPP [$\mu$mol m$^{-2}$ s$^{-1}$]",
    "dRECO": r"ΔR$_\text{eco}$ [$\mu$mol m$^{-2}$ s$^{-1}$]",
}
###############################42
if plot_cloudy_and_clear:
    input_file = os.path.join(
        csv_folder, f"dPdT_timeseries_{start_date}_{end_date}{ref_tag}.csv"
    )
    ref_sim = "_REF"
    input_file_REF = os.path.join(
        csv_folder,
        f"dPdT_timeseries_{start_date}_{end_date}{ref_tag}{ref_sim}.csv",
    )
    # Read data
    df = pd.read_csv(input_file, index_col="datetime", parse_dates=True)
    # Read REF data
    df_ref = pd.read_csv(input_file_REF, index_col="datetime", parse_dates=True)
    input_file_cloudy = os.path.join(
        csv_folder, f"dPdT_timeseries_{start_date}_{end_date}{ref_tag}_cloudy.csv"
    )
    ref_sim = "_REF"
    input_file_REF_cloudy = os.path.join(
        csv_folder,
        f"dPdT_timeseries_{start_date}_{end_date}{ref_tag}{ref_sim}_cloudy.csv",
    )
    # Read data
    df2 = pd.read_csv(input_file_cloudy, index_col="datetime", parse_dates=True)
    # Read REF data
    df_ref2 = pd.read_csv(input_file_REF_cloudy, index_col="datetime", parse_dates=True)
    # merge the dfs
    df = pd.concat([df, df2])
    df_ref = pd.concat([df_ref, df_ref2])
    # sort by date
    df = df.sort_index()
    df_ref = df_ref.sort_index()
else:
    input_file = os.path.join(
        csv_folder, f"dPdT_timeseries_{start_date}_{end_date}{ref_tag}{sim_type}.csv"
    )
    ref_sim = "_REF"
    input_file_REF = os.path.join(
        csv_folder,
        f"dPdT_timeseries_{start_date}_{end_date}{ref_tag}{ref_sim}{sim_type}.csv",
    )
    # Read data
    df = pd.read_csv(input_file, index_col="datetime", parse_dates=True)
    # Read REF data
    df_ref = pd.read_csv(input_file_REF, index_col="datetime", parse_dates=True)


resolution_colors = {
    "54km": "red",
    "9km": "blue",
}
resolution_colors_light = {
    "54km": mcolors.to_rgba("red", alpha=0.4),
    "9km": mcolors.to_rgba("blue", alpha=0.4),
}


# --- Refactored Plot Function ---
def plot_combined(df, df_ref, variable_groups):
    for var_prefix, ylabel in variable_groups.items():
        plt.figure(figsize=(10, 6))
        label_shown = set()

        if var_prefix == "dGPP":
            type_x = "GPP"
        else:
            type_x = r"R$_{eco}$"

        for res in resolutions:
            columns = [
                col
                for col in df.columns
                if col.startswith(var_prefix + "_") and col.endswith("_" + res)
            ]
            ref_columns = [
                col
                for col in df_ref.columns
                if col.startswith(var_prefix + "_") and col.endswith("_" + res)
            ]

            for column in columns[::-1]:
                data_series = df[column].dropna()
                grouped = data_series.groupby(data_series.index.hour)
                hours = np.array(sorted(grouped.groups.keys()))
                avg_values = [grouped.get_group(h).mean() for h in hours]

                if "model" in column:
                    type_key = r"$\Delta_{\partial\text{T}}$"
                    color = resolution_colors_light.get(res, "gray")
                    if var_prefix == "dT":
                        type_key = f"{var_prefix}"
                        color = resolution_colors.get(res, "gray")
                elif "real" in column:
                    type_key = r"$\Delta_\text{res}$"
                    color = resolution_colors.get(res, "gray")
                else:
                    continue

                label = rf"{type_key}{type_x} ({res}, ALPS)"
                show_label = label not in label_shown

                plt.plot(
                    hours,
                    avg_values,
                    label=label if show_label else None,
                    color=color,
                )
                label_shown.add(label)

            for column in ref_columns[::-1]:
                data_series = df_ref[column].dropna()
                grouped = data_series.groupby(data_series.index.hour)
                hours = np.array(sorted(grouped.groups.keys()))
                avg_values = [grouped.get_group(h).mean() for h in hours]

                if "model" in column:
                    type_key = r"$\Delta_{\partial\text{T}}$"
                    color = resolution_colors_light.get(res, "gray")
                    if var_prefix == "dT":
                        type_key = f"{var_prefix}"
                        color = resolution_colors.get(res, "gray")
                elif "real" in column:
                    type_key = r"$\Delta_\text{res}$"
                    color = resolution_colors.get(res, "gray")
                else:
                    continue

                label = rf"{type_key}{type_x} ({res}, REF)"
                show_label = label not in label_shown

                plt.plot(
                    hours,
                    avg_values,
                    linestyle="dotted",
                    label=label if show_label else None,
                    color=color,
                )
                label_shown.add(label)

        plt.xticks(hours)
        plt.xlabel("UTC [h]", fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        plt.tick_params(labelsize=16)
        plt.grid(True)
        if var_prefix:
            plt.legend(loc="lower right", fontsize=20, frameon=True, framealpha=0.4)
        else:
            plt.legend(fontsize=20, frameon=True, framealpha=0.4)
        plt.tight_layout()
        if plot_cloudy_and_clear:
            sim_type = "all"
        plotname = f"{outfolder}hourly_avg_{var_prefix}_combined_std_topo_{STD_TOPO}{sim_type}_{start_date}_{end_date}.pdf"
        print(f"saved plot: {plotname}")
        # plt.show()
        plt.savefig(plotname, bbox_inches="tight")
        plt.close()


def plot_gpp_percent_explained(
    df, df_ref, variable_groups, resolutions, outfolder, STD_TOPO, start_date, end_date
):
    for var_prefix, ylabel in variable_groups.items():
        plt.figure(figsize=(10, 6))
        width = 0.4
        hours = np.arange(24)

        for i, res in enumerate(resolutions):
            col_real = f"{var_prefix}_real_mean_{res}"
            col_model = f"{var_prefix}_model_mean_{res}"
            col_real_ref = f"{var_prefix}_real_mean_{res}"
            col_model_ref = f"{var_prefix}_model_mean_{res}"

            if col_real not in df.columns or col_model not in df.columns:
                print(f"Missing OPT columns for {res}. Skipping...")
                continue

            if (
                col_real_ref not in df_ref.columns
                or col_model_ref not in df_ref.columns
            ):
                print(f"Missing REF columns for {res}. Skipping...")
                continue

            # --- OPT ---
            real_series = df[col_real].dropna()
            model_series = df[col_model].dropna()
            common_index = real_series.index.intersection(model_series.index)
            real_series = real_series.loc[common_index]
            model_series = model_series.loc[common_index]
            real_hourly = real_series.groupby(real_series.index.hour).mean()
            model_hourly = model_series.groupby(model_series.index.hour).mean()
            percent_explained = (model_hourly / real_hourly) * 100
            mean_percent = percent_explained.mean()
            plt.bar(
                hours + i * width - width / 2,
                percent_explained,
                width=width,
                color=resolution_colors.get(res, "gray"),
                alpha=0.7,
                label=f"{res} OPT (avg: {mean_percent:.1f}%)",
            )

            # --- REF ---
            real_series_ref = df_ref[col_real_ref].dropna()
            model_series_ref = df_ref[col_model_ref].dropna()
            common_index_ref = real_series_ref.index.intersection(
                model_series_ref.index
            )
            real_series_ref = real_series_ref.loc[common_index_ref]
            model_series_ref = model_series_ref.loc[common_index_ref]
            real_hourly_ref = real_series_ref.groupby(real_series_ref.index.hour).mean()
            model_hourly_ref = model_series_ref.groupby(
                model_series_ref.index.hour
            ).mean()
            percent_explained_ref = (model_hourly_ref / real_hourly_ref) * 100
            mean_percent_ref = percent_explained_ref.mean()
            plt.bar(
                hours + i * width - width / 2 + width / 2,
                percent_explained_ref,
                width=width,
                color=resolution_colors.get(res, "gray"),
                alpha=0.4,
                label=f"{res} REF (avg: {mean_percent_ref:.1f}%)",
            )

        plt.axhline(100, color="black", linestyle="--", linewidth=1)
        plt.xticks(hours)
        plt.tick_params(labelsize=16)
        plt.xlabel("UTC [h]")
        plt.ylabel(f"{ylabel} explained by model [%]")
        plt.ylim(-100, 100)
        plt.legend(frameon=True, framealpha=0.4)
        plt.grid(True)
        plt.tight_layout()
        if plot_cloudy_and_clear:
            sim_type = "all"
        outpath = os.path.join(
            outfolder,
            f"hourly_percent_{var_prefix}_model_real_std_topo_{STD_TOPO}{sim_type}_{start_date}_{end_date}.pdf",
        )
        print(f"saved percent GPP explained plot: {outpath}")
        plt.savefig(outpath, bbox_inches="tight")
        plt.close()


plot_combined(df, df_ref, variable_groups)
variable_subset = {
    "dGPP": "ΔGPP ",
    "dRECO": "ΔRECO ",
}
plot_gpp_percent_explained(
    df, df_ref, variable_subset, resolutions, outfolder, STD_TOPO, start_date, end_date
)
