import pandas as pd
import re
import matplotlib.pyplot as plt

# === INPUT FILE ===
timespan = "2012-01-01 00:00:00_2012-12-31 00:00:00"
sim_type = ""  # "", "_parm_err" or "_cloudy"
csv_file = f"/home/c707/c7071034/Github/WRF_VPRM_post/plots/Validation_FLUXNET_hourly_all{sim_type}_{timespan}.csv"  # <- put your CSV filename here
output_file = f"/home/c707/c7071034/Github/WRF_VPRM_post/plots/latex_table_{sim_type}_{timespan}.txt"

# === Load CSV ===
df = pd.read_csv(csv_file)

# Extract values
rmse = df[df.iloc[:, 0] == "RMSE"].iloc[:, 1:].values.flatten()
r2 = df[df.iloc[:, 0] == "R2"].iloc[:, 1:].values.flatten()
cols = df.columns[1:]  # skip "RMSE"/"R2"

# === Parse column names ===
records = []
for col, rmse_val, r2_val in zip(cols, rmse, r2):
    if col.startswith(("t2m", "fco2")):
        site = col.split("_")[0]
        product = "OBS"
        var = col.split("_")[1].upper()
        res = "1"
    else:
        m = re.match(
            r"(?P<site>[A-Z]{2}-[A-Za-z0-9]+)_(?P<product>ALPS|REF|SITE)_(?P<var>[A-Z0-9]+)_WRF_(?P<res>\d+)km",
            col,
        )
        if not m:
            continue
        site, product, var, res = m.groups()
    records.append((site, var, product, res, float(rmse_val), float(r2_val)))

data = pd.DataFrame(records, columns=["site", "var", "product", "res", "RMSE", "R2"])

# Keep only 1 km resolution
data = data[data["res"] == "1"]

# === Build LaTeX table ===
variables = ["T2", "NEE", "GPP", "RECO"]
products = ["SITE", "ALPS", "REF"]

lines = []
header = "\\begin{tabular}{l" + "c" * len(variables) * len(products) + "}\n\\hline"
header += "\nSite"
for var in variables:
    for prod in products:
        header += f" & {var} {prod}"
header += " \\\\\n\\hline"
lines.append(header)

for site in sorted(data["site"].unique()):
    row = site
    for var in variables:
        for prod in products:
            sub = data[
                (data["site"] == site)
                & (data["var"] == var)
                & (data["product"] == prod)
            ]
            if not sub.empty:
                val = f"{sub['RMSE'].values[0]:.2f} ({sub['R2'].values[0]:.2f})"
            else:
                val = "-"
            row += " & " + val
    row += " \\\\"
    lines.append(row)

lines.append("\\hline\n\\end{tabular}")

with open(output_file, "w") as f:
    f.write("\n".join(lines))

print(f"LaTeX site-level 1km table written to {output_file}")

# === Plot grouped bar charts ===
sites = sorted(data["site"].unique())
colors = plt.cm.tab10.colors  # distinct colors

fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

# RMSE plot
width = 0.8 / len(sites)
x = range(len(variables))

for i, site in enumerate(sites):
    vals = []
    for var in variables:
        sub = data[(data["site"] == site) & (data["var"] == var)]
        vals.append(sub["RMSE"].mean() if not sub.empty else 0)
    ax[0].bar(
        [p + i * width for p in x],
        vals,
        width=width,
        color=colors[i % len(colors)],
        label=site,
    )

ax[0].set_xticks([p + 0.4 for p in x])
ax[0].set_xticklabels(variables)
ax[0].set_ylabel("RMSE")
ax[0].set_title("RMSE at 1 km across sites")
ax[0].legend()

# R² plot
for i, site in enumerate(sites):
    vals = []
    for var in variables:
        sub = data[(data["site"] == site) & (data["var"] == var)]
        vals.append(sub["R2"].mean() if not sub.empty else 0)
    ax[1].bar(
        [p + i * width for p in x],
        vals,
        width=width,
        color=colors[i % len(colors)],
        label=site,
    )

ax[1].set_xticks([p + 0.4 for p in x])
ax[1].set_xticklabels(variables)
ax[1].set_ylabel("R²")
ax[1].set_title("R² at 1 km across sites")
ax[1].legend()

plt.tight_layout()
plt.savefig(
    f"/home/c707/c7071034/Github/WRF_VPRM_post/plots/site_comparison_1km{sim_type}.pdf",
    dpi=300,
)
plt.show()
