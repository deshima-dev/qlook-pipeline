# standard libraries
import warnings
import argparse
import pathlib
import yaml

# dependent packages
import decode as dc
import numpy as np
from scipy.signal import argrelmax, argrelmin
import matplotlib.pyplot as plt
from astropy import table
from astropy.modeling import models, fitting

# original package
from utils import functions as fc

# module settings
warnings.filterwarnings("ignore")
plt.style.use("seaborn-darkgrid")
plt.style.use("seaborn-muted")

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("dfits_file", help="DFITS name")
parser.add_argument("antlog_file", help="antenna log")
parser.add_argument("yaml_file", help="parameter file")
args = parser.parse_args()

dfits_file = pathlib.Path(args.dfits_file)
antlog_file = pathlib.Path(args.antlog_file)
yaml_file = pathlib.Path(args.yaml_file)
with open(yaml_file) as file:
    params = yaml.load(file, Loader=yaml.SafeLoader)

# directory settings
obsid = dfits_file.name.split("_")[1].split(".")[0]
output_dir = pathlib.Path(params["file"]["output_dir"]) / obsid
if not output_dir.exists():
    output_dir.mkdir(parents=True)
result_file = output_dir / params["file"]["result_file"]
image_format = params["file"]["image_format"]
do_plot = params["file"]["do_plot"]
dpi = params["file"]["dpi"]

# fc.loaddfits parameters
ch = params["loaddfits"]["ch"]
array = fc.loaddfits(dfits_file, antlog_file, **params["loaddfits"])

# 1st step: check time stream
print("#1: check time stream")

scantypes = np.unique(array.scantype)
print(f"scantypes: {scantypes}")

fig, ax = plt.subplots(2, 1, figsize=(10, 5), dpi=dpi)
tstart0 = params["check_scantypes"]["tstart0"]
tend0 = params["check_scantypes"]["tend0"]
tstart1 = params["check_scantypes"]["tstart1"]
tend1 = params["check_scantypes"]["tend1"]
subarray0 = array[tstart0:tend0, :]
subarray1 = array[tstart1:tend1, :]

refch = params["check_scantypes"]["refch"]
plot_params = {"marker": ".", "markersize": 0.5, "linestyle": "None"}

dc.plot.plot_timestream(subarray0, ch, scantypes=["GRAD"], ax=ax[0], **plot_params)
dc.plot.plot_timestream(subarray0, ch, scantypes=["ON"], ax=ax[0], **plot_params)
dc.plot.plot_timestream(subarray1, ch, scantypes=["ON"], ax=ax[1], **plot_params)

fig.tight_layout()
fig.savefig(output_dir / f"time_stream.{image_format}")
if do_plot:
    plt.show()
else:
    plt.clf()
    plt.close()

# 2nd step: plot subref offsets vs time
print("#2: plot subref offsets vs time")

fig, ax = plt.subplots(3, 1, figsize=(10, 7), dpi=dpi)
dc.plot.plot_tcoords(array, ("time", "subref_x"), scantypes=["ON"], ax=ax[0])
dc.plot.plot_tcoords(array, ("time", "subref_y"), scantypes=["ON"], ax=ax[1])
dc.plot.plot_tcoords(array, ("time", "subref_z"), scantypes=["ON"], ax=ax[2])

maxid = list(argrelmax(array.subref_z[array.scantype == "ON"].values, order=1)[0])
minid = list(argrelmin(array.subref_z[array.scantype == "ON"].values, order=1)[0])
ax[2].plot(
    array.time[array.scantype == "ON"][maxid],
    array.subref_z[array.scantype == "ON"][maxid],
    "o",
    color="C1",
    label="local max",
)
ax[2].plot(
    array.time[array.scantype == "ON"][minid],
    array.subref_z[array.scantype == "ON"][minid],
    "o",
    color="C2",
    label="local min",
)
ax[2].legend()

fig.tight_layout()
fig.savefig(output_dir / f"subref_movement.{image_format}")
if do_plot:
    plt.show()
else:
    plt.clf()
    plt.close()

# 3rd step: plot temperature vs subref_z
print("#3: plot temperature vs subref_z")

fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=dpi)
ax.plot(
    array.subref_z[array.scantype == "ON"],
    array[:, ch][array.scantype == "ON"],
    label="ON",
)
ax.set_xlabel("subref_z")
ax.set_ylabel("temperature")
ax.set_title(f"ch #{ch}")
ax.legend()

fig.tight_layout()
fig.savefig(output_dir / f"temp_vs_subrefz.{image_format}")
if do_plot:
    plt.show()
else:
    plt.clf()
    plt.close()

# 4th step: Gauss-fit
print("#4: Gauss-fit")

alldata = table.QTable(
    names=("scan_speed", "peak", "z_mean", "z_stddev", "slope", "intercept")
)

if maxid[0] < minid[0]:
    minid.insert(0, np.nan)
if minid[-1] < maxid[-1]:
    minid.append(np.nan)

amp0 = params["fitting"]["amplitude"]
z0 = params["fitting"]["z_mean"]
s0 = params["fitting"]["z_stddev"]
sl = params["fitting"]["slope"]
ic = params["fitting"]["intercept"]

g_init = models.Gaussian1D(amplitude=amp0, mean=z0, stddev=s0) + models.Linear1D(sl, ic)
fit_g = fitting.LevMarLSQFitter()
for n in range(len(maxid)):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    if n == 0:
        if not np.isnan(minid[n]):
            ax.plot(
                array.subref_z[array.scantype == "ON"][minid[n] : maxid[n]],
                array[:, refch][array.scantype == "ON"][minid[n] : maxid[n]],
                color="C0",
                label="obs (dZ > 0)",
            )
            g = fit_g(
                g_init,
                array.subref_z[array.scantype == "ON"][minid[n] : maxid[n]],
                array[:, refch][array.scantype == "ON"][minid[n] : maxid[n]],
            )
            ax.plot(
                array.subref_z[array.scantype == "ON"][minid[n] : maxid[n]],
                g(array.subref_z[array.scantype == "ON"][minid[n] : maxid[n]]),
                color="C2",
                label="model (dZ > 0)",
            )

            dt = (
                array.time[array.scantype == "ON"][maxid[n]]
                - array.time[array.scantype == "ON"][minid[n]]
            ).values.item() / 1e9
            ds = float(
                (
                    array.subref_z[array.scantype == "ON"][maxid[n]]
                    - array.subref_z[array.scantype == "ON"][minid[n]]
                ).values
            )
            ss = ds / dt
            alldata.add_row(
                (ss, g.amplitude_0, g.mean_0, g.stddev_0, g.slope_1, g.intercept_1)
            )

        ax.plot(
            array.subref_z[array.scantype == "ON"][maxid[n] : minid[n + 1]],
            array[:, refch][array.scantype == "ON"][maxid[n] : minid[n + 1]],
            color="C1",
            label="obs (dZ < 0)",
        )
        g = fit_g(
            g_init,
            array.subref_z[array.scantype == "ON"][maxid[n] : minid[n + 1]],
            array[:, refch][array.scantype == "ON"][maxid[n] : minid[n + 1]],
        )
        ax.plot(
            array.subref_z[array.scantype == "ON"][maxid[n] : minid[n + 1]],
            g(array.subref_z[array.scantype == "ON"][maxid[n] : minid[n + 1]]),
            color="C3",
            label="model (dZ < 0)",
        )

        dt = (
            array.time[array.scantype == "ON"][minid[n + 1]]
            - array.time[array.scantype == "ON"][maxid[n]]
        ).values.item() / 1e9
        ds = float(
            (
                array.subref_z[array.scantype == "ON"][minid[n + 1]]
                - array.subref_z[array.scantype == "ON"][maxid[n]]
            ).values
        )
        ss = ds / dt
        alldata.add_row(
            (ss, g.amplitude_0, g.mean_0, g.stddev_0, g.slope_1, g.intercept_1)
        )

    elif n == len(maxid) - 1:
        ax.plot(
            array.subref_z[array.scantype == "ON"][minid[n] : maxid[n]],
            array[:, refch][array.scantype == "ON"][minid[n] : maxid[n]],
            color="C0",
            label="obs (dZ > 0)",
        )
        g = fit_g(
            g_init,
            array.subref_z[array.scantype == "ON"][minid[n] : maxid[n]],
            array[:, refch][array.scantype == "ON"][minid[n] : maxid[n]],
        )
        ax.plot(
            array.subref_z[array.scantype == "ON"][minid[n] : maxid[n]],
            g(array.subref_z[array.scantype == "ON"][minid[n] : maxid[n]]),
            color="C2",
            label="model (dZ > 0)",
        )

        dt = (
            array.time[array.scantype == "ON"][maxid[n]]
            - array.time[array.scantype == "ON"][minid[n]]
        ).values.item() / 1e9
        ds = float(
            (
                array.subref_z[array.scantype == "ON"][maxid[n]]
                - array.subref_z[array.scantype == "ON"][minid[n]]
            ).values
        )
        ss = ds / dt
        alldata.add_row(
            (ss, g.amplitude_0, g.mean_0, g.stddev_0, g.slope_1, g.intercept_1)
        )

        if not np.isnan(minid[n + 1]):
            ax.plot(
                array.subref_z[array.scantype == "ON"][maxid[n] : minid[n + 1]],
                array[:, refch][array.scantype == "ON"][maxid[n] : minid[n + 1]],
                color="C1",
                label="obs (dZ < 0)",
            )
            g = fit_g(
                g_init,
                array.subref_z[array.scantype == "ON"][maxid[n] : minid[n + 1]],
                array[:, refch][array.scantype == "ON"][maxid[n] : minid[n + 1]],
            )
            ax.plot(
                array.subref_z[array.scantype == "ON"][maxid[n] : minid[n + 1]],
                g(array.subref_z[array.scantype == "ON"][maxid[n] : minid[n + 1]]),
                color="C3",
                label="model (dZ < 0)",
            )

            dt = (
                array.time[array.scantype == "ON"][minid[n + 1]]
                - array.time[array.scantype == "ON"][maxid[n]]
            ).values.item() / 1e9
            ds = float(
                (
                    array.subref_z[array.scantype == "ON"][minid[n + 1]]
                    - array.subref_z[array.scantype == "ON"][maxid[n]]
                ).values
            )
            ss = ds / dt
            alldata.add_row(
                (ss, g.amplitude_0, g.mean_0, g.stddev_0, g.slope_1, g.intercept_1)
            )
    else:
        ax.plot(
            array.subref_z[array.scantype == "ON"][minid[n] : maxid[n]],
            array[:, refch][array.scantype == "ON"][minid[n] : maxid[n]],
            color="C0",
            label="obs (dZ > 0)",
        )
        g = fit_g(
            g_init,
            array.subref_z[array.scantype == "ON"][minid[n] : maxid[n]],
            array[:, refch][array.scantype == "ON"][minid[n] : maxid[n]],
        )
        ax.plot(
            array.subref_z[array.scantype == "ON"][minid[n] : maxid[n]],
            g(array.subref_z[array.scantype == "ON"][minid[n] : maxid[n]]),
            color="C2",
            label="model (dZ > 0)",
        )

        dt = (
            array.time[array.scantype == "ON"][maxid[n]]
            - array.time[array.scantype == "ON"][minid[n]]
        ).values.item() / 1e9
        ds = float(
            (
                array.subref_z[array.scantype == "ON"][maxid[n]]
                - array.subref_z[array.scantype == "ON"][minid[n]]
            ).values
        )
        ss = ds / dt
        alldata.add_row(
            (ss, g.amplitude_0, g.mean_0, g.stddev_0, g.slope_1, g.intercept_1)
        )

        ax.plot(
            array.subref_z[array.scantype == "ON"][maxid[n] : minid[n + 1]],
            array[:, refch][array.scantype == "ON"][maxid[n] : minid[n + 1]],
            color="C1",
            label="obs (dZ < 0)",
        )
        g = fit_g(
            g_init,
            array.subref_z[array.scantype == "ON"][maxid[n] : minid[n + 1]],
            array[:, refch][array.scantype == "ON"][maxid[n] : minid[n + 1]],
        )
        ax.plot(
            array.subref_z[array.scantype == "ON"][maxid[n] : minid[n + 1]],
            g(array.subref_z[array.scantype == "ON"][maxid[n] : minid[n + 1]]),
            color="C3",
            label="model (dZ < 0)",
        )

        dt = (
            array.time[array.scantype == "ON"][minid[n + 1]]
            - array.time[array.scantype == "ON"][maxid[n]]
        ).values.item() / 1e9
        ds = float(
            (
                array.subref_z[array.scantype == "ON"][minid[n + 1]]
                - array.subref_z[array.scantype == "ON"][maxid[n]]
            ).values
        )
        ss = ds / dt
        alldata.add_row(
            (ss, g.amplitude_0, g.mean_0, g.stddev_0, g.slope_1, g.intercept_1)
        )

    ax.set_xlabel("subref_z")
    ax.set_ylabel("temparature")
    ax.set_title(f"Group #{n}")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / f"subrefz_fit_#{n}.{image_format}")
    if do_plot:
        plt.show()
    else:
        plt.clf()
        plt.close()

# 5th step: plot subref_z vs scan_speed
print("#5: plot subref_z vs scan_speed")

fig, ax = plt.subplots(1, 1, figsize=(7, 7))

ax.scatter([], [], color="C0", label="dZ > 0")
ax.scatter([], [], color="C1", label="dZ < 0")
for ss, zm in zip(alldata["scan_speed"], alldata["z_mean"]):
    if ss < 0:
        ax.plot(-ss, zm, "o", color="C1")
    else:
        ax.plot(ss, zm, "o", color="C0")
sz_mean = alldata["z_mean"].mean()
ax.axhline(sz_mean, color="C2", label=f"mean: {sz_mean:.2f}")

ax.set_xlabel("scan_speed")
ax.set_ylabel("subref_z")
ax.legend()

fig.tight_layout()
fig.savefig(output_dir / f"subrefz_vs_scanspeed.{image_format}")
if do_plot:
    plt.show()
else:
    plt.clf()
    plt.close()

alldata.write(result_file, format="ascii", overwrite=True)
