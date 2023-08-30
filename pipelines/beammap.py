# standard libraries
import warnings
import argparse
import pathlib
import yaml

# dependent packages
import decode as dc
import xarray as xr
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from astropy import table
from astropy.io import fits
import astropy.units as u
import aplpy
from tqdm import tqdm
# original package
from utils import functions as fc

# module settings
warnings.filterwarnings("ignore")
plt.style.use("seaborn-muted")

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("dfits_file", help="DFITS name")
parser.add_argument("yaml_file", help="parameter file")
parser.add_argument("flux_file", help="flux list file made by planetFlux")
args = parser.parse_args()

dfits_file = pathlib.Path(args.dfits_file)
yaml_file = pathlib.Path(args.yaml_file)
flux_file = pathlib.Path(args.flux_file)
with open(yaml_file) as file:
    params = yaml.load(file, Loader=yaml.SafeLoader)

# directory settings
obsid = dfits_file.name.split("_")[1].split(".")[0]
output_dir = pathlib.Path(params["file"]["output_dir"]).expanduser() / obsid
if not output_dir.exists():
    output_dir.mkdir(parents=True)
cube_dir = output_dir / "cube"
if not cube_dir.exists():
    cube_dir.mkdir()
image_format = params["file"]["image_format"]
do_plot = params["file"]["do_plot"]
dpi = params["file"]["dpi"]

# dc.io.loaddfits parameters
ch = params["loaddfits"]["ch"]
array = dc.io.loaddfits(dfits_file, **params["loaddfits"])
scanarray = array[array.scantype == "SCAN"]

# 1st step: check automatic R/SKY assignment
print("#1: automatic R/SKY assignment")

scantypes = np.unique(array.scantype)
Tr = array[array.scantype == "R"][:, ch].values.mean()
print(f"scantypes: {scantypes}")
print(f"Tr: {Tr:.1f} [K]")


fc.plot_timestream_ch(array, ch, image_name=f"{output_dir}/time_stream_ch{ch}.{image_format}", do_plot=do_plot, dpi=dpi)


# plot antenna movements
fc.plot_antenna_movement(array=scanarray, image_name=f"{output_dir}/antenna_movement.{image_format}", do_plot=do_plot, dpi=dpi)


# 2nd step: advanced baseline fit
print("#2: advanced baseline fit")

Tamb = params["calibration"]["Tamb"]

times = []
medians = []
for sid in np.unique(array.scanid):
    subarray = array[array.scanid == sid]
    scantype = np.unique(subarray.scantype)
    t = subarray.time.values[int(len(subarray) / 2)]
    if scantype == "SCAN":
        m0 = subarray[: int(1 / 4 * len(subarray))].median("t").values
        m1 = subarray[int(3 / 4 * len(subarray)) :].median("t").values
        times.append(t)
        medians.append((m0 + m1) / 2)
    elif scantype == "TRAN":
        times.append(t)
        medians.append(subarray.median("t").values)
    elif scantype == "ACC":
        times.append(t)
        medians.append(subarray.median("t").values)

times = np.array(times).astype(float)
medians = np.array(medians).astype(float)
medians[np.isnan(medians)] = 0

blarray = interp1d(times, medians, axis=0, kind="cubic", fill_value="extrapolate")(
    array.time.astype(float)
)
blarray = dc.full_like(array, blarray)
scanarray_cal = (Tamb * (array - blarray) / (Tr - blarray))[array.scantype == "SCAN"]


fc.plot_timestream_cal(array, scanarray_cal, ch, image_name=f"{output_dir}/time_stream_cal_ch{ch}.{image_format}", do_plot=do_plot, dpi=dpi)

# 3rd step: make cube/continuum
print("#3: make cube/continuum")
cube_obs_fits = output_dir / "cube_obs.fits"
cont_obs_fits = output_dir / "continuum_obs.fits"
cont_mod_fits = output_dir / "continuum_model.fits"
cont_res_fits = output_dir / "continuum_residual.fits"

gx = params["imaging"]["gx"]
gy = params["imaging"]["gy"]
xmin = scanarray_cal.x.min().values
xmax = scanarray_cal.x.max().values
ymin = scanarray_cal.y.min().values
ymax = scanarray_cal.y.max().values

cube_array = dc.tocube(scanarray_cal, gx=gx, gy=gy, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
dc.io.savefits(cube_array, cube_obs_fits, overwrite=True)

exchs = params["imaging"]["exchs"]
mask = np.full_like(scanarray_cal.kidid.values, True, dtype=np.bool)
mask[exchs] = False
mask[np.where(scanarray_cal.kidtp != 1)] = False
masked_cube_array = cube_array[:, :, mask]

weight = dc.ones_like(masked_cube_array)
cont_array = fc.makecontinuum(masked_cube_array, weight=weight)
dc.io.savefits(cont_array, cont_obs_fits, dropdeg=True, overwrite=True)

# 4th step: 2D-Gauss fit on the continuum map
print("#4: 2D-Gauss fit on the continuum map")
cont_result_file = output_dir / params["file"]["cont_result_file"]

x_stddev = params["fitting"]["x_stddev"]
y_stddev = params["fitting"]["y_stddev"]
theta = params["fitting"]["theta"]
floor = params["fitting"]["floor"]

f = fc.cont_2d_gaussfit(cont_array, x_stddev=x_stddev, y_stddev=y_stddev, theta=theta, floor=floor, cont_obs_fits=cont_obs_fits, cont_mod_fits=cont_mod_fits, cont_res_fits=cont_res_fits, image_name=f"{output_dir}/continuum_image.{image_format}", result_file=cont_result_file, do_plot=do_plot, dpi=dpi)

# 5th step: 2D-Gauss fit on the cube map
print("#5: 2D-Gauss fit on the cube map")
header = fits.getheader(cont_obs_fits)

alldata = table.QTable()

h = fc.gauss_fit(
    cube_array,
    mode="deg",
    amplitude=f.peak,
    x_mean=f.x_mean,
    y_mean=f.y_mean,
    x_stddev=f.x_stddev,
    y_stddev=f.y_stddev,
    theta=f.theta,
    fixed={"x_mean": True, "y_mean": True, "theta": True},
)

for exch in exchs:
    h.peak[exch] = np.nan
    h.x_mean[exch] = np.nan
    h.y_mean[exch] = np.nan
    h.x_stddev[exch] = np.nan
    h.y_stddev[exch] = np.nan
    h.theta[exch] = np.nan

freqdata = np.sort(h.kidfq) * u.GHz
freqdata_sub = freqdata[~np.isnan(freqdata)]
peakdata = h.peak[np.argsort(h.kidfq)] * u.K  # temperature
peakdata_sub = peakdata[~np.isnan(freqdata)]
xmeandata = h.x_mean[np.argsort(h.kidfq)] * u.deg  # degree
xmeandata_sub = xmeandata[~np.isnan(freqdata)]
ymeandata = h.y_mean[np.argsort(h.kidfq)] * u.deg  # degree
ymeandata_sub = ymeandata[~np.isnan(freqdata)]
xstddevdata = h.x_stddev[np.argsort(h.kidfq)] * u.deg  # degree
xstddevdata_sub = xstddevdata[~np.isnan(freqdata)]
ystddevdata = h.y_stddev[np.argsort(h.kidfq)] * u.deg  # degree
ystddevdata_sub = ystddevdata[~np.isnan(freqdata)]
thetadata = h.theta[np.argsort(h.kidfq)] * u.rad  # radian
thetadata_sub = thetadata[~np.isnan(freqdata)]

alldata["frequency"] = freqdata_sub
alldata["peak"] = peakdata_sub
alldata["x_mean"] = xmeandata_sub
alldata["y_mean"] = ymeandata_sub
alldata["x_stddev"] = xstddevdata_sub
alldata["y_stddev"] = ystddevdata_sub
alldata["theta"] = thetadata_sub

iterator = tqdm(range(len(h.kidfq[(h.kidtp != 0) & (h.kidtp != 2)])))

for i in iterator:
    cube_obs_ch_fits = cube_dir / f"cube_{7+i}_obs.fits"
    cube_mod_ch_fits = cube_dir / f"cube_{7+i}_model.fits"
    cube_res_ch_fits = cube_dir / f"cube_{7+i}_residual.fits"

    fits.writeto(
        cube_obs_ch_fits,
        cube_array[:, :, 7 + i].values.T,
        header,
        overwrite=True,
    )
    fits.writeto(
        cube_mod_ch_fits,
        h[:, :, 7 + i].values.T,
        header,
        overwrite=True,
    )
    fits.writeto(
        cube_res_ch_fits,
        (cube_array[:, :, 7 + i] - h[:, :, 7 + i]).values.T,
        header,
        overwrite=True,
    )

    fig = plt.figure(figsize=(12, 4), dpi=dpi)

    ax = aplpy.FITSFigure(str(cube_obs_ch_fits), figure=fig, subplot=(1, 3, 1))
    ax.show_colorscale(cmap="viridis", stretch="linear")
    ax.add_colorbar(width=0.15)
    ax.set_title("Observation")

    ax = aplpy.FITSFigure(str(cube_mod_ch_fits), figure=fig, subplot=(1, 3, 2))
    ax.show_colorscale(cmap="viridis", stretch="linear")
    ax.add_colorbar(width=0.15)
    ax.set_title("Model")
    ax.tick_labels.hide_y()
    ax.axis_labels.hide_y()

    ax = aplpy.FITSFigure(str(cube_res_ch_fits), figure=fig, subplot=(1, 3, 3))
    ax.show_colorscale(cmap="viridis", stretch="linear")
    ax.add_colorbar(width=0.15)
    ax.set_title("Residual")
    ax.tick_labels.hide_y()
    ax.axis_labels.hide_y()

    plt.tight_layout(pad=4.0, w_pad=0.5)
    fig.savefig(cube_dir / f"cube_model_#{7+i}.{image_format}")
    plt.clf()
    plt.close()

# 6th step: planet propeties
print("#6: planet properties")

plt.style.use("seaborn-darkgrid")
planet_data = np.loadtxt(flux_file, comments="#", usecols=(1, 2, 3, 4))
fig, ax = plt.subplots(2, 1, figsize=(10, 5), dpi=dpi)
#x = np.linspace(300, 399, 100)  # frequency
x = planet_data[:,0]  # frequency
ll = interp1d(planet_data[:, 0], planet_data[:, 1], kind="cubic")
theta_s = planet_data[:, 1].mean()

ax[0].plot(planet_data[:, 0], planet_data[:, 1], ".")
ax[0].plot(planet_data[:, 0], ll(x), "k-")
ax[0].set_xlabel("frequency [GHz]")
ax[0].set_ylabel("angular diameter [arcsec]")

mm = interp1d(planet_data[:, 0], planet_data[:, 2], kind="cubic")

ax[1].plot(planet_data[:, 0], planet_data[:, 2], ".")
ax[1].plot(planet_data[:, 0], mm(x), "k-")
ax[1].set_xlabel("frequency [GHz]")
ax[1].set_ylabel("Tb [K]")

fig.tight_layout()
fig.savefig(output_dir / f"planet_info.{image_format}")
if do_plot:
    plt.show()
else:
    plt.clf()
    plt.close()

f_omega_s = 0.7854 #???
sigma2fwhm = 2 * np.sqrt(2 * np.log(2))

omega_mb = float(
    np.pi
    / (4 * np.log(2))
    * (f.x_stddev * 3600 * sigma2fwhm)
    * (f.y_stddev * 3600 * sigma2fwhm)
)  # [arcsec**2]
c = xr.DataArray(mm(h.kidfq) * ll(h.kidfq) ** 2 * f_omega_s / omega_mb, dims="ch")

# 7th step: beam FWHM vs frequency
print("#7: beam FWHM")

fig, ax = plt.subplots(2, 1, figsize=(10, 5), dpi=dpi)

hpbw_major = np.sqrt(
    ((h.x_stddev * 3600 * sigma2fwhm) ** 2) - np.log(2) / 2 * theta_s ** 2
)
hpbw_minor = np.sqrt(
    ((h.y_stddev * 3600 * sigma2fwhm) ** 2) - np.log(2) / 2 * theta_s ** 2
)

hpbw_major_sub = hpbw_major[~np.isnan(freqdata)] * u.arcsec  # deconvolved
hpbw_minor_sub = hpbw_minor[~np.isnan(freqdata)] * u.arcsec  # deconvolved

alldata["HPBW_maj"] = hpbw_major_sub
alldata["HPBW_min"] = hpbw_minor_sub

ax[0].errorbar(
    np.sort(h.kidfq),
    h.x_stddev[np.argsort(h.kidfq)] * 3600 * sigma2fwhm,
    yerr=h.uncert1[np.argsort(h.kidfq)] * 3600 * sigma2fwhm,
    fmt="o-",
    label="raw",
)
ax[0].errorbar(
    np.sort(h.kidfq),
    hpbw_major[np.argsort(h.kidfq)],
    yerr=h.uncert1[np.argsort(h.kidfq)] * 3600 * sigma2fwhm,
    fmt="o-",
    label="deconvolved",
)
ax[0].set_xlabel("frequency [GHz]")
ax[0].set_ylabel("HPBW maj [arcsec]")
ax[0].legend()

ax[1].errorbar(
    np.sort(h.kidfq),
    h.y_stddev[np.argsort(h.kidfq)] * 3600 * sigma2fwhm,
    yerr=h.uncert2[np.argsort(h.kidfq)] * 3600 * sigma2fwhm,
    fmt="o-",
    label="raw",
)
ax[1].errorbar(
    np.sort(h.kidfq),
    hpbw_minor[np.argsort(h.kidfq)],
    yerr=h.uncert2[np.argsort(h.kidfq)] * 3600 * sigma2fwhm,
    fmt="o-",
    label="deconvolved",
)
ax[1].set_xlabel("frequency [GHz]")
ax[1].set_ylabel("HPBW min [arcsec]")
ax[1].legend()

fig.tight_layout()
fig.savefig(output_dir / f"beam_FWHM.{image_format}")
if do_plot:
    plt.show()
else:
    plt.clf()
    plt.close()

# 8th step: calculate main beam efficiency
print("#8: main beam efficiency")

eta = np.array([])
eta_er = np.array([])
freq = np.sort(h.kidfq.values[h.kidtp == 1])
for i in range(len(h.kidfq[h.kidtp == 1])):
    Ta = h.peak[h.kidtp == 1][np.argsort(h.kidfq.values[h.kidtp == 1])][i]
    er = h.uncert0[h.kidtp == 1][np.argsort(h.kidfq.values[h.kidtp == 1])][i]
    Tmb = (
        mm(h.kidfq[h.kidtp == 1][np.argsort(h.kidfq.values[h.kidtp == 1])][i])
        * ll(h.kidfq[h.kidtp == 1][np.argsort(h.kidfq.values[h.kidtp == 1])][i]) ** 2
        * f_omega_s
        / omega_mb
    )
    eta = np.append(eta, [Ta / Tmb], axis=0)
    eta_er = np.append(eta_er, [er / Tmb], axis=0)
eta_med = np.nanmedian(eta)
print(f"eta_median: {eta_med:.2f}")

alldata["beam_efficiency"] = eta

fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=dpi)

ax.errorbar(
    h.kidfq[h.kidtp == 1][np.argsort(h.kidfq.values[h.kidtp == 1])],
    eta,
    yerr=eta_er,
    fmt="o",
    label="beam efficiency",
)
ax.axhline(eta_med, color="C1", label=f"median: {eta_med:.2f}")
ax.set_ylim(0, 1.2)
ax.set_xlabel("frequency [GHz]")
ax.set_ylabel(r"$\eta_{mb}$")
ax.legend()

fig.tight_layout()
fig.savefig(output_dir / f"beam_efficiency.{image_format}")
if do_plot:
    plt.show()
else:
    plt.clf()
    plt.close()



cube_result_file = output_dir / params["file"]["cont_result_file"]
alldata.write(cube_result_file, format="ascii", overwrite=True)
