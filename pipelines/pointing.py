# standard libraries
import warnings
import argparse
import pathlib
import yaml

# dependent packages
import decode as dc
import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
from astropy import table
from astropy.io import fits

# original package
from utils import functions as fc

# module settings
warnings.filterwarnings("ignore")
plt.style.use("seaborn-muted")

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("dfits_file", help="DFITS name")
parser.add_argument("yaml_file", help="parameter file")
args = parser.parse_args()

dfits_file = pathlib.Path(args.dfits_file)
yaml_file = pathlib.Path(args.yaml_file)
with open(yaml_file) as file:
    params = yaml.load(file, Loader=yaml.SafeLoader)

# directory settings
obsid = dfits_file.name.split("_")[1].split(".")[0]
output_dir = pathlib.Path(params["file"]["output_dir"]).expanduser() / obsid
if not output_dir.exists():
    output_dir.mkdir(parents=True)
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
if "R" in scantypes:
    Tr = array[array.scantype == "R"][:, ch].values.mean()
else:
    Tr = 290.0
print(f"scantypes: {scantypes}")
print(f"Tr: {Tr:.1f} [K]")

fc.plot_timestream_ch(array, ch, image_name=f"{output_dir}/time_stream_ch{ch}.{image_format}", do_plot=do_plot, dpi=dpi)

# plot antenna movements

fc.plot_antenna_movement(array=scanarray, image_name=f"{output_dir}/antenna_movement.{image_format}", do_plot=do_plot, dpi=dpi) 

# 2nd step: baseline subtraction
print("#2: baseline subtraction")
# require chopper calibration?
scanarray_sky = dc.xarrayfunc(signal.savgol_filter)(scanarray, 1001, 5, axis=0)
scanarray_cal = scanarray - scanarray_sky


fc.plot_timestream_cal(array, scanarray_cal, ch, image_name=f"{output_dir}/time_stream_cal_ch{ch}.{image_format}", do_plot=do_plot, dpi=dpi)


# 3rd step: make cube/continuum

print("#3: make cube/continuum")
scanarray_cal.kidtp[list(params["imaging"]["exchs"])] = -1
scanarray_cal = scanarray_cal.where(scanarray_cal.kidtp == 1, drop=True)
scanarray_cal = scanarray_cal[:-20000] # if you remove this, not worked, why?


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

cube_array = dc.tocube( scanarray_cal, gx=gx, gy=gy, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
dc.io.savefits(cube_array, cube_obs_fits, overwrite=True)

# exchs = params["imaging"]["exchs"] mask =
# np.full_like(scanarray_cal.kidid.values, True, dtype=np.bool) mask[exchs] =
# False mask[np.where(scanarray_cal.kidtp != 1)] = False masked_cube_array =
# cube_array[:, :, mask]

# weight = dc.ones_like(masked_cube_array)
weight = dc.ones_like(cube_array)
# cont_array = fc.makecontinuum(masked_cube_array, weight=weight)
cont_array = fc.makecontinuum(cube_array, weight=weight)
dc.io.savefits(cont_array, cont_obs_fits, dropdeg=True, overwrite=True)


# 4th step: 2D-Gauss fit on the continuum map
print("#4: 2D-Gauss fit on the continuum map")
cont_result_file = output_dir / params["file"]["result_file"]
x_stddev = params["fitting"]["x_stddev"]
y_stddev = params["fitting"]["y_stddev"]
theta = params["fitting"]["theta"]
floor = params["fitting"]["floor"]


f = fc.cont_2d_gaussfit(cont_array, x_stddev=x_stddev, y_stddev=y_stddev, theta=theta, floor=floor, cont_obs_fits=cont_obs_fits, cont_mod_fits=cont_mod_fits, cont_res_fits=cont_res_fits, image_name=f"{output_dir}/continuum_image.{image_format}", result_file=cont_result_file, do_plot=do_plot, dpi=dpi)
