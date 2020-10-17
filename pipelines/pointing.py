# standard libraries
import warnings
warnings.filterwarnings('ignore')
import argparse
import pathlib
import yaml

# dependent packages
import decode as dc
import xarray as xr
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
plt.style.use('seaborn-muted')
from astropy import table
from astropy.io import fits
import astropy.units as u
import aplpy

# original package
from . import functions as fc


##### command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('dfits_file', help='DFITS name')
# parser.add_argument('antlog_file', help='antenna log')
parser.add_argument('yaml_file', help='parameter file')
args = parser.parse_args()

dfits_file  = pathlib.Path(args.dfits_file)
# antlog_file = pathlib.Path(args.antlog_file)
yaml_file   = pathlib.Path(args.yaml_file)
with open(yaml_file) as file:
    params = yaml.load(file, Loader=yaml.SafeLoader)

##### directory settings
obsid      = dfits_file.name.split('_')[1].split('.')[0]
output_dir = pathlib.Path(params['file']['output_dir']) / obsid
if not output_dir.exists():
    output_dir.mkdir(parents=True)
cont_obs_fits = output_dir / f'continuum_obs.fits'
cube_obs_fits = output_dir / f'cube_obs.fits'
cont_mod_fits = output_dir / f'continuum_model.fits'
cont_res_fits = output_dir / f'continuum_residual.fits'
result_file   = output_dir / params['file']['result_file']
image_format  = params['file']['image_format']
do_plot       = params['file']['do_plot']
dpi           = params['file']['dpi']

##### dc.io.loaddfits parameters
ch = params['loaddfits']['ch']
array = dc.io.loaddfits(dfits_file, **params['loaddfits'])

##### 1st step: check automatic R/SKY assignment
print('#1: automatic R/SKY assignment')

scantypes = np.unique(array.scantype)
if 'R' in scantypes:
    Tr = array[array.scantype=='R'][:, ch].values.mean()
else:
    Tr = 290.0
print(f'scantypes: {scantypes}')
print(f'Tr: {Tr:.1f} [K]')

# fig, ax = plt.subplots(2, 1, figsize=(10, 5), dpi=dpi)
fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=dpi)
tstart0 = params['check_scantypes']['tstart0']
tend0   = params['check_scantypes']['tend0']
tstart1 = params['check_scantypes']['tstart1']
tend1   = params['check_scantypes']['tend1']
subarray0 = array[tstart0:tend0, :]
subarray1 = array[tstart1:tend1, :]

plot_params0 = {'marker': '.', 'markersize': 1.0, 'linewidth': 0.5}
plot_params1 = {'marker': '.', 'markersize': 1.0, 'linestyle': 'None'}

# dc.plot.plot_timestream(subarray0, ch,                  ax=ax[0], **plot_params0)
# dc.plot.plot_timestream(subarray0, ch, scantypes=['R'], ax=ax[0], **plot_params1)

# dc.plot.plot_timestream(subarray1, ch,                     ax=ax[1], **plot_params0)
# dc.plot.plot_timestream(subarray1, ch, scantypes=['SCAN'], ax=ax[1], **plot_params1)
# dc.plot.plot_timestream(subarray1, ch, scantypes=['TRAN'], ax=ax[1], **plot_params1)
# dc.plot.plot_timestream(subarray1, ch, scantypes=['ACC'],  ax=ax[1], **plot_params1)
# dc.plot.plot_timestream(subarray1, ch, scantypes=['GRAD'], ax=ax[1], **plot_params1)


fig.tight_layout()
fig.savefig(output_dir / f'time_stream.{image_format}')
if do_plot:
    plt.show()
else:
    plt.clf()
    plt.close()

import sys; sys.exit()

##### 2nd step: plot subref offsets vs time
print('#2: plot subref offsets vs time')

fig, ax = plt.subplots(3, 1, figsize=(10, 7), dpi=dpi)
dc.plot.plot_tcoords(array, ('time', 'subref_x'), scantypes=['SCAN'], ax=ax[0])
dc.plot.plot_tcoords(array, ('time', 'subref_y'), scantypes=['SCAN'], ax=ax[1])
dc.plot.plot_tcoords(array, ('time', 'subref_z'), scantypes=['SCAN'], ax=ax[2])

fig.tight_layout()
fig.savefig(output_dir / f'subref_movement.{image_format}')
if do_plot:
    plt.show()
else:
    plt.clf()
    plt.close()

##### 3rd step: baseline subtraction
print('#3: baseline subtraction')

Tamb = params['calibration']['Tamb']

times   = []
medians = []
for sid in np.unique(array.scanid):
    subarray = array[array.scanid==sid]
    scantype = np.unique(subarray.scantype)
    t = subarray.time.values[int(len(subarray)/2)]
    if scantype == 'SCAN': ### <- remove??
        # m0 = subarray[:int(1/4 * len(subarray))].median('t').values
        # m1 = subarray[int(3/4 * len(subarray)):].median('t').values
        # times.append(t)
        # medians.append((m0+m1) / 2)
        pass
    elif scantype == 'TRAN':
        times.append(t)
        medians.append(subarray.median('t').values)
    elif scantype == 'ACC':
        times.append(t)
        medians.append(subarray.median('t').values)

times   = np.array(times).astype(float)
medians = np.array(medians).astype(float)
medians[np.isnan(medians)] = 0

blarray = interp1d(times, medians, axis=0, kind='cubic', fill_value='extrapolate')(array.time.astype(float))
blarray = dc.full_like(array, blarray)
scanarray_cal = (Tamb * (array - blarray) / (Tr - blarray))[array.scantype=='SCAN']

fig, ax = plt.subplots(2, 1, figsize=(10, 5), dpi=dpi)

plot_params = {'linewidth': 0.2}

dc.plot.plot_timestream(array,         ch, scantypes=['SCAN'], ax=ax[0], **plot_params)
dc.plot.plot_timestream(scanarray_cal, ch, scantypes=['SCAN'], ax=ax[1], **plot_params)

fig.tight_layout()
fig.savefig(output_dir / f'baseline_subtraction.{image_format}')
if do_plot:
    plt.show()
else:
    plt.clf()
    plt.close()

##### 4th step: make cube/continuum
print('#4: make cube/continuum')

gx   = params['imaging']['gx']
gy   = params['imaging']['gy']
xmin = scanarray_cal.x.min().values
xmax = scanarray_cal.x.max().values
ymin = scanarray_cal.y.min().values
ymax = scanarray_cal.y.max().values

cube_array = dc.tocube(scanarray_cal, gx=gx, gy=gy, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
dc.io.savefits(cube_array, cube_obs_fits, overwrite=True)

exchs = params['imaging']['exchs']
mask  = np.full_like(scanarray_cal.kidid.values, True, dtype=np.bool)
mask[exchs] = False
mask[np.where(scanarray_cal.kidtp != 1)] = False
masked_cube_array = cube_array[:, :, mask]

weight     = dc.ones_like(masked_cube_array)
cont_array = fc.makecontinuum(masked_cube_array, weight=weight)
dc.io.savefits(cont_array, cont_obs_fits, dropdeg=True, overwrite=True)

##### 5th step: 2D-Gauss fit on the continuum map
print('#5: 2D-Gauss fit on the continuum map')

alldata = table.QTable(names=('subref_x', 'subref_y', 'peak', 'x_mean', 'y_mean', 'x_stddev', 'y_stddev', 'theta'))

amplitude = float(cont_array.max().values)
x_mean    = float(cont_array.where(cont_array==cont_array.max(), drop=True).x.values)
y_mean    = float(cont_array.where(cont_array==cont_array.max(), drop=True).y.values)
x_stddev  = params['fitting']['x_stddev']
y_stddev  = params['fitting']['y_stddev']
theta     = params['fitting']['theta']

f = fc.gauss_fit(cont_array,
                 mode='deg',
                 chs=[0],
                 amplitude=amplitude,
                 x_mean=x_mean,
                 y_mean=y_mean,
                 x_stddev=x_stddev,
                 y_stddev=y_stddev,
                 theta=theta)

sigma2hpbw = 2 * np.sqrt(2 * np.log(2))
hpbw_major_arcsec = float(f.x_stddev * 3600 * sigma2hpbw)
hpbw_major_rad    = float(f.x_stddev * sigma2hpbw * np.pi / 180)
hpbw_minor_arcsec = float(f.y_stddev * 3600 * sigma2hpbw)
hpbw_minor_rad    = float(f.y_stddev * sigma2hpbw * np.pi / 180)
print(f'hpbw_major: {hpbw_major_arcsec:.1f} [arcsec], {hpbw_major_rad:.1e} [rad]')
print(f'hpbw_minor: {hpbw_minor_arcsec:.1f} [arcsec], {hpbw_minor_rad:.1e} [rad]')

header = fits.getheader(cont_obs_fits)
fits.writeto(cont_mod_fits, f[:, :, 0].values.T, header, overwrite=True)
fits.writeto(cont_res_fits, (cont_array[:, :, 0] - f[:, :, 0]).values.T, header, overwrite=True)

fig = plt.figure(figsize=(12, 4), dpi=dpi)

ax = aplpy.FITSFigure(str(cont_obs_fits), figure=fig, subplot=(1, 3, 1))
ax.show_colorscale(cmap='viridis', stretch='linear')
ax.add_colorbar(width=0.15)
ax.set_title('Observation')

ax = aplpy.FITSFigure(str(cont_mod_fits), figure=fig, subplot=(1, 3, 2))
ax.show_colorscale(cmap='viridis', stretch='linear')
ax.add_colorbar(width=0.15)
ax.set_title('Model')
ax.tick_labels.hide_y()
ax.axis_labels.hide_y()

ax = aplpy.FITSFigure(str(cont_res_fits), figure=fig, subplot=(1, 3, 3))
ax.show_colorscale(cmap='viridis', stretch='linear')
ax.add_colorbar(width=0.15)
ax.set_title('Residual')
ax.tick_labels.hide_y()
ax.axis_labels.hide_y()

plt.tight_layout(pad=4.0, w_pad=0.5)
plt.savefig(output_dir / f'continuum_model.{image_format}')
if do_plot:
    plt.show()
else:
    plt.clf()
    plt.close()

alldata.add_row([np.median(array.subref_x), np.median(array.subref_y), f.peak, f.x_mean, f.y_mean, f.x_stddev, f.y_stddev, f.theta])
alldata.write(result_file, format='ascii', overwrite=True)