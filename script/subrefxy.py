# standard libraries
import warnings
warnings.filterwarnings('ignore')
import argparse
import pathlib
import yaml
import sys

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

# original package
import functions as fc


##### command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('dfitsname', help='DFITS name')
parser.add_argument('antname', help='antenna log')
parser.add_argument('ymlname', help='parameter file')
args = parser.parse_args()

dfitsname = pathlib.Path(args.dfitsname)
antname   = pathlib.Path(args.antname)
ymlname   = pathlib.Path(args.ymlname)
with open(ymlname) as file:
    ymldata = yaml.load(file, Loader=yaml.SafeLoader)

##### directory settings
obsid  = dfitsname.name.split('_')[1].split('.')[0]
outdir = pathlib.Path(ymldata['file']['outdir']) / obsid
if not outdir.exists():
    outdir.mkdir(parents=True)
dcontname = outdir / f'dcont_{obsid}.fits'
dcubename = outdir / f'dcube_{obsid}.fits'
# cubedir   = outdir / 'cube'
# if not cubedir.exists():
#     cubedir.mkdir()
outtxt = outdir / ymldata['file']['outtxt']
imgfmt = ymldata['file']['imgfmt']
pltflg = ymldata['file']['pltflg']

##### fc.loaddfits parameters
coordtype = ymldata['loaddfits']['coordtype']
starttime = ymldata['loaddfits']['starttime']
endtime   = ymldata['loaddfits']['endtime']
mode      = ymldata['loaddfits']['mode']
loadtype  = ymldata['loaddfits']['loadtype']
findR     = ymldata['loaddfits']['findR']
Rth       = ymldata['loaddfits']['Rth']
skyth     = ymldata['loaddfits']['skyth']
ch        = ymldata['loaddfits']['ch']
cutnum    = ymldata['loaddfits']['cutnum']

array = fc.loaddfits(
            dfitsname,
            antname,
            coordtype=coordtype,
            starttime=starttime,
            endtime=endtime,
            mode=mode,
            loadtype=loadtype,
            findR=findR,
            Rth=Rth,
            skyth=skyth,
            ch=ch,
            cutnum=cutnum
        )

##### 1st step: check automatic R/SKY assignment
print('#1: automatic R/SKY assignment')

scantypes = np.unique(array.scantype)
# Tr = array[array.scantype == 'R'][:, ch].values.mean()
Tr = 290

print(f'scantypes: {scantypes}')
print(f'Tr: {Tr:.1f} [K]')

fig, ax = plt.subplots(2, 1, figsize=(10, 5))
tstart0 = ymldata['check_scantypes']['tstart0']
tend0   = ymldata['check_scantypes']['tend0']
tstart1 = ymldata['check_scantypes']['tstart1']
tend1   = ymldata['check_scantypes']['tend1']
subarray0 = array[tstart0:tend0, :]
subarray1 = array[tstart1:tend1, :]

xtick  = 'time'
marker = '.'
linestyle  = 'None'

dc.plot.plot_timestream(subarray0, kidid=ch, xtick=xtick, scantypes=None, ax=ax[0],
                        marker=marker)
dc.plot.plot_timestream(subarray0, kidid=ch, xtick=xtick, scantypes=['R'], ax=ax[0],
                        marker=marker, linestyle=linestyle)
dc.plot.plot_timestream(subarray1, kidid=ch, xtick=xtick, scantypes=None, ax=ax[1],
                        marker=marker)
dc.plot.plot_timestream(subarray1, kidid=ch, xtick=xtick, scantypes=['SCAN'], ax=ax[1],
                        marker=marker, linestyle=linestyle)
dc.plot.plot_timestream(subarray1, kidid=ch, xtick=xtick, scantypes=['TRAN'], ax=ax[1],
                        marker=marker, linestyle=linestyle)
dc.plot.plot_timestream(subarray1, kidid=ch, xtick=xtick, scantypes=['ACC'], ax=ax[1],
                        marker=marker, linestyle=linestyle)
dc.plot.plot_timestream(subarray1, kidid=ch, xtick=xtick, scantypes=['GRAD'], ax=ax[1],
                        marker=marker, linestyle=linestyle)

fig.tight_layout()
fig.savefig(outdir / f'timestream_{obsid}.{imgfmt}')
if pltflg:
    plt.show()
else:
    plt.clf()
    plt.close()

# sys.exit()

##### 2nd step: plot subref offsets vs time
print('#2: plot subref offsets vs time')

fig, ax = plt.subplots(3, 1, figsize=(10, 7))
dc.plot.plot_tcoords(array, coords=('time', 'subref_x') ,scantypes=['SCAN'], ax=ax[0])
dc.plot.plot_tcoords(array, coords=('time', 'subref_y') ,scantypes=['SCAN'], ax=ax[1])
dc.plot.plot_tcoords(array, coords=('time', 'subref_z') ,scantypes=['SCAN'], ax=ax[2])

# maxid = list(argrelmax(array.subref_z[array.scantype == 'ON'].values, order=1)[0])
# minid = list(argrelmin(array.subref_z[array.scantype == 'ON'].values, order=1)[0])
# ax[2].plot(array.time[array.scantype == 'ON'][maxid],
#            array.subref_z[array.scantype == 'ON'][maxid],
#            'o', color='C1', label='local max')
# ax[2].plot(array.time[array.scantype == 'ON'][minid],
#            array.subref_z[array.scantype == 'ON'][minid],
#            'o', color='C2', label='local min')
# ax[2].legend()

fig.tight_layout()
fig.savefig(outdir / f'subrefxyz_vs_time_{obsid}.{imgfmt}')
if pltflg:
    plt.show()
else:
    plt.clf()
    plt.close()

# sys.exit()

##### 3rd step: baseline subtraction
print('#3: baseline subtraction')

Tamb = ymldata['calibration']['Tamb']
# scanarray = array[array.scantype == 'SCAN']
# offarray  = array[array.scantype == 'ACC']
# rarray    = array[array.scantype == 'R']
# scanarray_cal, offarray_cal = dc.models.chopper_calibration(scanarray, offarray, rarray, Tamb, mode='mean')

times = []
medians = []
for sid in np.unique(array.scanid):
    subarray = array[array.scanid == sid]
    scantype = np.unique(subarray.scantype)
    t = subarray.time.values[int(len(subarray) / 2)]
    if scantype == 'SCAN':
        m0 = subarray[:int(1 / 4 * len(subarray))].median('t').values
        m1 = subarray[int(3 / 4 * len(subarray)):].median('t').values
        times.append(t)
        medians.append((m0 + m1) / 2)
    elif scantype == 'TRAN':
        times.append(t)
        medians.append(subarray.median('t').values)
    elif scantype == 'ACC':
        times.append(t)
        medians.append(subarray.median('t').values)

times = np.array(times).astype(float)
medians = np.array(medians).astype(float)
medians[np.isnan(medians)] = 0

blarray = interp1d(times, medians, axis=0, kind='cubic', fill_value='extrapolate')(array.time.astype(float))
blarray = dc.full_like(array, blarray)

scanarray_cal2 = (Tamb * (array - blarray) / (Tr - blarray))[array.scantype == 'SCAN']
scanarray_cal3 = scanarray_cal2.copy()
# scanarray_cal3.values = scanarray_cal2.values - np.nanmean(offarray_cal.values)

# fig, ax = plt.subplots(3, 1, figsize=(10, 7))
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
refch   = ymldata['calibration']['refch']

# dc.plot.plot_timestream(scanarray_cal, kidid=refch, xtick=xtick, scantypes=['SCAN'], ax=ax[0])
# dc.plot.plot_timestream(scanarray_cal3, kidid=refch, xtick=xtick, scantypes=['SCAN'], ax=ax[1])
dc.plot.plot_timestream(scanarray_cal3, kidid=refch, xtick=xtick, scantypes=['SCAN'], ax=ax)
# dc.plot.plot_timestream(scanarray_cal - scanarray_cal3, kidid=refch, xtick=xtick, scantypes=['SCAN'], ax=ax[2])
# ax[0].set_title(f'chppper calibration ch #{refch}')
# ax[1].set_title(f'advanced baseline fitting ch #{refch}')
# ax[2].set_title(f'chopper calibration - advanced baseline fitting ch #{refch}')

fig.tight_layout()
fig.savefig(outdir / f'calibration_{obsid}.{imgfmt}')
if pltflg:
    plt.show()
else:
    plt.clf()
    plt.close()

##### 4th step: make cube/continuum
print('#4: make cube/continuum')

gx = ymldata['imaging']['gx']
gy = ymldata['imaging']['gy']
xmin = scanarray_cal3.x.min().values
xmax = scanarray_cal3.x.max().values
ymin = scanarray_cal3.y.min().values
ymax = scanarray_cal3.y.max().values
cube2 = dc.tocube(scanarray_cal3, gx=gx, gy=gy, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
dc.io.savefits(cube2, dcubename, overwrite=True)

exchs = ymldata['imaging']['exchs']
mask  = np.full_like(scanarray_cal3.kidid.values, True, dtype=np.bool)
mask[exchs] = False
mask[np.where(scanarray_cal3.kidtp != 1)] = False

subcube2 = cube2[:, :, mask]
weight   = dc.ones_like(subcube2)
cont2    = fc.makecontinuum(subcube2, weight=weight)
dc.io.savefits(cont2, dcontname, dropdeg=True, overwrite=True)

##### 5th step: 2D-Gauss fit on the continuum map
print('#5: 2D-Gauss fit on the continuum map')

alldata = table.QTable(names=('subref_x', 'peak', 'x_mean', 'y_mean', 'x_stddev', 'y_stddev', 'theta')) ### <- added

amplitude = ymldata['fitting']['amplitude']
x_mean    = ymldata['fitting']['x_mean']
y_mean    = ymldata['fitting']['y_mean']
x_stddev  = ymldata['fitting']['x_stddev']
y_stddev  = ymldata['fitting']['y_stddev']
theta     = ymldata['fitting']['theta']

f = fc.gauss_fit(cont2,
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

plt.style.use('seaborn-dark')
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

cmap = 'viridis'
ax[0].imshow(cont2[:, :, 0].T, cmap=cmap, origin='lower')
ax[0].set_title('Observation')
ax[1].imshow(f[:, :, 0].T, cmap=cmap, origin='lower')
ax[1].set_title('Model')
ax[2].imshow(cube2[:, :, 0].T - f[:, :, 0].T, cmap=cmap, origin='lower')
ax[2].set_title('Residual')

fig.tight_layout()
fig.savefig(outdir / f'contfit_{obsid}.{imgfmt}')
if pltflg:
    plt.show()
else:
    plt.clf()
    plt.close()

alldata.add_row([np.median(array.subref_x), f.peak, f.x_mean, f.y_mean, f.x_stddev, f.y_stddev, f.theta])
alldata.write(outtxt, format='ascii', overwrite=True)