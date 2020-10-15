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
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    print('tqdm is not installed')
    isTqdmInstalled = False
else:
    isTqdmInstalled = True

# original package
from . import functions as fc


##### command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('dfitsname', help='DFITS name')
parser.add_argument('fluxtxt', help='flux text')
parser.add_argument('ymlname', help='parameter file')
args = parser.parse_args()

dfitsname = pathlib.Path(args.dfitsname)
fluxtxt   = pathlib.Path(args.fluxtxt)
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
cubedir   = outdir / 'cube'
if not cubedir.exists():
    cubedir.mkdir()
outtxt = outdir / ymldata['file']['outtxt']
imgfmt = ymldata['file']['imgfmt']
pltflg = ymldata['file']['pltflg']

##### dc.io.loaddfits parameters
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

array = dc.io.loaddfits(
            dfitsname,
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
Tr = array[array.scantype == 'R'][:, ch].values.mean()

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

##### 2nd step: advanced baseline fit
print('#2: advanced baseline fit')

Tamb = ymldata['calibration']['Tamb']
scanarray = array[array.scantype == 'SCAN']
offarray  = array[array.scantype == 'ACC']
rarray    = array[array.scantype == 'R']
scanarray_cal, offarray_cal = dc.models.chopper_calibration(scanarray, offarray, rarray, Tamb, mode='mean')

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

# scanarray_cal2 = (Tr * (array - blarray) / (Tr - blarray))[array.scantype == 'SCAN']
scanarray_cal2 = (Tamb * (array - blarray) / (Tr - blarray))[array.scantype == 'SCAN']
scanarray_cal3 = scanarray_cal2.copy()
scanarray_cal3.values = scanarray_cal2.values - np.nanmean(offarray_cal.values)

fig, ax = plt.subplots(3, 1, figsize=(10, 7))
refch   = ymldata['calibration']['refch']

dc.plot.plot_timestream(scanarray_cal, kidid=refch, xtick=xtick, scantypes=['SCAN'], ax=ax[0])
dc.plot.plot_timestream(scanarray_cal2, kidid=refch, xtick=xtick, scantypes=['SCAN'], ax=ax[1])
dc.plot.plot_timestream(scanarray_cal3, kidid=refch, xtick=xtick, scantypes=['SCAN'], ax=ax[2])
# dc.plot.plot_timestream(scanarray_cal - scanarray_cal3, kidid=refch, xtick=xtick, scantypes=['SCAN'], ax=ax[3])
ax[0].set_title(f'chppper calibration ch #{refch}')
ax[1].set_title(f'advanced baseline fitting ch #{refch}')
ax[2].set_title(f'advanced baseline fitting II ch #{refch}')
# ax[2].set_title(f'chopper calibration - advanced baseline fitting ch #{refch}')

fig.tight_layout()
fig.savefig(outdir / f'calibration_{obsid}.{imgfmt}')
if pltflg:
    plt.show()
else:
    plt.clf()
    plt.close()

##### 3rd step: make cube/continuum
print('#3: make cube/continuum')

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

##### 4th step: 2D-Gauss fit on the continuum map
print('#4: 2D-Gauss fit on the continuum map')

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
ax[0].imshow(cont2[:, :, 0].T, cmap=cmap)
ax[0].set_title('Observation')
ax[1].imshow(f[:, :, 0].T, cmap=cmap)
ax[1].set_title('Model')
ax[2].imshow(cube2[:, :, 0].T - f[:, :, 0].T, cmap=cmap)
ax[2].set_title('Residual')

fig.tight_layout()
fig.savefig(outdir / f'contfit_{obsid}.{imgfmt}')
if pltflg:
    plt.show()
else:
    plt.clf()
    plt.close()

##### 5th step: 2D-Gauss fit on the cube map
print('#5: 2D-Gauss fit on the cube map')

alldata = table.QTable()

h = fc.gauss_fit(cube2,
                 mode='deg',
                 amplitude=f.peak,
                 x_mean=f.x_mean,
                 y_mean=f.y_mean,
                 x_stddev=f.x_stddev,
                 y_stddev=f.y_stddev,
                 theta=f.theta,
                 fixed={'x_mean':True, 'y_mean':True, 'theta':True})

for exch in exchs:
    h.peak[exch]     = np.nan
    h.x_mean[exch]   = np.nan
    h.y_mean[exch]   = np.nan
    h.x_stddev[exch] = np.nan
    h.y_stddev[exch] = np.nan
    h.theta[exch]    = np.nan

freqdata        = np.sort(h.kidfq) * u.GHz
freqdata_sub    = freqdata[~np.isnan(freqdata)]
peakdata        = h.peak[np.argsort(h.kidfq)] * u.K     # temperature
peakdata_sub    = peakdata[~np.isnan(freqdata)]
xmeandata       = h.x_mean[np.argsort(h.kidfq)] * u.deg # degree
xmeandata_sub   = xmeandata[~np.isnan(freqdata)]
ymeandata       = h.y_mean[np.argsort(h.kidfq)] * u.deg # degree
ymeandata_sub   = ymeandata[~np.isnan(freqdata)]
xstddevdata     = h.x_stddev[np.argsort(h.kidfq)] * u.deg # degree
xstddevdata_sub = xstddevdata[~np.isnan(freqdata)]
ystddevdata     = h.y_stddev[np.argsort(h.kidfq)] * u.deg # degree
ystddevdata_sub = ystddevdata[~np.isnan(freqdata)]
thetadata       = h.theta[np.argsort(h.kidfq)] * u.rad # radian
thetadata_sub   = thetadata[~np.isnan(freqdata)]

alldata['frequency'] = freqdata_sub
alldata['peak']      = peakdata_sub
alldata['x_mean']    = xmeandata_sub
alldata['y_mean']    = ymeandata_sub
alldata['x_stddev']  = xstddevdata_sub
alldata['y_stddev']  = ystddevdata_sub
alldata['theta']     = thetadata_sub

if isTqdmInstalled:
    iterator = tqdm(range(len(h.kidfq[(h.kidtp != 0) & (h.kidtp != 2)])))
else:
    iterator = range(len(h.kidfq[(h.kidtp != 0) & (h.kidtp != 2)]))
for i in iterator:
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(cube2[:, :, 7+i].T, cmap=cmap)
    ax[0].set_title('Observation')
    ax[1].imshow(h[:, :, 7+i].T, cmap=cmap)
    ax[1].set_title('Model')
    ax[2].imshow(cube2[:, :, 7+i].T - h[:, :, 7+i].T, cmap=cmap)
    ax[2].set_title('Residual')
    plt.suptitle(f'ch #{7+i}')

    fig.tight_layout()
    fig.savefig(cubedir / f'cubefit_#{7+i}_{obsid}.{imgfmt}')
    plt.clf()
    plt.close()

##### 6th step: planet propeties
print('#6: planet properties')

plt.style.use('seaborn-darkgrid')

data   = np.loadtxt(fluxtxt, comments='#', usecols = (1, 2, 3, 4, 5, 6))
dtable = table.Table(data)
istart = ymldata['planet']['istart']
iend   = ymldata['planet']['iend']

fig, ax = plt.subplots(2, 1, figsize=(10, 5))
x  = np.linspace(300, 399, 100) # frequency
ll = interp1d(data[istart:iend, 0], data[istart:iend, 1], kind='cubic')
theta_s = data[istart:iend, 1].mean()

ax[0].plot(data[istart:iend, 0], data[istart:iend, 1], '.')
ax[0].plot(data[istart:iend, 0], ll(x), 'k-')
ax[0].set_xlabel('frequency [GHz]')
ax[0].set_ylabel('angular diameter [arcsec]')

mm = interp1d(data[istart:iend, 0], data[istart:iend, 2], kind='cubic')

ax[1].plot(data[istart:iend, 0], data[istart:iend, 2], '.')
ax[1].plot(data[istart:iend, 0], mm(x), 'k-')
ax[1].set_xlabel('frequency [GHz]')
ax[1].set_ylabel('Tb [K]')

fig.tight_layout()
fig.savefig(outdir / f'planet_{obsid}.{imgfmt}')
if pltflg:
    plt.show()
else:
    plt.clf()
    plt.close()

f_omega_s = 0.7854
omega_mb  = float(np.pi / (4 * np.log(2)) * \
            (f.x_stddev * 3600 * sigma2hpbw) * (f.y_stddev * 3600 * sigma2hpbw)) ### [arcsec**2]
c = xr.DataArray(mm(h.kidfq) * ll(h.kidfq)**2 * f_omega_s / omega_mb, dims='ch')

##### 7th step: beam FWHM vs frequency
print('#7: beam FWHM')

fig, ax = plt.subplots(2, 1, figsize=(10, 5))

hpbw_major = np.sqrt(((h.x_stddev * 3600 * sigma2hpbw)**2) - np.log(2) / 2 * theta_s**2)
hpbw_minor = np.sqrt(((h.y_stddev * 3600 * sigma2hpbw)**2) - np.log(2) / 2 * theta_s**2)

hpbw_major_sub = hpbw_major[~np.isnan(freqdata)] * u.arcsec # deconvolved
hpbw_minor_sub = hpbw_minor[~np.isnan(freqdata)] * u.arcsec # deconvolved

alldata['HPBW_maj'] = hpbw_major_sub
alldata['HPBW_min'] = hpbw_minor_sub

ax[0].errorbar(np.sort(h.kidfq),
               h.x_stddev[np.argsort(h.kidfq)] * 3600 * sigma2hpbw,
               yerr=h.uncert1[np.argsort(h.kidfq)] * 3600 * sigma2hpbw,
               fmt='o-',
               label='raw')
ax[0].errorbar(np.sort(h.kidfq),
               hpbw_major[np.argsort(h.kidfq)],
               yerr=h.uncert1[np.argsort(h.kidfq)] * 3600 * sigma2hpbw,
               fmt='o-',
               label='deconvolved')
ax[0].set_xlabel('frequency [GHz]')
ax[0].set_ylabel('HPBW maj [arcsec]')
ax[0].legend()

ax[1].errorbar(np.sort(h.kidfq),
               h.y_stddev[np.argsort(h.kidfq)] * 3600 * sigma2hpbw,
               yerr=h.uncert2[np.argsort(h.kidfq)] * 3600 * sigma2hpbw,
               fmt='o-',
               label='raw')
ax[1].errorbar(np.sort(h.kidfq),
               hpbw_minor[np.argsort(h.kidfq)],
               yerr=h.uncert2[np.argsort(h.kidfq)] * 3600 * sigma2hpbw,
               fmt='o-',
               label='deconvolved')
ax[1].set_xlabel('frequency [GHz]')
ax[1].set_ylabel('HPBW min [arcsec]')
ax[1].legend()

fig.tight_layout()
fig.savefig(outdir / f'beam_{obsid}.{imgfmt}')
if pltflg:
    plt.show()
else:
    plt.clf()
    plt.close()

##### 8th step: calculate main beam efficiency
print('#8: main beam efficiency')

eta    = np.array([])
eta_er = np.array([])
freq   = np.sort(h.kidfq.values[h.kidtp == 1])
for i in range(len(h.kidfq[h.kidtp == 1])):
    Ta  = h.peak[h.kidtp == 1][np.argsort(h.kidfq.values[h.kidtp == 1])][i]
    er  = h.uncert0[h.kidtp == 1][np.argsort(h.kidfq.values[h.kidtp == 1])][i]
    Tmb = mm(h.kidfq[h.kidtp == 1][np.argsort(h.kidfq.values[h.kidtp == 1])][i]) * \
          ll(h.kidfq[h.kidtp == 1][np.argsort(h.kidfq.values[h.kidtp == 1])][i])**2 * f_omega_s / omega_mb
    eta    = np.append(eta, [Ta / Tmb], axis=0)
    eta_er = np.append(eta_er, [er / Tmb], axis=0)
eta_med = np.nanmedian(eta)
print(f'eta_median: {eta_med:.2f}')

alldata['beam_efficiency'] = eta

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

ax.errorbar(h.kidfq[h.kidtp == 1][np.argsort(h.kidfq.values[h.kidtp == 1])],
            eta,
            yerr=eta_er,
            fmt='o', label='beam efficiency')
ax.axhline(eta_med, color='C1', label=f'median: {eta_med:.2f}')
ax.set_ylim(0, 1)
ax.set_xlabel('frequency [GHz]')
ax.set_ylabel('$\eta_{mb}$')
ax.legend()

fig.tight_layout()
fig.savefig(outdir / f'eta_{obsid}.{imgfmt}')
if pltflg:
    plt.show()
else:
    plt.clf()
    plt.close()

alldata.write(outtxt, format='ascii', overwrite=True)