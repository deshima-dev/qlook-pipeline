# usage: python subrefz.py dfits_(obsid).fits antenna_log.ant params.yaml
import warnings
warnings.filterwarnings('ignore')
import sys
import pathlib
import yaml
import decode as dc
import xarray as xr
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import argrelmax, argrelmin
import matplotlib.pyplot as  plt
from astropy.io import fits
from astropy import table
from astropy.modeling import models, fitting
import functions as fc
plt.style.use('seaborn-darkgrid')
plt.style.use('seaborn-muted')

##### command line arguments
argv = sys.argv
argc = len(argv)
if argc != 4:
    raise SyntaxError('The number of arguments is wrong!')
dfitsname = pathlib.Path(argv[1])
antname   = pathlib.Path(argv[2])
ymlname   = pathlib.Path(argv[3])
with open(ymlname) as file:
    ymldata = yaml.load(file, Loader=yaml.SafeLoader)

##### directory settings
obsid  = dfitsname.name.split('_')[1].split('.')[0]
outdir = pathlib.Path(ymldata['file']['outdir']) / obsid
if not outdir.exists():
    outdir.mkdir(parents=True)
fitdir = outdir / 'fit'
if not fitdir.exists():
    fitdir.mkdir()

##### fc.loaddfits parameters
coordtype = ymldata['loaddfits']['coordtype']
mode      = ymldata['loaddfits']['mode']
loadtype  = ymldata['loaddfits']['loadtype']

array = fc.loaddfits(
            dfitsname,
            antname,
            coordtype=coordtype,
            mode=mode,
            loadtype=loadtype
        )

##### check time stream
scantypes = np.unique(array.scantype)
print(f'scantypes: {scantypes}')

fig, ax = plt.subplots(2, 1, figsize=(10, 5))
tstart0 = ymldata['check_scantypes']['tstart0']
tend0   = ymldata['check_scantypes']['tend0']
tstart1 = ymldata['check_scantypes']['tstart1']
tend1   = ymldata['check_scantypes']['tend1']
subarray0 = array[tstart0:tend0, :]
subarray1 = array[tstart1:tend1, :]

refch  = ymldata['check_scantypes']['refch']
xtick  = 'time'
marker = '.'
linestyle  = 'None'

dc.plot.plot_timestream(subarray0, kidid=refch, xtick=xtick, scantypes=['GRAD'], ax=ax[0],
                        marker=marker, linestyle=linestyle)
dc.plot.plot_timestream(subarray0, kidid=refch, xtick=xtick, scantypes=['ON'], ax=ax[0],
                        marker=marker, linestyle=linestyle)
dc.plot.plot_timestream(subarray1, kidid=refch, xtick=xtick, scantypes=['ON'], ax=ax[1],
                        marker=marker, linestyle=linestyle)

fig.tight_layout()
fig.savefig(outdir / f'timestream_{obsid}.png')
plt.show()

##### plot subref offsets vs time
fig, ax = plt.subplots(3, 1, figsize=(10, 7))

dc.plot.plot_tcoords(array, coords=('time', 'subref_x') ,scantypes=['ON'], ax=ax[0])
dc.plot.plot_tcoords(array, coords=('time', 'subref_y') ,scantypes=['ON'], ax=ax[1])
dc.plot.plot_tcoords(array, coords=('time', 'subref_z') ,scantypes=['ON'], ax=ax[2])

maxid = list(argrelmax(array.subref_z[array.scantype == 'ON'].values, order=1)[0])
minid = list(argrelmin(array.subref_z[array.scantype == 'ON'].values, order=1)[0])
# minid.insert(0, maxid[0] - (minid[0] - maxid[0]))
ax[2].plot(array.time[array.scantype == 'ON'][maxid],
           array.subref_z[array.scantype == 'ON'][maxid],
           'o', color='C1', label='local max')
ax[2].plot(array.time[array.scantype == 'ON'][minid],
           array.subref_z[array.scantype == 'ON'][minid],
           'o', color='C2', label='local min')
ax[2].legend()

fig.tight_layout()
fig.savefig(outdir / f'subrefxyz_vs_time_{obsid}.png')
plt.show()

##### plot temperature vs subref_z
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

ax.plot(array.subref_z[array.scantype == 'ON'], array[:, refch][array.scantype == 'ON'], label='ON')
ax.set_xlabel('subref_z')
ax.set_ylabel('temperature')
ax.set_title(f'ch #{refch}')
ax.legend()

fig.tight_layout()
fig.savefig(outdir / f'temp_vs_subrefz_{obsid}.png')
plt.show()

##### plot temperature vs subref_z (up)
if maxid[0] < minid[0]:
    minid.insert(0, np.nan)
if minid[-1] < maxid[-1]:
    minid.append(np.nan)

amp0 = ymldata['fitting']['amplitude']
x0   = ymldata['fitting']['x_mean']
s0   = ymldata['fitting']['x_stddev']
sl   = ymldata['fitting']['slope']
ic   = ymldata['fitting']['intercept']
g_init = models.Gaussian1D(amplitude=amp0, mean=x0, stddev=s0) + models.Linear1D(sl, ic)
fit_g  = fitting.LevMarLSQFitter()
for n in range(len(maxid)):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    if n == 0:
        if not np.isnan(minid[n]):
            ax.plot(array.subref_z[array.scantype == 'ON'][minid[n]:maxid[n]],
                    array[:, refch][array.scantype == 'ON'][minid[n]:maxid[n]],
                    color='C0', label='obs (dZ > 0)')            
            g = fit_g(g_init,
                      array.subref_z[array.scantype == 'ON'][minid[n]:maxid[n]],
                      array[:, refch][array.scantype == 'ON'][minid[n]:maxid[n]])
            ax.plot(array.subref_z[array.scantype == 'ON'][minid[n]:maxid[n]],
                    g(array.subref_z[array.scantype == 'ON'][minid[n]:maxid[n]]),
                    color='C2', label='model (dZ > 0)')
        
        ax.plot(array.subref_z[array.scantype == 'ON'][maxid[n]:minid[n+1]],
                array[:, refch][array.scantype == 'ON'][maxid[n]:minid[n+1]],
                color='C1', label='obs (dZ < 0)')
        g = fit_g(g_init,
                  array.subref_z[array.scantype == 'ON'][maxid[n]:minid[n+1]],
                  array[:, refch][array.scantype == 'ON'][maxid[n]:minid[n+1]])
        ax.plot(array.subref_z[array.scantype == 'ON'][maxid[n]:minid[n+1]],
                g(array.subref_z[array.scantype == 'ON'][maxid[n]:minid[n+1]]),
                color='C3', label='model (dZ < 0)')
    
    elif n == len(maxid) - 1:
        ax.plot(array.subref_z[array.scantype == 'ON'][minid[n]:maxid[n]],
                array[:, refch][array.scantype == 'ON'][minid[n]:maxid[n]],
                color='C0', label='obs (dZ > 0)')
        g = fit_g(g_init,
                  array.subref_z[array.scantype == 'ON'][minid[n]:maxid[n]],
                  array[:, refch][array.scantype == 'ON'][minid[n]:maxid[n]])
        ax.plot(array.subref_z[array.scantype == 'ON'][minid[n]:maxid[n]],
                g(array.subref_z[array.scantype == 'ON'][minid[n]:maxid[n]]),
                color='C2', label='model (dZ > 0)')

        if not np.isnan(minid[n+1]):
            ax.plot(array.subref_z[array.scantype == 'ON'][maxid[n]:minid[n+1]],
                    array[:, refch][array.scantype == 'ON'][maxid[n]:minid[n+1]],
                    color='C1', label='obs (dZ < 0)')
            g = fit_g(g_init,
                      array.subref_z[array.scantype == 'ON'][maxid[n]:minid[n+1]],
                      array[:, refch][array.scantype == 'ON'][maxid[n]:minid[n+1]])
            ax.plot(array.subref_z[array.scantype == 'ON'][maxid[n]:minid[n+1]],
                    g(array.subref_z[array.scantype == 'ON'][maxid[n]:minid[n+1]]),
                    color='C3', label='model (dZ < 0)')
    else:
        ax.plot(array.subref_z[array.scantype == 'ON'][minid[n]:maxid[n]],
                array[:, refch][array.scantype == 'ON'][minid[n]:maxid[n]],
                color='C0', label='obs (dZ > 0)')
        g = fit_g(g_init,
                  array.subref_z[array.scantype == 'ON'][minid[n]:maxid[n]],
                  array[:, refch][array.scantype == 'ON'][minid[n]:maxid[n]])
        ax.plot(array.subref_z[array.scantype == 'ON'][minid[n]:maxid[n]],
                g(array.subref_z[array.scantype == 'ON'][minid[n]:maxid[n]]),
                color='C2', label='model (dZ > 0)')
        
        ax.plot(array.subref_z[array.scantype == 'ON'][maxid[n]:minid[n+1]],
                array[:, refch][array.scantype == 'ON'][maxid[n]:minid[n+1]],
                color='C1', label='obs (dZ < 0)')
        g = fit_g(g_init,
                  array.subref_z[array.scantype == 'ON'][maxid[n]:minid[n+1]],
                  array[:, refch][array.scantype == 'ON'][maxid[n]:minid[n+1]])
        ax.plot(array.subref_z[array.scantype == 'ON'][maxid[n]:minid[n+1]],
                g(array.subref_z[array.scantype == 'ON'][maxid[n]:minid[n+1]]),
                color='C3', label='model (dZ < 0)')

    ax.set_xlabel('subref_z')
    ax.set_ylabel('temparature')
    ax.set_title(f'Group #{n}')
    ax.legend()

    fig.tight_layout()
    fig.savefig(fitdir / f'subrefz_fit_#{n}_{obsid}.png')
    plt.show()