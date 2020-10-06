# standard libraries
import argparse
import pathlib

# dependent packages
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
plt.style.use('seaborn-muted')
from astropy import table
from astropy.modeling import models, fitting


##### command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('output_dir', help='output directory')
parser.add_argument('result_files', help='fitting results', nargs='*')
args = parser.parse_args()

output_dir   = pathlib.Path(args.output_dir)
result_files = [pathlib.Path(rf) for rf in args.result_files]

##### directory settings
if not output_dir.exists():
    output_dir.mkdir(parents=True)

##### read fitting results
subref_xs = []
peaks = []
for rf in result_files:
    fit_result = table.Table.read(rf, format='ascii')
    subref_xs.append(fit_result['subref_x'][0])
    peaks.append(fit_result['peak'][0])
subref_xs = np.array(subref_xs)
peaks = np.array(peaks)

##### Gaussian fit
g_init = models.Gaussian1D(amplitude=peaks.max())
fit_g  = fitting.LevMarLSQFitter()
g      = fit_g(g_init, subref_xs, peaks)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

sxs = np.linspace(subref_xs.min(), subref_xs.max(), 100)

ax.plot(subref_xs, peaks, 'o', label='data')
ax.plot(sxs, g(sxs), label=f'model (mean: {g.mean.value:.2f})')
ax.axvline(g.mean.value, linestyle='--')
ax.set_xlabel('subref X/Y')
ax.set_ylabel('peak')
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / 'gaussian_fit.png')
plt.show()