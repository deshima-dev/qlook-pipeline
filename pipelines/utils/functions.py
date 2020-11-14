from pathlib import Path
import decode as dc
import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits
from astropy import table
from astropy.modeling import models, fitting

from pandas import date_range
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy import signal
from xarray import DataArray
from typing import List, Tuple, Union, Optional, Any

from logging import getLogger

logger = getLogger(__name__)


def loaddfits(
    fitsname,
    antlogfile,  # <- added
    coordtype="azel",
    loadtype="temperature",
    starttime=None,
    endtime=None,
    pixelids=None,
    scantypes=None,
    mode=0,
    **kwargs
):
    if mode not in [0, 1, 2]:
        raise KeyError(mode)

    logger.info("coordtype starttime endtime mode loadtype")
    logger.info("{} {} {} {} {}".format(coordtype, starttime, endtime, mode, loadtype))

    # pick up kwargs
    # for findR
    findR = kwargs.pop("findR", False)
    ch = kwargs.pop("ch", 0)
    Rth = kwargs.pop("Rth", 280)
    skyth = kwargs.pop("skyth", 150)
    cutnum = kwargs.pop("cutnum", 1)
    # for still
    still = kwargs.pop("still", False)
    period = kwargs.pop("period", 2)
    # for shuttle
    shuttle = kwargs.pop("shuttle", False)
    xmin_off = kwargs.pop("xmin_off", 0)
    xmax_off = kwargs.pop("xmax_off", 0)
    xmin_on = kwargs.pop("xmin_on", 0)
    xmax_on = kwargs.pop("xmax_on", 0)

    # load data
    fitsname = str(Path(fitsname).expanduser())

    with fits.open(fitsname) as hdulist:
        obsinfo = hdulist["OBSINFO"].data
        obshdr = hdulist["OBSINFO"].header
        antlog = hdulist["ANTENNA"].data
        readout = hdulist["READOUT"].data
        wealog = hdulist["WEATHER"].data

    rawantlog = table.Table.read(antlogfile, format="ascii")[:-1]  # <- added

    # obsinfo
    masterids = obsinfo["masterids"][0].astype(np.int64)
    kidids = obsinfo["kidids"][0].astype(np.int64)
    kidfreqs = obsinfo["kidfreqs"][0].astype(np.float64)
    kidtypes = obsinfo["kidtypes"][0].astype(np.int64)

    # parse start/end time
    t_ant = np.array(antlog["time"]).astype(np.datetime64)
    t_out = np.array(readout["starttime"]).astype(np.datetime64)
    t_wea = np.array(wealog["time"]).astype(np.datetime64)

    if starttime is None:
        startindex = 0
    elif isinstance(starttime, int):
        startindex = starttime
    elif isinstance(starttime, str):
        startindex = np.searchsorted(t_out, np.datetime64(starttime))
    elif isinstance(starttime, np.datetime64):
        startindex = np.searchsorted(t_out, starttime)
    else:
        raise ValueError(starttime)

    if endtime is None:
        endindex = t_out.shape[0]
    elif isinstance(endtime, int):
        endindex = endtime
    elif isinstance(endtime, str):
        endindex = np.searchsorted(t_out, np.datetime64(endtime), "right")
    elif isinstance(endtime, np.datetime64):
        endindex = np.searchsorted(t_out, endtime, "right")
    else:
        raise ValueError(starttime)

    if t_out[endindex - 1] > t_ant[-1]:
        logger.warning("Endtime of readout is adjusted to that of ANTENNA HDU.")
        endindex = np.searchsorted(t_out, t_ant[-1], "right")

    t_out = t_out[startindex:endindex]

    # readout
    if loadtype == "temperature":
        response = readout["Tsignal"][startindex:endindex].astype(np.float64)
    elif loadtype == "power":
        response = readout["Psignal"][startindex:endindex].astype(np.float64)
    elif loadtype == "amplitude":
        response = readout["amplitude"][startindex:endindex].astype(np.float64)
    elif loadtype == "phase":
        response = readout["phase"][startindex:endindex].astype(np.float64)
    elif loadtype == "linphase":
        response = readout["line_phase"][startindex:endindex].astype(np.float64)
    else:
        raise KeyError(loadtype)

    # antenna
    if coordtype == "azel":
        x = antlog["az"].copy()
        y = antlog["el"].copy()
        xref = np.median(antlog["az_center"])
        yref = np.median(antlog["el_center"])
        if mode in [0, 1]:
            x -= antlog["az_center"]
            y -= antlog["el_center"]
            if mode == 0:
                x *= np.cos(np.deg2rad(antlog["el"]))
    elif coordtype == "radec":
        x = antlog["ra"].copy()
        y = antlog["dec"].copy()
        xref = obshdr["RA"]
        yref = obshdr["DEC"]
        if mode in [0, 1]:
            x -= xref
            y -= yref
            if mode == 0:
                x *= np.cos(np.deg2rad(antlog["dec"]))
    else:
        raise KeyError(coordtype)
    scantype = antlog["scantype"]

    subref_x = np.array(rawantlog["x"])  # <- added
    subref_y = np.array(rawantlog["y"])  # <- added
    subref_z = np.array(rawantlog["z"])  # <- added

    # weatherlog
    temp = wealog["temperature"]
    pressure = wealog["pressure"]
    vpressure = wealog["vapor-pressure"]
    windspd = wealog["windspd"]
    winddir = wealog["winddir"]

    # interpolation
    dt_out = (t_out - t_out[0]) / np.timedelta64(1, "s")
    dt_ant = (t_ant - t_out[0]) / np.timedelta64(1, "s")
    dt_wea = (t_wea - t_out[0]) / np.timedelta64(1, "s")
    x_i = np.interp(dt_out, dt_ant, x)
    y_i = np.interp(dt_out, dt_ant, y)

    subref_xi = np.interp(dt_out, dt_ant, subref_x)  # <- added
    subref_yi = np.interp(dt_out, dt_ant, subref_y)  # <- added
    subref_zi = np.interp(dt_out, dt_ant, subref_z)  # <- added

    temp_i = np.interp(dt_out, dt_wea, temp)
    pressure_i = np.interp(dt_out, dt_wea, pressure)
    vpressure_i = np.interp(dt_out, dt_wea, vpressure)
    windspd_i = np.interp(dt_out, dt_wea, windspd)
    winddir_i = np.interp(dt_out, dt_wea, winddir)

    scandict = {t: n for n, t in enumerate(np.unique(scantype))}
    scantype_v = np.zeros(scantype.shape[0], dtype=int)
    for k, v in scandict.items():
        scantype_v[scantype == k] = v
    scantype_vi = interp1d(
        dt_ant,
        scantype_v,
        kind="nearest",
        bounds_error=False,
        fill_value=(scantype_v[0], scantype_v[-1]),
    )(dt_out)
    scantype_i = np.full_like(scantype_vi, "GRAD", dtype="<U8")
    for k, v in scandict.items():
        scantype_i[scantype_vi == v] = k

    # for still data
    if still:
        for n in range(int(dt_out[-1]) // period + 1):
            offmask = (period * 2 * n <= dt_out) & (dt_out < period * (2 * n + 1))
            onmask = (period * (2 * n + 1) <= dt_out) & (dt_out < period * (2 * n + 2))
            scantype_i[offmask] = "OFF"
            scantype_i[onmask] = "SCAN"

    if shuttle:
        offmask = (xmin_off < x_i) & (x_i < xmax_off)
        onmask = (xmin_on < x_i) & (x_i < xmax_on)
        scantype_i[offmask] = "OFF"
        scantype_i[onmask] = "SCAN"
        scantype_i[(~offmask) & (~onmask)] = "JUNK"

    if findR:
        Rindex = np.where(response[:, ch] >= Rth)
        scantype_i[Rindex] = "R"
        movemask = np.hstack(
            [[False] * cutnum, scantype_i[cutnum:] != scantype_i[:-cutnum]]
        ) | np.hstack(
            [scantype_i[:-cutnum] != scantype_i[cutnum:], [False] * cutnum]
        ) & (
            scantype_i == "R"
        )
        scantype_i[movemask] = "JUNK"
        scantype_i[(response[:, ch] > skyth) & (scantype_i != "R")] = "JUNK"
        scantype_i[(response[:, ch] <= skyth) & (scantype_i == "R")] = "JUNK"
        skyindex = np.where(response[:, ch] <= skyth)
        scantype_i_temp = scantype_i.copy()
        scantype_i_temp[skyindex] = "SKY"
        movemask = np.hstack(
            [[False] * cutnum, scantype_i_temp[cutnum:] != scantype_i_temp[:-cutnum]]
        ) | np.hstack(
            [scantype_i_temp[:-cutnum] != scantype_i_temp[cutnum:], [False] * cutnum]
        ) & (
            scantype_i_temp == "SKY"
        )
        scantype_i[movemask] = "JUNK"

    # scanid
    scanid_i = np.cumsum(np.hstack([False, scantype_i[1:] != scantype_i[:-1]]))

    # coordinates
    tcoords = {
        "x": x_i,
        "y": y_i,
        "subref_x": subref_xi,  # <- added
        "subref_y": subref_yi,  # <- added
        "subref_z": subref_zi,  # <- added
        "time": t_out,
        "temp": temp_i,
        "pressure": pressure_i,
        "vapor-pressure": vpressure_i,
        "windspd": windspd_i,
        "winddir": winddir_i,
        "scantype": scantype_i,
        "scanid": scanid_i,
    }
    chcoords = {
        "masterid": masterids,
        "kidid": kidids,
        "kidfq": kidfreqs,
        "kidtp": kidtypes,
    }
    scalarcoords = {
        "coordsys": coordtype.upper(),
        "datatype": loadtype,
        "xref": xref,
        "yref": yref,
    }

    # make array
    array = dc.array(
        response, tcoords=tcoords, chcoords=chcoords, scalarcoords=scalarcoords
    )
    if scantypes is not None:
        mask = np.full(array.shape[0], False)
        for scantype in scantypes:
            mask |= array.scantype == scantype
        array = array[mask]

    return array


def makecontinuum(cube, **kwargs):
    inchs = kwargs.pop("inchs", None)
    exchs = kwargs.pop("exchs", None)
    weight = kwargs.pop("weight", None)

    if (inchs is not None) or (exchs is not None):
        raise KeyError("Inchs and exchs are no longer supported. Use weight instead.")

    if weight is None:
        weight = 1.0

    cont = (cube * (1 / weight ** 2)).sum(dim="ch") / (1 / weight ** 2).sum(dim="ch")
    cont = cont.expand_dims(dim="ch", axis=2)  # <- added

    xcoords = {"x": cube.x.values}
    ycoords = {"y": cube.y.values}
    chcoords = {
        "masterid": np.array([0]),
        "kidid": np.array([0]),
        "kidfq": np.array([0]),
        "kidtp": np.array([1]),
    }
    scalarcoords = {
        "coordsys": cube.coordsys.values,
        "datatype": cube.datatype.values,
        "xref": cube.xref.values,
        "yref": cube.yref.values,
    }

    return dc.cube(
        cont.values,
        xcoords=xcoords,
        ycoords=ycoords,
        chcoords=chcoords,
        scalarcoords=scalarcoords,
    )


def gauss_fit(
    map_data,
    chs=None,
    mode="pix",
    amplitude=1,
    x_mean=0,
    y_mean=0,
    x_stddev=None,
    y_stddev=None,
    theta=None,
    cov_matrix=None,
    noise=0,
    **kwargs
):
    if chs is None:
        chs = np.ogrid[0:63]

    if len(chs) > 1:
        for n, ch in enumerate(chs):
            subdata = np.transpose(
                np.full_like(map_data[:, :, ch], map_data.values[:, :, ch])
            )
            subdata[np.isnan(subdata)] = 0

            if mode == "deg":
                mX, mY = np.meshgrid(map_data.x, map_data.y)
            elif mode == "pix":
                mX, mY = np.mgrid[0 : len(map_data.y), 0 : len(map_data.x)]

            g_init = models.Gaussian2D(
                amplitude=np.nanmax(subdata),
                x_mean=x_mean,
                y_mean=y_mean,
                x_stddev=x_stddev,
                y_stddev=y_stddev,
                theta=theta,
                cov_matrix=cov_matrix,
                **kwargs
            ) + models.Const2D(noise)
            fit_g = fitting.LevMarLSQFitter()
            g = fit_g(g_init, mX, mY, subdata)

            g_init2 = models.Gaussian2D(
                amplitude=np.nanmax(subdata - g.amplitude_1),
                x_mean=x_mean,
                y_mean=y_mean,
                x_stddev=x_stddev,
                y_stddev=y_stddev,
                theta=theta,
                cov_matrix=cov_matrix,
                **kwargs
            )
            fit_g2 = fitting.LevMarLSQFitter()
            g2 = fit_g2(g_init2, mX, mY, subdata)

            if n == 0:
                results = np.array([g2(mX, mY)])
                peaks = np.array([g2.amplitude.value])
                x_means = np.array([g2.x_mean.value])
                y_means = np.array([g2.y_mean.value])
                x_stddevs = np.array([g2.x_stddev.value])
                y_stddevs = np.array([g2.y_stddev.value])
                thetas = np.array([g2.theta.value])

                if fit_g2.fit_info["param_cov"] is None:
                    unserts0 = np.array([0])  # <- added
                    unserts1 = np.array([0])  # <- added
                    unserts2 = np.array([0])  # <- added
                else:
                    error = np.diag(fit_g2.fit_info["param_cov"]) ** 0.5
                    uncerts0 = np.array([error[0]])  # <- added
                    uncerts1 = np.array([error[1]])  # <- added
                    uncerts2 = np.array([error[2]])  # <- added

            else:
                results = np.append(results, [g2(mX, mY)], axis=0)
                peaks = np.append(peaks, [g2.amplitude.value], axis=0)
                x_means = np.append(x_means, [g2.x_mean.value], axis=0)
                y_means = np.append(y_means, [g2.y_mean.value], axis=0)
                x_stddevs = np.append(x_stddevs, [g2.x_stddev.value], axis=0)
                y_stddevs = np.append(y_stddevs, [g2.y_stddev.value], axis=0)
                thetas = np.append(thetas, [g2.theta.value], axis=0)
                if fit_g2.fit_info["param_cov"] is None:
                    uncerts0 = np.append(uncerts0, [0], axis=0)  # <- added
                    uncerts1 = np.append(uncerts1, [0], axis=0)  # <- added
                    uncerts2 = np.append(uncerts2, [0], axis=0)  # <- added
                else:
                    error = np.diag(fit_g2.fit_info["param_cov"]) ** 0.5
                    uncerts0 = np.append(uncerts0, [error[0]], axis=0)  # <- added
                    uncerts1 = np.append(uncerts1, [error[1]], axis=0)  # <- added
                    uncerts2 = np.append(uncerts2, [error[2]], axis=0)  # <- added

        result = map_data.copy()
        result.values = np.transpose(results)
        result.attrs.update(
            {
                "peak": peaks,
                "x_mean": x_means,
                "y_mean": y_means,
                "x_stddev": x_stddevs,
                "y_stddev": y_stddevs,
                "theta": thetas,
                "uncert0": uncerts0,  # <- added
                "uncert1": uncerts1,  # <- added
                "uncert2": uncerts2,
            }
        )  # <- added
    else:
        subdata = np.transpose(
            np.full_like(map_data[:, :, 0], map_data.values[:, :, 0])
        )
        subdata[np.isnan(subdata)] = 0

        if mode == "deg":
            mX, mY = np.meshgrid(map_data.x, map_data.y)
        elif mode == "pix":
            mX, mY = np.mgrid[0 : len(map_data.y), 0 : len(map_data.x)]

        g_init = models.Gaussian2D(
            amplitude=np.nanmax(subdata),
            x_mean=x_mean,
            y_mean=y_mean,
            x_stddev=x_stddev,
            y_stddev=y_stddev,
            theta=theta,
            cov_matrix=cov_matrix,
            **kwargs
        ) + models.Const2D(noise)
        fit_g = fitting.LevMarLSQFitter()
        g = fit_g(g_init, mX, mY, subdata)

        g_init2 = models.Gaussian2D(
            amplitude=np.nanmax(subdata - g.amplitude_1),
            x_mean=x_mean,
            y_mean=y_mean,
            x_stddev=x_stddev,
            y_stddev=y_stddev,
            theta=theta,
            cov_matrix=cov_matrix,
            **kwargs
        )
        fit_g2 = fitting.LevMarLSQFitter()
        g2 = fit_g2(g_init2, mX, mY, subdata)

        results = np.array([g2(mX, mY)])
        peaks = np.array([g2.amplitude.value])
        x_means = np.array([g2.x_mean.value])
        y_means = np.array([g2.y_mean.value])
        x_stddevs = np.array([g2.x_stddev.value])
        y_stddevs = np.array([g2.y_stddev.value])
        thetas = np.array([g2.theta.value])
        error = np.diag(fit_g2.fit_info["param_cov"]) ** 0.5
        uncerts0 = np.array(error[0])  # <- added
        uncerts3 = np.array(error[3])  # <- added
        uncerts4 = np.array(error[4])  # <- added

        result = map_data.copy()
        result.values = np.transpose(results)
        result.attrs.update(
            {
                "peak": peaks,
                "x_mean": x_means,
                "y_mean": y_means,
                "x_stddev": x_stddevs,
                "y_stddev": y_stddevs,
                "theta": thetas,
                "uncert0": uncerts0,  # <- added
                "uncert3": uncerts3,  # <- added
                "uncert4": uncerts4,
            }
        )  # <- added

    return result


def resample_with_equal_dt(
    scanarray: DataArray,
    dt_ns: Optional[int] = None,
    **kwargs: Any
) -> DataArray:
    """Resample the data array along the time axis.

    ToDo:
        * Accept ``numpy.datetime64`` as ``dt_ns``.

    Args:
        scanarray: Data array to be resampled.
            Its ``scantype`` should be ``SCAN``.
        dt_ns: Time spacing in the units of nano second. If ``None`` (by default),
            it is calculated from ``numpy.mean(numpy.diff(scanrray.time))``.
        **kwargs: Keyword arguments passed to ``xarray.DataArray.interp``.

    Returns:
        Resampled data array.
    """

    if dt_ns is None:
        dt_ns = np.mean(np.diff(scanarray.time))

    t_start, t_end = scanarray.time[[0, -1]].values
    t_interp = date_range(t_start, t_end, freq=f"{dt_ns}N", closed="left")

    scanarray.coords["t"] = scanarray.time
    scanarray_interp = scanarray.interp(t=t_interp, **kwargs)
    scanarray_interp.coords["time"] = scanarray_interp.t
    del scanarray.coords["t"]
    del scanarray_interp.coords["t"]

    # hack
    scantype_array = np.full_like(scanarray_interp.time, "SCAN", dtype="<U4")
    scanarray_interp = scanarray_interp.assign_coords(scantype=("t", scantype_array))

    return scanarray_interp


def detrend(
    scanarray: DataArray,
    periodic_boundary: bool = True,
    **kwargs: Any
) -> DataArray:
    """Remove linear trend from the data array.

    Args:
        scanarray: Data array to be modeled.
        periodic_boundary: Flag for periodic boundary condition.
            If ``True`` (by default), the baseline is linearly estimated from the both edge of the data.
            If ``False``, ``scipy.signal.detrend`` is called.
        **kwargs: Keyword arguments passed to ``scipy.signal.detrend``.

    Returns:
        Data array whose linear trend is removed.
    """

    if not periodic_boundary:
        scipy_detrend = dc.xarrayfunc(signal.detrend)
        return scipy_detrend(scanarray, **kwargs)

    baseline = scanarray[[0, -1]]
    baseline.coords["t"] = baseline.time
    baseline = baseline.interp(t=scanarray.time).values

    return scanarray - baseline


def plot_filter_response(
    coefficients: Union[Tuple, List[Tuple]],
    filtertypes: Union[str, List[str]] = "digital",
    worN: int = 1024,
    fs: float = 2 * np.pi,
    ax: Optional[Axes] = None,
    **kwargs: Any
) -> Axes:
    """Plot frequency response of FIR/IIR filters.

    Args:
        coefficients: Pair of filter coefficients (numerator and denominator) in tuple.
            List of tuple is available for multiple filters as well.
        filtertypes: ``digital`` (by default) for digital filters, "analog" for analog filters.
        worN: Parameter of ``scipy.signal.freqz`` or ``scipy.signal.freqs``.
        fs: Sampling frequency passed to ``scipy.signal.freqz``.
        ax: Matplotlib axes object. If ``None`` (by default), ``matplotlib.pyplot.gca`` is called.
        **kwargs: Keyword arguments passed to ``ax.plot``.

    Returns:
        Matplotlib axes object.
    """

    if ax is None:
        ax = plt.gca()

    if isinstance(coefficients, tuple):
        coefficients = [coefficients]
    n_filter = len(coefficients)

    if isinstance(filtertypes, str):
        filtertypes = [filtertypes] * n_filter
    elif len(filtertypes) != n_filter:
        raise ValueError("Length of filtertypes and coefficients is not matched.")

    for coefficient, filtertype in zip(coefficients, filtertypes):
        if filtertype == "digital":
            w, h = signal.freqz(coefficient[0], coefficient[1], worN=worN, fs=fs)
        elif filtertype == "analog":
            w, h = signal.freqs(coefficient[0], coefficient[1], worN=worN)

        ax.semilogx(w, 20 * np.log10(np.abs(h)), **kwargs)

    ax.grid(which="both")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_title("Frequency response of filters")

    return ax


def subtract_baseline_by_HPF():
    pass
