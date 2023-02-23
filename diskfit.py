import numpy as np
import sys
from numpy import meshgrid, arange
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units
import os
from scipy.optimize import least_squares as lsq
from scipy.optimize import curve_fit as cf
import astropy.constants as constants
from scipy.stats import skew, kurtosis
from utils import imstd, eval_mauch_beam, wrap_angle90, angle_diff
import logging
from astropy.visualization.wcsaxes import WCSAxes
import argparse

def create_logger():
    """ Create a console logger """
    log = logging.getLogger("DiskFit")
    cfmt = logging.Formatter(('%(name)s - %(asctime)s %(levelname)s - %(message)s'))
    log.setLevel(logging.DEBUG)
    log.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(cfmt)

    log.addHandler(console)

    return log, console, cfmt
log, log_console_handler, log_formatter = create_logger()

parser = argparse.ArgumentParser(description="DiskFit -- program to fit for dipole offsets based on a planetary/lunar disk")
parser.add_argument("--debugplot", dest="DEBUGPLOT", action="store_true", help="Enable channel by channel interactive visual inspection plots for model and measured EVPA")
parser.add_argument("--doPlot", dest="DOPLOT", action="store_true", help="Enable plotting for fitted EVPA offsets and histograms")
parser.add_argument("--snr", "-s", dest="SNR_CUTOFF", default=10., type=float, help="SNR cutoff to apply to Stokes P and I (per channel image)")
parser.add_argument("--iqs", dest="IQS_SPREAD_CUTOFF", default=10., type=float, help="Inter Quartile Spread in angle distribution cutoff to apply to EVPA (per channel image)")
parser.add_argument("--skew", dest="ABS_SKEW_CUTOFF", default=2., type=float, help="Absolute skew in angle distribution cutoff to apply to EVPA (per channel image)")
parser.add_argument("--kurt", dest="ABS_KURT_CUTOFF", default=2., type=float, help="Absolute fisher kurtosis in angle distribution cutoff to apply to EVPA (per channel image)")
parser.add_argument("--torusout", "-o", dest="DISK_TORUS_OUTER", default=0.276040, type=float, help="Outer torus boundary for contributing EVPA angles (should be the outer limb radius in degrees)")
parser.add_argument("--torusin", "-i", dest="DISK_TORUS_INNER", default=0.25, type=float, help="Inner torus boundary for contributing EVPA angles (should exclude reflected RFI)")
parser.add_argument("--obsPAOffset",  dest="OBS_PA_OFFSET", default=0.0, type=float, help="Apply offset rotation (e.g. uncorrected parallactic angle) before fitting")
parser.add_argument("--minPFrac",  dest="FIT_MIN_STOKES_P", default=0.01, type=float, help="Cutoff fractional polarization contribution to the fit below this")
parser.add_argument("--torusFillCutoff",  dest="TORUS_FILL_CUTOFF", default=0.35, type=float, help="Discard slices with fewer than this fractional number of points meeting SNR criteria within the torus (mostly empty torii). Expect 0 <= x <= 1.0")
parser.add_argument("--doFitHV",  dest="DO_FIT_HV", action='store_true', help="Fit also for crosshand phase (assume no circular emission from blackbody -- inner torus cut should be big enough to discard reflected terrestial RFI)")
parser.add_argument("--verbose", "-v", dest="VERBOSE", action='store_true', help="Increase verbosity")
parser.add_argument("--signconv",  dest="SIGNCONV", default=+1, help="Ninja parameter -- flips the sign to counter clockwise rotation if negative if the EVPA rotates North through West")
parser.add_argument("--lowestfreq", dest="LOWFREQ", default=-np.inf, type=float, help="Lowest frequency (default disabled) -- specifies cutoff for loading frequency (in MHz) cubes for fitting")
parser.add_argument("--highestfreq", dest="HIGHFREQ", default=+np.inf, type=float, help="Highest frequency (default disabled) -- specifies cutoff for loading frequency (in MHz) cubes for fitting")
parser.add_argument("--rmscutoff", dest="RMSCUTOFF", default=np.inf, type=float, help="Plane RMS cutoff - specify as percentage")
parser.add_argument("imagePattern", type=str, help="Pattern specifying which images to run fitter on. Expects each slice to conform to WSClean style output, e.g. "
                                                    "'lunarimgs/moon_snapshot-t0000-$xxx-{}-image.fits', where $xxx will be replaced by frequency slice numbers")
args = parser.parse_args()

VERBOSE = args.VERBOSE
SIGNCONV = args.SIGNCONV
DEBUGPLOT = args.DEBUGPLOT
DOPLOT = args.DOPLOT
SNR_CUTOFF = args.SNR_CUTOFF
IQS_SPREAD_CUTOFF = args.IQS_SPREAD_CUTOFF
ABS_SKEW_CUTOFF = args.ABS_SKEW_CUTOFF
ABS_KURT_CUTOFF = args.ABS_KURT_CUTOFF
DISK_TORUS_OUTER = args.DISK_TORUS_OUTER
DISK_TORUS_INNER = args.DISK_TORUS_INNER
OBS_PA_OFFSET = args.OBS_PA_OFFSET
FIT_MIN_STOKES_P = args.FIT_MIN_STOKES_P
TORUS_FILL_CUTOFF = args.TORUS_FILL_CUTOFF
DO_FIT_HV = args.DO_FIT_HV
LOWFREQ = args.LOWFREQ
HIGHFREQ = args.HIGHFREQ
RMSCUTOFF = args.RMSCUTOFF
log.info(':::DiskFit running with the following parameters:::\n'+
         '\n'.join(f'{k.ljust(30, " ")} = {v}' for k, v in vars(args).items())+
         "\n === DiskFit ===")

if DEBUGPLOT or DOPLOT:
    import matplotlib.pyplot as plt
    from astropy.visualization.wcsaxes import WCSAxes
    from matplotlib.transforms import Affine2D
    import matplotlib.patches as patches

fio,fqo,fuo,fvo = [args.imagePattern.format(st) for st in "IQUV"]

all_model_evpa = []
all_measured_evpa = []
all_measured_weight = []
all_lambda = []
all_mask = []
all_hv_offset = []
fitted_slices = 0
nu_considered = set([])

### POPULATE CUBE FILS TO FIT ###
map_list = list(map(lambda a:"{0:04d}".format(a), range(9999)))
plane_rms = []
plane_nu = []
plane_q3_evpa = {}
plane_q1_evpa = {}
plane_q2_evpa = {}
plane_mean_evpa = {}
plane_std_evpa = {}
bmin = -np.inf
bmaj = -np.inf
cellsize = -np.inf
for nui in map_list:
    fi = fio.replace("$xxx",nui)
    if not os.path.exists(fi):
        continue
    else:
        log.info(f"Pattern will load '{fi}'")
    fq = fqo.replace("$xxx",nui)
    fu = fuo.replace("$xxx",nui)
    fv = fvo.replace("$xxx",nui)
    if any(map(lambda x: not os.path.exists(x), [fq, fu, fv])):
        log.critical(f"Missing Q, U or V slices associated with {fi}")
        sys.exit(1)

    hdu = fits.open(fi)[0]
    wcs = WCS(hdu.header)
    if hdu.header["CUNIT3"].strip() != "Hz":
        raise RuntimeError("Expect axis 3 of FITS file to be frequency axis -- unit Hz")
    freq = hdu.header["CRVAL3"]
    log.info(f"Loading frequency {freq*1.e-6} MHz")
    bmaj = max(bmaj, ((hdu.header["BMAJ"] if np.isfinite(hdu.header["BMAJ"]) else -np.inf) * 3600))
    bmin = max(bmin, ((hdu.header["BMIN"] if np.isfinite(hdu.header["BMIN"]) else -np.inf) * 3600))
    cellsize = max(cellsize, abs(hdu.header["CDELT1"]))

    npix = wcs.pixel_shape[0]
    assert wcs.pixel_shape[0] == wcs.pixel_shape[1]
    assert wcs.naxis == 4
    scale = np.abs(wcs.pixel_scale_matrix[0,0])
    crfreq = (wcs.wcs.crval[2] * 1e-9)
    v = fits.open(fv)[0].data
    rms = imstd(v) # detect rms in stokes V
    plane_rms.append(rms)
    plane_nu.append(crfreq)

if len(plane_nu) == 0:
    log.info("Pattern matches no files")
    sys.exit(1)
elif len(plane_nu) == 1:
    log.info("Pattern matches only single file")
    sys.exit(1)
else: pass

beam_area = (np.pi * bmin * bmaj) / (4*np.log(2)) / (cellsize * 3600.)**2 # beam area in npixels
if not np.isfinite(beam_area) and beam_area <= 0:
    raise RuntimeError("Beams areas not fitted")
log.info(f"Using worst case beam area of {beam_area:.2f} pixels")

flagged_planes = np.array(plane_rms) > (np.nanpercentile(plane_rms, RMSCUTOFF) if np.isfinite(RMSCUTOFF) else RMSCUTOFF)

nuii = -1
for nui in map_list:
    fi = fio.replace("$xxx",nui)
    if not os.path.exists(fi):
        continue
    fq = fqo.replace("$xxx",nui)
    fu = fuo.replace("$xxx",nui)
    fv = fvo.replace("$xxx",nui)

    nuii += 1
    if flagged_planes[nuii]:
        log.warn("Skipping plane at {0:0.3f} GHz due to excessive RMS noise".format(plane_nu[nuii]))
        continue
    hdu = fits.open(fi)[0]
    wcs = WCS(hdu.header)
    bmaj = hdu.header["BMAJ"] * 3600
    bmin = hdu.header["BMIN"] * 3600

    npix = wcs.pixel_shape[0]
    assert wcs.pixel_shape[0] == wcs.pixel_shape[1]
    assert wcs.naxis == 4
    scale = np.abs(wcs.pixel_scale_matrix[0,0])
    crfreq = (wcs.wcs.crval[2] * 1e-9)
    nu_considered.add(crfreq * 1e9)
    mauch_beam = eval_mauch_beam(npix, scale, crfreq)

    # get the measured EVPA
    i = hdu.data
    u = fits.open(fu)[0].data
    q = fits.open(fq)[0].data
    v = fits.open(fv)[0].data
    # correct for slight offset in PA correction
    # due to the radec being that of the first timestamp
    PA = np.deg2rad(OBS_PA_OFFSET)

    Qcorr = q * np.cos(-2*PA) + u * np.sin(-2*PA)
    Ucorr = u * np.cos(-2*PA) - q * np.sin(-2*PA)
    q = Qcorr
    u = Ucorr

    wcsslice = wcs.slice(np.s_[0,0,:,:])
    p = (u**2 + q**2) / i**2
    rms = imstd(v) # detect rms in stokes V
    if VERBOSE:
        log.info(f"Estimated background noise as {rms*1e6:.3f} muJy")
    isnrmask = i > rms * SNR_CUTOFF

    psnrmask = np.logical_and(np.logical_and(p > FIT_MIN_STOKES_P, p <= 1.0),
                              np.sqrt(u**2 + q**2) > rms * SNR_CUTOFF)
    xx, yy = meshgrid((arange(npix)-npix//2)*scale,
                      (arange(npix)-npix//2)*scale)

    rim_mask = np.logical_and(xx**2 + yy**2 < DISK_TORUS_OUTER**2,
                              xx**2 + yy**2 > DISK_TORUS_INNER**2)
    mask = rim_mask * psnrmask * isnrmask 
    if VERBOSE:
        log.info(f"Number of points in rim mask: {np.sum(rim_mask)}")
        log.info(f"Number of points in stokes P mask: {np.sum(psnrmask)}")
        log.info(f"Number of points in stokes I mask: {np.sum(isnrmask)}")

    if mask.sum() == 0:
        log.warn(f"Empty / noisy slice at {crfreq:.3f} GHz")
        continue
    rim_sel_frac = np.sum(mask) * 1.0 / np.sum(rim_mask)
    if rim_sel_frac < TORUS_FILL_CUTOFF:
        log.warn(f"Only {rim_sel_frac*100.:.2f}% of the limb is of sufficient SNR, "
                 f"skipping slice at {crfreq:.3f} GHz")
        continue
    freq = hdu.header["CRVAL3"]
    if freq * 1e-6 < LOWFREQ:
        log.warning(f"Skipping frequency slice at {freq*1.e-6} MHz per user request -- lower than cutoff frequency")
        continue
    if freq * 1e-6 > HIGHFREQ:
        log.warning(f"Skipping frequency slice at {freq*1.e-6} MHz per user request -- higher than cutoff frequency")
        continue

    nanmasksel = mask == 0
    mask = np.float64(mask)
    mask[nanmasksel] = np.nan

    masked_q = q.copy()[0,0,:,:] * mask
    masked_u = u.copy()[0,0,:,:] * mask
    masked_v = v.copy()[0,0,:,:] * mask
    masked_i = i.copy()[0,0,:,:] * mask

    measured_hvphaseang = np.rad2deg(np.arctan(masked_v/masked_u))
    measured_evpa = 0.5 * np.rad2deg(np.arctan2(masked_u, masked_q))
    measured_p = (masked_u**2 + masked_q**2) / masked_i
    measured_weight = (np.sqrt(masked_u**2 + masked_q**2) / rms)

    # set up radial model for expected EVPA (NCP through East)
    model_evpa = wrap_angle90(SIGNCONV * (np.rad2deg(np.arctan2(yy,xx)) + 90.)) * mask

    selvec = np.logical_not(np.logical_or(np.isnan(mask), np.isnan(measured_hvphaseang))).ravel()
    normw = measured_weight.ravel()[selvec] / np.sum(measured_weight.ravel()[selvec])
    errvec = angle_diff(model_evpa.ravel()[selvec],
                        measured_evpa.ravel()[selvec])
    if np.abs(np.percentile(errvec, 75.0) - np.percentile(errvec, 25.0)) > \
        IQS_SPREAD_CUTOFF * 180:
        log.warn("Noisy slice at {0:.3f} GHz. Discarding due to IQS offset".format(crfreq))
        continue
    if np.abs(skew(errvec)) > ABS_SKEW_CUTOFF:
        log.warn("Slice offsets are signficantly tailed at {0:.3f} GHz. Discarding due to {1:s} skew of {2:.3f}".format(crfreq, "left" if skew(errvec) < 0 else "right", skew(errvec)))
        continue
    if np.abs(kurtosis(errvec)) > ABS_KURT_CUTOFF:
        log.warn("Slice offsets at {0:.3f} GHz have a high fisher kurtosis of {1:.3f}".format(crfreq, kurtosis(errvec)))
        continue
    log.info("At {0:.3f} GHz:".format(crfreq))
    log.info("\tSelected {0:.3f}% of the limb between {1:.3f} and {2:.3f} degrees".format(rim_sel_frac*100., DISK_TORUS_INNER, DISK_TORUS_OUTER))
    log.info("\tMean offset: {0:.3f}".format(np.average(errvec, weights=normw)))
    log.info("\t50% Percentile offset: {0:.3f}".format(np.percentile(errvec, 50.0)))
    log.info("\t25% Percentile offset: {0:.3f}".format(np.percentile(errvec, 25.0)))
    log.info("\t75% Percentile offset: {0:.3f}".format(np.percentile(errvec, 75.0)))
    log.info("\tSkew: {0:.3f}".format(skew(errvec)))
    log.info("\tKurtosis: {0:.3f}".format(kurtosis(errvec, fisher=True)))
    log.info("\tStd: {0:.3f}".format(np.std(errvec)))
    if DO_FIT_HV:
        log.info("\tMean HVphase: {0:.3f}".format(np.average(measured_hvphaseang.ravel()[selvec], weights=normw)))
        log.info("\tStd HVphase: {0:.3f}".format(np.std(measured_hvphaseang.ravel()[selvec])))
        log.info("\tSkew HVphase: {0:.3f}".format(skew(measured_hvphaseang.ravel()[selvec])))
        log.info("\tKurtosis HVphase: {0:.3f}".format(kurtosis(measured_hvphaseang.ravel()[selvec])))
        log.info("\t50% Percentile HVphase: {0:.3f}".format(np.percentile(measured_hvphaseang.ravel()[selvec], 50.0)))
        log.info("\t25% Percentile HVphase: {0:.3f}".format(np.percentile(measured_hvphaseang.ravel()[selvec], 25.0)))
        log.info("\t75% Percentile HVphase: {0:.3f}".format(np.percentile(measured_hvphaseang.ravel()[selvec], 75.0)))

    plane_q3_evpa[crfreq] = np.percentile(errvec,75.0)
    plane_q2_evpa[crfreq] = np.percentile(errvec,50.0)
    plane_q1_evpa[crfreq] = np.percentile(errvec,25.0)
    plane_mean_evpa[crfreq] = np.average(errvec, weights=normw)
    plane_std_evpa[crfreq] = np.std(errvec)

    all_hv_offset.append(measured_hvphaseang.ravel()[selvec])
    all_model_evpa.append(model_evpa.ravel()[selvec])
    all_measured_evpa.append(measured_evpa.ravel()[selvec])
    all_measured_weight.append(measured_weight.ravel()[selvec])
    all_mask.append(np.ones_like(all_measured_weight[-1], dtype=np.bool))
    all_lambda.append(constants.c.value /
                      (np.ones_like(all_measured_weight[-1]) * crfreq * 1e9))

    # dump a histogram of offsets for this freq bin
    if DEBUGPLOT:
        transform = Affine2D()
        transform.scale(scale)
        transform.translate(-npix*0.5*scale, -npix*0.5*scale)
        metadata = {
                "name": ['lon','lat'],
                "type": ['longitude','latitude'],
                "wrap": [180,None],
                "unit": [units.deg, units.deg],
                "format_unit": [None, None]
        }
        fig = plt.figure(figsize=(10,6))
        wcsax = WCSAxes(fig, [0.1, 0.1, 0.2, 0.5], aspect='equal',
                        transform=transform, coord_meta=metadata)
        fig.add_axes(wcsax)
        pl1 = wcsax.imshow(model_evpa[0,0,:,:], cmap='tab20b')
        wcsax.set_title("Lunar refraction Model")
        wcsax.grid(True)
        wcsax.coords['lon'].set_axislabel("Relative Ra [deg]")
        wcsax.coords['lat'].set_axislabel("Relative Dec [deg]")
        wcsax.set_ylim(npix//2-1024,npix//2+1024)
        wcsax.set_xlim(npix//2-1024,npix//2+1024)

        plt.colorbar(pl1,ax=wcsax)

        wcsax = WCSAxes(fig, [0.4, 0.1, 0.2, 0.5], aspect='equal',
                        transform=transform, coord_meta=metadata)
        fig.add_axes(wcsax)
        pl2 = wcsax.imshow(measured_evpa[0,0,:,:], cmap='tab20b')
        wcsax.set_title("Measured refraction")
        wcsax.grid(True)
        wcsax.coords['lon'].set_axislabel("Relative Ra [deg]")
        wcsax.coords['lat'].set_axislabel("Relative Dec [deg]")
        wcsax.set_ylim(npix//2-1024,npix//2+1024)
        wcsax.set_xlim(npix//2-1024,npix//2+1024)
        plt.colorbar(pl2,ax=wcsax)

        wcsax = WCSAxes(fig, [0.7, 0.1, 0.2, 0.5], aspect='equal',
                        transform=transform, coord_meta=metadata)
        fig.add_axes(wcsax)
        diff = angle_diff(model_evpa, measured_evpa)[0,0,:,:]
        pl3 = wcsax.imshow(diff,
                           vmin=np.nanpercentile(diff, 0.5),
                           vmax=np.nanpercentile(diff, 99.5),
                           cmap='tab20b')
        wcsax.set_title("Diff")
        wcsax.grid(True)
        wcsax.coords['lon'].set_axislabel("Relative Ra [deg]")
        wcsax.coords['lat'].set_axislabel("Relative Dec [deg]")
        wcsax.set_ylim(npix//2-1024,npix//2+1024)
        wcsax.set_xlim(npix//2-1024,npix//2+1024)
        plt.colorbar(pl3,ax=wcsax)

        plt.show()
    if DOPLOT:
        plt.figure(figsize=(8,5))
        plt.title("Offset distribution {0:.3f} GHz".format(crfreq))
        plt.hist(errvec, bins=50, weights=normw)
        plt.axvline(x=np.percentile(errvec, 50.0),
                    c="k",
                    dashes=(5, 2, 1, 2),
                    label="Median offset={0:.3f}$^o$".format(np.percentile(errvec, 50.0)))
        plt.axvline(x=np.mean(errvec),
                    c="k",
                    alpha=0.3,
                    label="Mean offset={0:.3f}$^o$".format(np.mean(errvec)))
        plt.xlabel("Angle offset [deg]")
        plt.ylabel("Weighted count")
        plt.legend()
        log.info("Saving 'Offsets.{0:.3f}GHz.png'...".format(crfreq))
        plt.savefig("Offsets.{0:.3f}GHz.png".format(crfreq))
        if DO_FIT_HV:
            plt.figure(figsize=(8,5))
            plt.title("HV phase distribution {0:.3f} GHz".format(crfreq))
            plt.hist(measured_hvphaseang.ravel()[selvec], bins=50, weights=normw)
            plt.xlabel("Angle offset [deg]")
            plt.ylabel("Weighted count")
            log.info("Saving 'HVphaseOffsets.{0:.3f}GHz.png'...".format(crfreq))
            plt.savefig("HVphaseOffsets.{0:.3f}GHz.png".format(crfreq))
        plt.close()
    fitted_slices += 1

log.info("Number of fitted slices: {}".format(fitted_slices))
if fitted_slices == 0:
    log.critical("No slices can be fitted - your data is probably too noisy")
    sys.exit(1)
if fitted_slices == 1:
    log.critical("Only a single slice can be fitted - your data is probably too noisy")
    sys.exit(1)

def __error_func(argvec, d, m, w, mask, lda):
    offset, fd = argvec
    # RM measure is assumed to be from a thin Faraday screen and constant
    # across the lunar disk. We follow the convention in BJ Burn 1966
    # our definition of EVPA is half a radian in the range
    # therefore fitting for RM on the EVPA gives half the RM in radians / m2
    rmphase = np.rad2deg(lda.ravel()[mask]**2 * fd)
    # note the offset is fitted for 2 * angle because that is how it is applied
    # in the rotation matrix together with the parallactic angle rotation!
    return w.ravel()[mask] * angle_diff(m.ravel()[mask] + 2 * offset + rmphase, d.ravel()[mask])

def __hv_error_func(argvec, d, m, w, mask, lda):
    offset,gradient = argvec
    return w.ravel()[mask] * angle_diff(offset + gradient*lda, d.ravel()[mask])


def corrective_term(nu_considered, offset, fd, d, w, lda, mask, cov,
                    planeevpa_q1,
                    planeevpa_q2,
                    planeevpa_q3,
                    planeevpa_mean,
                    planeevpa_std,
                    ndraws=50000):
    lda_linspace = np.linspace(constants.c.value / np.max(list(nu_considered)),
                               constants.c.value / np.min(list(nu_considered)),
                               1024)

    rmphase = np.rad2deg(lda_linspace**2 * fd)
    corrective_phase_band = -(2.0 * offset + rmphase)
    L = np.linalg.cholesky(cov + 1e-10*np.eye(2))
    sampled_lda = []
    sampled_cov = []
    for di in np.random.randint(low=0, high=mask.size, size=ndraws):
        xi = np.random.randn(2)
        off_sample, fd_sample = L.dot(xi)
        sample_lda = lda.ravel()[di]
        rmphasemean = np.rad2deg(sample_lda**2 * fd)
        dist_mean = (2.0 * offset + rmphasemean)
        rmphasecov = np.rad2deg(sample_lda**2 * fd_sample)
        sampled_lda.append(sample_lda)
        sampled_cov.append(-(rmphasecov + off_sample) - dist_mean)
    if DOPLOT:
        plt.figure()
        plt.scatter(constants.c.value / np.array(sampled_lda) * 1e-9,
                    np.array(sampled_cov), color="b")
        stdfit = np.sqrt(np.diag(cov))
        freqs = constants.c.value / lda_linspace * 1e-9
        plt.plot(freqs,
                wrap_angle90(corrective_phase_band),
                "b--", label=(r"Fit offset"))
        plt.errorbar(list(sorted(planeevpa_q2.keys())),
                    [-planeevpa_mean[k] for k in sorted(planeevpa_q2.keys())],
                    yerr=[planeevpa_std[k] for k in sorted(planeevpa_q2.keys())],
                    color="None",
                    ecolor="k",
                    alpha=0.4,
                    capsize=5,
                    fmt="x",
                    label="Measured EVPA distribution")
        plt.errorbar(list(sorted(planeevpa_q2.keys())),
                    [-planeevpa_q2[k] for k in sorted(planeevpa_q2.keys())],
                    marker="x",
                    alpha=0.4,
                    color="k")
        ax = plt.gca()
        freqs = np.sort(list(planeevpa_q1.keys()))
        freqsdiff = np.min(np.abs(freqs[1:] - freqs[:-1]))
        for i, f in enumerate(sorted(planeevpa_q1.keys())):
            q1 = -planeevpa_q1[f]   
            q3 = -planeevpa_q3[f]    
            p = patches.Rectangle((freqs[i]-freqsdiff*0.25,
                                   q1),
                                  freqsdiff*0.5,
                                  q3-q1,
                                  linewidth=1,
                                  edgecolor="k",
                                  facecolor='none')
            ax.add_patch(p)
        plt.xlabel("Freq [GHz]")
        plt.ylabel("Corrective EVPA [deg]")
        plt.legend()
        plt.savefig("Offset.corrective.png")
        plt.close()

normw = np.hstack(all_measured_weight).ravel() / np.sum(np.hstack(all_measured_weight).ravel())
kwargs={"d": np.hstack(all_measured_evpa).ravel(),
                     "m": np.hstack(all_model_evpa).ravel(),
                     "w": normw,
                     "lda": np.hstack(all_lambda).ravel(),
                     "mask": np.hstack(all_mask).ravel()
                    }
fitres = lsq(fun=__error_func, x0=[0,0], bounds=([-90,-10],[+90,+10]),
             kwargs=kwargs)
if not fitres.success > 0:
    log.critical("Failure during solve for parameters for EVPA")
    sys.exit(1)
cov = np.linalg.inv(np.dot(fitres.jac.T, fitres.jac)) # hessian inverse
cost = np.sum(__error_func(fitres.x, **kwargs)**2, dtype=np.float64)
N = np.sum(np.hstack(all_mask).ravel(), dtype=np.float64)
cov = cov * fitres.cost / (N / beam_area - 2)
stdfit = np.sqrt(np.diag(cov))
corrective_term(nu_considered, fitres.x[0], fitres.x[1], 
                np.hstack(all_measured_evpa).ravel(),
                np.hstack(all_measured_weight).ravel(),
                np.hstack(all_lambda).ravel(),
                np.hstack(all_mask).ravel(),
                cov,
                plane_q1_evpa,
                plane_q2_evpa,
                plane_q3_evpa,
                plane_mean_evpa,
                plane_std_evpa)

log.info("Fitted feed angle offset: {0:.3f} +/- {1:.3f} deg".format(fitres.x[0], stdfit[0]))
log.info("Fitted ionospheric RM: {0:.3f} +/- {1:.3f} rad/m^2".format(fitres.x[1], stdfit[1]))
if DO_FIT_HV:
    fitres = lsq(fun=__hv_error_func, x0=[0,0], bounds=([-90,-np.inf],[+90,+np.inf]),
                kwargs={"d": np.hstack(all_hv_offset).ravel(),
                        "m": np.zeros_like(np.hstack(all_hv_offset).ravel()),
                        "w": normw,
                        "lda": np.hstack(all_lambda).ravel(),
                        "mask": np.hstack(all_mask).ravel()
                        })
    if not fitres.success > 0:
        log.critical("Failure during solve for parameters for HV phase")
        sys.exit(1)
    cov = np.linalg.inv(np.dot(fitres.jac.T, fitres.jac)) # hessian inverse
    cov = cov * fitres.cost / (normw.size - 2)
    stdfit = np.sqrt(np.diag(cov))
    log.info("Fitted HV: {0:.3f} +/- {1:.3f} deg".format(fitres.x[0], stdfit[0]))
    log.info("Fitted HV slope: {0:.3f} +/- {1:.3f} deg".format(fitres.x[1], stdfit[1]))

