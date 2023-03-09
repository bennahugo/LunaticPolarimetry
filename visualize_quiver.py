import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import meshgrid, arange
from astropy.wcs import WCS
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy import units
from matplotlib.transforms import Affine2D
from astropy.visualization.wcsaxes import WCSAxes
import os
import sys
from astropy import constants
import logging
import argparse
from utils import imstd, eval_mauch_beam, wrap_angle90, angle_diff

def create_logger():
    """ Create a console logger """
    log = logging.getLogger("EVPA plotter - quiver")
    cfmt = logging.Formatter(('%(name)s - %(asctime)s %(levelname)s - %(message)s'))
    log.setLevel(logging.DEBUG)
    log.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(cfmt)

    log.addHandler(console)

    return log, console, cfmt
log, log_console_handler, log_formatter = create_logger()


parser = argparse.ArgumentParser(description="EVPA plotter - quiver")
parser.add_argument("imagePattern", type=str, help="Pattern specifying which images to run fitter on. Expects each slice to conform to WSClean style output, e.g. "
                                                    "'lunarimgs/moon_snapshot-t0000-$xxx-{}-image.fits', where $xxx will be replaced by frequency slice numbers")
parser.add_argument("--torusout", "-o", dest="DISK_TORUS_OUTER", default=0.276040, type=float, help="Outer torus boundary for contributing EVPA angles (should be the outer limb radius in degrees)")
parser.add_argument("--torusin", "-i", dest="DISK_TORUS_INNER", default=0.25, type=float, help="Inner torus boundary for contributing EVPA angles (should exclude reflected RFI)")
parser.add_argument("--torusFillCutoff",  dest="TORUS_FILL_CUTOFF", default=0.35, type=float, help="Discard slices with fewer than this fractional number of points meeting SNR criteria within the torus (mostly empty torii). Expect 0 <= x <= 1.0")
parser.add_argument("--correctfeed", "-cf", dest="CORRECTIVE_FEED_ANGLE", default=0.0, type=float, help="Add corrective feed angle - as fited. Default 0.0")
parser.add_argument("--correctfd", "-cfd", dest="CORRECTIVE_FARADAY_DEPTH", default=0.0, type=float, help="Add corrective faraday depth / RM - as fited. Default 0.0")
parser.add_argument("--quiverspacing", dest="QUIVER_SPACE", default=30, type=int, help="Aimed quiver spacing in pixels")
parser.add_argument("--snr", "-s", dest="SNR_CUTOFF", default=10., type=float, help="SNR cutoff to apply to Stokes P and I (per channel image)")
parser.add_argument("--obsPAOffset",  dest="OBS_PA_OFFSET", default=0.0, type=float, help="Apply offset rotation (e.g. uncorrected parallactic angle) before plotting")
parser.add_argument("--minPFrac",  dest="FIT_MIN_STOKES_P", default=0.01, type=float, help="Cutoff fractional polarization contribution to the fit below this")
parser.add_argument("--maxPFrac",  dest="FIT_MAX_STOKES_P", default=1.00, type=float, help="Cutoff fractional polarization contribution (default 1.0)")
parser.add_argument("--quiverScale",  dest="QUIVER_SCALE", default=16., type=float, help="Quiver scale (default 16.0)")
parser.add_argument("--verbose", "-v", dest="VERBOSE", action='store_true', help="Increase verbosity")
args = parser.parse_args()

fio,fqo,fuo,fvo = [args.imagePattern.format(st) for st in "IQUV"]

DISK_TORUS_OUTER = args.DISK_TORUS_OUTER
DISK_TORUS_INNER = args.DISK_TORUS_INNER
FIT_MIN_STOKES_P = args.FIT_MIN_STOKES_P
FIT_MAX_STOKES_P = args.FIT_MAX_STOKES_P
CORRECT_FA = args.CORRECTIVE_FEED_ANGLE
CORRECT_FD = args.CORRECTIVE_FARADAY_DEPTH
TORUS_FILL_CUTOFF = args.TORUS_FILL_CUTOFF
QUIVER_SCALE = args.QUIVER_SCALE

VERBOSE = args.VERBOSE
QUIVER_SPACE = np.abs(args.QUIVER_SPACE)
log.info(("Using corrective feed angle: {0:0.3f} deg".format(CORRECT_FA)))
log.info(("Using corrective Faraday depth {0:0.3f} rad/m^2".format(CORRECT_FD)))

SNR_CUTOFF = args.SNR_CUTOFF
OBS_PA_OFFSET = args.OBS_PA_OFFSET
log.info("Using corrective PA rotation {0:0.3f} deg".format(OBS_PA_OFFSET))
log.info("Quivers will be plotted for SNR of {0:0.2f}x".format(SNR_CUTOFF))
fl = 0 
for nui in list(map(lambda a:"{0:04d}".format(a), range(9999))) + ["MFS"]:
    fi = fio.replace("$xxx",nui)
    fq = fqo.replace("$xxx",nui)
    fu = fuo.replace("$xxx",nui)
    fv = fvo.replace("$xxx",nui)

    if not all([os.path.exists(f) for f in [fi,fq,fu,fv]]):
        continue
    log.info(f"Pattern will load '{fi}'")
    fl += 1
    hdu = fits.open(fi)[0]
    wcs = WCS(hdu.header)
    bmaj = hdu.header["BMAJ"] * 3600
    bmin = hdu.header["BMIN"] * 3600
    
    npix = wcs.pixel_shape[0]
    assert wcs.pixel_shape[0] == wcs.pixel_shape[1]
    assert wcs.naxis == 4
    scale = np.abs(wcs.pixel_scale_matrix[0,0])
    crfreq = (wcs.wcs.crval[2] * 1e-9)
    #mjy2K = 1222 / ((wcs.wcs.crval[2] * 1e-9)**2*bmaj*bmin)
    mauchbeam = eval_mauch_beam(npix, scale, crfreq)
    feedcorr = np.deg2rad(2 * CORRECT_FA) + \
               ((constants.c.value / (crfreq * 1.0e9))**2 * CORRECT_FD)
    
    i = hdu.data
    u = fits.open(fu)[0].data
    q = fits.open(fq)[0].data
    v = fits.open(fv)[0].data
    PA = np.deg2rad(OBS_PA_OFFSET)

    Qcorr = q * np.cos(-2*PA) + u * np.sin(-2*PA)
    Ucorr = u * np.cos(-2*PA) - q * np.sin(-2*PA)
    q = Qcorr
    u = Ucorr

    ############################
    ### PLOT HISTOGRAM
    ############################
    p = (u**2 + q**2) / i**2
    rms = imstd(v) # detect rms in stokes V
    if VERBOSE:
        log.info(f"Estimated background noise as {rms*1e6:.3f} muJy")

    isnrmask = i > rms * SNR_CUTOFF
    psnrmask = np.logical_and(np.logical_and(p > FIT_MIN_STOKES_P, p <= FIT_MAX_STOKES_P),
                              np.sqrt(u**2 + q**2) > rms * SNR_CUTOFF)
    xx, yy = meshgrid((arange(npix)-npix//2)*scale,
                      (arange(npix)-npix//2)*scale)
    moonmask = xx**2 + yy**2 < DISK_TORUS_OUTER**2
    rim_mask = np.logical_and(xx**2 + yy**2 < DISK_TORUS_OUTER**2,
                              xx**2 + yy**2 > DISK_TORUS_INNER**2)
    if VERBOSE:
        log.info(f"Number of points in rim mask: {np.sum(rim_mask)}")
        log.info(f"Number of points in stokes P mask: {np.sum(psnrmask)}")
        log.info(f"Number of points in stokes I mask: {np.sum(isnrmask)}")

    mask = rim_mask * isnrmask * psnrmask
    rim_sel_frac = np.sum(mask) * 1.0 / np.sum(rim_mask)
    if rim_sel_frac < TORUS_FILL_CUTOFF:
        log.warn(f"Only {rim_sel_frac*100.:.2f}% of the limb is of sufficient SNR, "
                 f"skipping slice at {crfreq:.3f} GHz")
        continue
    if mask.sum() == 0:
        log.warn("Empty / noisy slice at {0:.3f} GHz".format(crfreq))
        continue
    mask = np.float64(mask)
    mask[mask == 0] = np.nan

    masked_q = q.copy()[0,0,:,:] * mask
    masked_u = u.copy()[0,0,:,:] * mask
    masked_i = i.copy()[0,0,:,:] * mask

    measured_evpa = 0.5 * np.rad2deg(np.arctan2(masked_u, masked_q)) - np.rad2deg(feedcorr)
    measured_p = (masked_u**2 + masked_q**2) / masked_i
    measured_weight = (np.sqrt(masked_u**2 + masked_q**2) / rms)

    # set up radial model for expected EVPA (North through East)
    model_evpa = wrap_angle90(np.rad2deg(np.arctan2(yy,xx)) + 90.) * mask

    selvec = np.logical_not(np.isnan(mask)).ravel()
    normw = measured_weight.ravel()[selvec] / np.sum(measured_weight.ravel()[selvec])
    errvec = angle_diff(model_evpa.ravel()[selvec],
                        measured_evpa.ravel()[selvec])
    log.info(f"At {crfreq} GHz:")
    log.info("\tMean offset: {0:.3f}".format(np.average(errvec, weights=normw)))
    log.info("\t50% Percentile offset: {0:.3f}".format(np.percentile(errvec, 50.0)))
    log.info("\t25% Percentile offset: {0:.3f}".format(np.percentile(errvec, 25.0)))
    log.info("\t75% Percentile offset: {0:.3f}".format(np.percentile(errvec, 75.0)))

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
                label="Residual weighted mean offset={0:.3f}$^o$".format(np.average(errvec, weights=normw)))
    plt.axvline(x=np.mean(errvec),
                c="k",
                alpha=0.3,
                dashes=(2, 1, 2, 1),
                label="Prior weighted mean offset={0:.3f}$^o$".format(np.average(errvec + np.rad2deg(feedcorr),
                                                                                 weights=normw)))
    plt.xlabel("Angle offset [deg]")
    plt.ylabel("Weighted count")
    plt.legend()
    log.info("Saving 'Offsets.corrected.{0:.3f}GHz.png'...".format(crfreq))
    plt.savefig("Offsets.corrected.{0:.3f}GHz.png".format(crfreq))
    

    ######################################
    # Plot quiver plot over Stokes I
    ######################################
    transform = Affine2D()
    transform.scale(-scale,scale)
    transform.translate(npix*0.5*scale,
                        -npix*0.5*scale)
    metadata = {
            "name": ['lon','lat'],
            "type": ['longitude','latitude'],
            "wrap": [180,None],
            "unit": [units.deg, units.deg],
            "format_unit": [None, None]
    }

    wcsslice = wcs.slice(np.s_[0,0,:,:])
    
    fig = plt.figure()
    ax = WCSAxes(fig, [0.1, 0.1, 0.8, 0.8], aspect='equal',
                             transform=transform, coord_meta=metadata)
    ax.set_title("Moon at {0:.3f} GHz".format(crfreq))

    fig.add_axes(ax)
    hdu.data[hdu.data == 0.0] = np.nan
    i_plot = ax.imshow(i[0,0,:,:] / mauchbeam * 1e3, origin='lower', cmap='coolwarm',
                       )
    cbar = fig.colorbar(i_plot)
    cbar.set_label("mJy/beam")
    ax.coords['lon'].set_axislabel("Relative RA [<---EAST] [deg]")
    ax.coords['lat'].set_axislabel("Relative DEC [NORTH--->] [deg]")
    
    ax.grid()
    
    # add quivers
    
    xx0, xx1 = ax.get_xlim()
    yy0, yy1 = ax.get_ylim()
    factors = np.array(list(filter(lambda x: npix % x == 0, np.arange(npix) + 1)))
    factor = factors[np.argmin(np.abs(factors - QUIVER_SPACE))] # closest divisor to given pixel spacing for quivers
    nx_new = npix // factor
    ny_new = npix // factor
    X,Y = np.meshgrid(np.linspace(xx0,xx1,nx_new,endpoint=True),
                                        np.linspace(yy0,yy1,ny_new,endpoint=True))
    delta_bin = X[0,1] - X[0,0]
    masked_q = q.copy()[0,0,:,:]
    masked_u = u.copy()[0,0,:,:]
    masked_i = i.copy()[0,0,:,:]
    mask = np.float64(rim_mask * isnrmask * psnrmask)
    mask[np.abs(mask) < 1.0e-3] = np.nan

    masked_q = q * mask
    masked_u = u * mask
    masked_i = i * mask

    I_bin = np.nanmedian(masked_i.reshape(nx_new, factor,
                                          ny_new, factor),
                         axis=(3,1))
    Q_bin = np.nanmedian(masked_q.reshape(nx_new, factor,
                                          ny_new, factor),
                         axis=(3,1))
    U_bin = np.nanmedian(masked_u.reshape(nx_new, factor,
                                          ny_new, factor),
                         axis=(3,1))
  
    # polarization angle
    psi = 0.5*np.arctan2(U_bin, Q_bin)
    
    # polarization fraction
    frac = np.sqrt(Q_bin**2+U_bin**2)/I_bin
    
    pixX = -frac*np.sin(psi - feedcorr) # X-vector
    pixY = frac*np.cos(psi - feedcorr) # Y-vector
    
    # keyword arguments for quiverplots
    quiveropts = dict(headlength=0, headwidth=1, pivot='middle')
    ax.quiver(X + delta_bin * 0.5, Y + delta_bin * 0.5, 
              pixX, pixY,
              scale=QUIVER_SCALE, **quiveropts)
    
    ax.set_ylim(npix//2-1024,npix//2+1024)
    ax.set_xlim(npix//2-1024,npix//2+1024)
    log.info("Saving 'Moon.{0:.3f}GHz.png'...".format(crfreq))
    plt.savefig("Moon.{0:.3f}GHz.png".format(crfreq))

    ###################################
    # Polarization fraction map
    ###################################
    fig = plt.figure()
    ax = WCSAxes(fig, [0.1, 0.1, 0.8, 0.8], aspect='equal',
                             transform=transform, coord_meta=metadata)
    ax.set_title("Moon at {0:.3f} GHz".format(crfreq))

    fig.add_axes(ax)
    mask = np.float64(moonmask * psnrmask * isnrmask)
    mask[mask==0.0] = np.nan
    p_plot = ax.imshow((p * mask)[0,0,:,:] * 100, origin='lower', cmap='tab20b',
                       vmin=0, vmax=50)
    cbar = fig.colorbar(p_plot)
    cbar.set_label("Fractional polarization [%]")
    ax.coords['lon'].set_axislabel("Relative RA [<---EAST] [deg]")
    ax.coords['lat'].set_axislabel("Relative DEC [NORTH--->] [deg]")
    
    ax.grid()
    ax.set_ylim(npix//2-1024,npix//2+1024)
    ax.set_xlim(npix//2-1024,npix//2+1024)
    log.info("Saving 'Moon.{0:.3f}GHz.polfrac.png'...".format(crfreq))
    plt.savefig("Moon.{0:.3f}GHz.polfrac.png".format(crfreq))
log.info(f"Loaded {fl} files matching pattern")
