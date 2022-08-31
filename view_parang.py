import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy import units as u
from matplotlib.transforms import Affine2D
from astropy.visualization.wcsaxes import WCSAxes
import os
import logging
import argparse
def create_logger():
    """ Create a console logger """
    log = logging.getLogger("EVPA plotter - colourwheel")
    cfmt = logging.Formatter(('%(name)s - %(asctime)s %(levelname)s - %(message)s'))
    log.setLevel(logging.DEBUG)
    log.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(cfmt)

    log.addHandler(console)

    return log, console, cfmt
log, log_console_handler, log_formatter = create_logger()


parser = argparse.ArgumentParser(description="EVPA plotter - colourwheel")
parser.add_argument("imagePattern", type=str, help="Pattern specifying which images to run fitter on. Expects each slice to conform to WSClean style output, e.g. "
                                                    "'lunarimgs/moon_snapshot-t0000-$xxx-{}-image.fits', where $xxx will be replaced by frequency slice numbers")
parser.add_argument("--torusout", "-o", dest="DISK_TORUS_OUTER", default=0.276040, type=float, help="Outer torus boundary for contributing EVPA angles (should be the outer limb radius in degrees)")
parser.add_argument("--torusin", "-i", dest="DISK_TORUS_INNER", default=0.15, type=float, help="Inner torus boundary for contributing EVPA angles (should exclude reflected RFI)")

args = parser.parse_args()

fio,fqo,fuo,fvo = [args.imagePattern.format(st) for st in "IQUV"]

for nui in list(map(lambda a:"{0:04d}".format(a), range(9999))) + ["MFS"]:
    fi = fio.replace("$xxx",nui)
    fq = fqo.replace("$xxx",nui)
    fu = fuo.replace("$xxx",nui)
    if not all([os.path.exists(f) for f in [fi,fq,fu]]):
        continue
    log.info(f"Pattern will load '{fi}'")

    with fits.open(fi) as ifi:
        hdu = ifi[0]
        wcs = WCS(hdu.header)
        wcsslice = wcs.slice(np.s_[0,0,:,:])

        assert "RA" in hdu.header["CTYPE1"]
        assert "DEC" in hdu.header["CTYPE2"]
        assert "FREQ" in hdu.header["CTYPE3"]
        assert hdu.header["NAXIS1"] == hdu.header["NAXIS1"]
        assert hdu.header["NAXIS"] == 4
        npix = abs(hdu.header["NAXIS1"])
        scale = abs(hdu.header["CDELT1"])
        crfreq = hdu.header["CRVAL3"]
    transform = Affine2D()
    transform.scale(-scale, scale)
    transform.translate(npix*0.5*scale, -npix*0.5*scale)
    metadata = {
        "name": ['lon','lat'],
        "type": ['longitude','latitude'],
        "wrap": [180,None],
        "unit": [u.deg, u.deg],
        "format_unit": [None, None]
    }
    fig = plt.figure(figsize=(12, 12))
    ax = WCSAxes(fig, [0.1, 0.1, 0.8, 0.8], aspect='equal',
                 transform=transform, coord_meta=metadata)
    fig.add_axes(ax)
    with fits.open(fq) as ifq, fits.open(fu) as ifu:
        Q = ifq[0].data[0,0,:,:]
        U = ifu[0].data[0,0,:,:]
    ang = np.rad2deg(0.5*np.arctan2(U,Q))
    xx, yy = np.meshgrid((np.arange(npix)-npix//2)*scale,
                         (np.arange(npix)-npix//2)*scale)

    # apply rim mask
    rim_mask = np.logical_and(xx**2 + yy**2 < args.DISK_TORUS_OUTER**2,
                              xx**2 + yy**2 > args.DISK_TORUS_INNER**2)
    U[np.logical_not(rim_mask)] = 0.0
    Q[np.logical_not(rim_mask)] = 0.0

    ang[U == 0.0] = np.nan
    ang[Q == 0.0] = np.nan
    angles = ax.imshow(ang, vmin=-90, vmax=90, origin='lower', cmap='tab20b')
    cbar = fig.colorbar(angles)
    cbar.set_label("EVPA [deg]")
    ax.coords['lon'].set_axislabel("Relative RA [<---EAST] [deg]")
    ax.coords['lat'].set_axislabel("Relative DEC [NORTH--->] [deg]")
    ax.set_title("Measured EVPA @ {0:.3f}MHz".format(crfreq*1e-6))
    ax.grid()
    plt.savefig("Moon_EVPA_colourwheel.{0:.3f}MHz.png".format(crfreq*1e-6))
    log.info("Writing 'Moon_EVPA_colourwheel.{0:.3f}MHz.png'".format(crfreq*1e-6))
    plt.close()

