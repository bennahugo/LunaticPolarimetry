#!/usr/bin/python3

import numpy as np
from numpy import meshgrid, arange
from astropy.io import fits
from astropy import units
from astropy.wcs import WCS
import logging
import argparse

def create_logger():
    """ Create a console logger """
    log = logging.getLogger("TorusMask")
    cfmt = logging.Formatter(('%(name)s - %(asctime)s %(levelname)s - %(message)s'))
    log.setLevel(logging.DEBUG)
    log.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(cfmt)

    log.addHandler(console)

    return log, console, cfmt
log, log_console_handler, log_formatter = create_logger()


parser = argparse.ArgumentParser(description="TorusMask - a program to make a torus mask for cleaning")
parser.add_argument("--torusout", "-o", dest="DISK_TORUS_OUTER", default=0.276040, type=float, help="Outer torus boundary for contributing EVPA angles (should be the outer limb radius in degrees)")
parser.add_argument("--torusin", "-i", dest="DISK_TORUS_INNER", default=0.25, type=float, help="Inner torus boundary for contributing EVPA angles (should exclude reflected RFI)")
parser.add_argument("image", type=str, help="Input dirty map with the WCS to be used for the mask")
parser.add_argument("output", type=str, help="Output name for mask")
parser.add_argument("-f", "--override", action="store_true", help="Override mask if exists")


args = parser.parse_args()

DISK_TORUS_OUTER = args.DISK_TORUS_OUTER
DISK_TORUS_INNER = args.DISK_TORUS_INNER

log.info(f"Mask outer radius: {DISK_TORUS_OUTER}")
log.info(f"Mask inner radius: {DISK_TORUS_INNER}")
log.info(f"Using '{args.image}' as input WCS")
log.info(f"Outputting mask to '{args.output}'")


im = fits.open(args.image)
hdu = im[0]
wcs = WCS(hdu.header)
npix = wcs.pixel_shape[0]
assert wcs.pixel_shape[0] == wcs.pixel_shape[1]
assert wcs.naxis == 4
scale = np.abs(wcs.pixel_scale_matrix[0,0])

xx, yy = meshgrid((arange(npix)-npix//2)*scale,
                  (arange(npix)-npix//2)*scale)

rim_mask = np.logical_and(xx**2 + yy**2 < DISK_TORUS_OUTER**2,
                          xx**2 + yy**2 > DISK_TORUS_INNER**2)

im[0].data[...] = rim_mask.repeat(hdu.shape[0] * hdu.shape[1]).reshape(rim_mask.shape)
im.writeto(args.output, overwrite=args.override)
im.close()
log.info(f"Written mask to '{args.output}'")
