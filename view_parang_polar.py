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
try:
    print("Using corrective feed angle: {0:0.3f} deg".format(float(sys.argv[1])))
    print("Using corrective Faraday depth {0:0.3f} rad/m^2".format(float(sys.argv[2])))
except:
    print("Usage view_parang_polar.py feed_offset_deg ionospheric_fd")


def imstd(polmap, nbox=100):
    nx_new = polmap.shape[3] // nbox
    ny_new = polmap.shape[2] // nbox
    boxrms = np.nanstd(polmap.reshape(nx_new, nbox, ny_new, nbox), 
                       axis=(1,3))
    return np.min(boxrms)

def wrap_angle90(x):
    return (x + 90.) % 180. - 90.

def angle_diff( angle1, angle2 ):
    diff = ( angle2 - angle1 + 90 ) % 180 - 90
    return ((diff + 180) * (diff < -90)) + diff

arr_lat = -30.71316944
SNR_CUTOFF = 20.0

#fio,fqo,fuo = ["imgs/robust+2.0.coretapered-xxx-{}-image.fits".format(st) for st in "IQU"]
#fio,fqo,fuo = ["imgs/robust-0.3.untapered.sc0-xxx-{}-image.fits".format(st) for st in "IQU"]
#fio,fqo,fuo = ["imgs/robust-0.3.untapered.finechan.sc0-xxx-{}-image.fits".format(st) for st in "IQU"]
#fio,fqo,fuo = ["moon/moon_avg-xxx-{}-image.fits".format(st) for st in "IQU"]
fio,fqo,fuo = ["moonwithlunarX/moon_snapshot-t0000-xxx-{}-image.fits".format(st) for st in "IQU"]


for nui in list(map(lambda a:"{0:04d}".format(a), range(50))) + ["MFS"]:
#for nui in ["MFS"]:
    fi = fio.replace("xxx",nui)
    fq = fqo.replace("xxx",nui)
    fu = fuo.replace("xxx",nui)
    if not all([os.path.exists(f) for f in [fi,fq,fu]]):
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
    #mjy2K = 1222 / ((wcs.wcs.crval[2] * 1e-9)**2*bmaj*bmin)
    x = arange(-npix//2,+npix//2) * scale / 60 * crfreq
    mauchbeam = 1 - \
                            0.3514 * np.abs(x**2) / 10**3 + \
                            0.5600 * np.abs(x**2)**2 / 10**7 - \
                            0.0474 * np.abs(x**2)**3 / 10**10 + \
                            0.00078 * np.abs(x**2)**4/10**13 + \
                            0.00019 * np.abs(x**2)**5 / 10**16
    feedcorr = np.deg2rad(2 * float(sys.argv[1])) + \
               (2.0 * (constants.c.value / (crfreq * 1.0e9))**2 * float(sys.argv[2]))
    
    i = hdu.data
    u = fits.open(fu)[0].data
    q = fits.open(fq)[0].data

    PA = np.deg2rad(-0.5695)

    Qcorr = q * np.cos(-2*PA) + u * np.sin(-2*PA)
    Ucorr = u * np.cos(-2*PA) - q * np.sin(-2*PA)
    q = Qcorr
    u = Ucorr

    ############################
    ### PLOT HISTOGRAM
    ############################
    p = (u**2 + q**2) / i**2
    rms = imstd(np.sqrt(u**2 + q**2)) # detect rms in stokes P
    isnrmask = i > rms * SNR_CUTOFF

    psnrmask = np.logical_and(np.logical_and(p > 0.02, p <= 1.0),
                              np.sqrt(u**2 + q**2) > rms * SNR_CUTOFF)
    xx, yy = meshgrid((arange(npix)-npix//2)*scale,
                      (arange(npix)-npix//2)*scale)

    rim_mask = np.logical_and(xx**2 + yy**2 < 0.276040**2,
                              xx**2 + yy**2 > 0.200000**2)
    mask = np.float64(rim_mask * isnrmask * psnrmask)
    if mask.sum() == 0:
        print("Empty / noisy slice at {0:.3f} GHz".format(crfreq))
        continue
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
    print("Mean offset: {0:.3f} @ {1:.3f} GHz".format(np.average(errvec, weights=normw), crfreq))
    print("50% Percentile offset: {0:.3f} @ {1:.3f} GHz".format(np.percentile(errvec, 50.0), crfreq))
    print("25% Percentile offset: {0:.3f} @ {1:.3f} GHz".format(np.percentile(errvec, 25.0), crfreq))
    print("75% Percentile offset: {0:.3f} @ {1:.3f} GHz".format(np.percentile(errvec, 75.0), crfreq))

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
    print("Saving 'Offsets.corrected.{0:.3f}GHz.png'...".format(crfreq))
    plt.savefig("Offsets.corrected.{0:.3f}GHz.png".format(crfreq))
    

    ######################################
    # Plot quiver plot over Stokes I
    ######################################
    transform = Affine2D()
    transform.scale(scale)
    transform.translate(-npix*0.5*scale,
                        -npix*0.5*scale)
    transform.rotate(arr_lat)    # radians
    metadata = {
            "name": ['lon','lat'],
            "type": ['longitude','latitude'],
            "wrap": [180,None],
            "unit": [units.deg, units.deg],
            "format_unit": [None, None]
    }

    wcsslice = wcs.slice(np.s_[0,0,:,:])
    moonmask = (meshgrid((arange(npix)-npix//2)*scale,(arange(npix)-npix//2)*scale)[0]**2 + 
                            meshgrid((arange(npix)-npix//2)*scale,(arange(npix)-npix//2)*scale)[1]**2 < 0.276040**2).reshape(wcs.pixel_shape[::-1])
    rms = imstd(np.sqrt(u**2 + q**2))
    isnrmask = i > rms * 5.0
    p = (u**2 + q**2) / i**2
    psnrmask = np.logical_and(np.logical_and(p > 0.02, p <= 1.0),
                              np.sqrt(u**2 + q**2) > rms * 5.0)
    
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
    ax.coords['lon'].set_axislabel("Relative Az [East through West] [deg]")
    ax.coords['lat'].set_axislabel("Relative Elev [deg]")
    
    ax.grid()
    
    # add quivers
    
    xx0, xx1 = ax.get_xlim()
    yy0, yy1 = ax.get_ylim()
    factor = [30, 30]
    nx_new = npix // factor[0]
    ny_new = npix // factor[1]
    X,Y = np.meshgrid(np.linspace(xx0,xx1,nx_new,endpoint=True),
                                        np.linspace(yy0,yy1,ny_new,endpoint=True))
    delta_bin = X[0,1] - X[0,0]
    masked_q = q.copy()[0,0,:,:]
    masked_u = u.copy()[0,0,:,:]
    masked_i = i.copy()[0,0,:,:]
    mask = np.float64(moonmask * isnrmask * psnrmask)
    mask[np.abs(mask) < 1.0e-3] = np.nan

    masked_q = q * mask
    masked_u = u * mask
    masked_i = i * mask

    I_bin = np.nanmedian(masked_i.reshape(nx_new, factor[0],
                                          ny_new, factor[1]),
                         axis=(3,1))
    Q_bin = np.nanmedian(masked_q.reshape(nx_new, factor[0],
                                          ny_new, factor[1]),
                         axis=(3,1))
    U_bin = np.nanmedian(masked_u.reshape(nx_new, factor[0],
                                          ny_new, factor[1]),
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
              scale=16, **quiveropts)
    
    ax.set_ylim(npix//2-1024,npix//2+1024)
    ax.set_xlim(npix//2-1024,npix//2+1024)
    print("Saving 'Moon.{0:.3f}GHz.png'...".format(crfreq))
    plt.savefig("Moon.{0:.3f}GHz.png".format(crfreq))

    ###################################
    # Polarization fraction map
    ###################################
    rms = imstd(np.sqrt(u**2 + q**2))
    isnrmask = i > rms * 10.0
    p = (u**2 + q**2) / i**2
    psnrmask = np.logical_and(np.logical_and(p > 0.02, p <= 1.0),
                              np.sqrt(u**2 + q**2) > rms * 10.0)

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
    ax.coords['lon'].set_axislabel("Relative Az [East through West] [deg]")
    ax.coords['lat'].set_axislabel("Relative Elev [deg]")
    
    ax.grid()
    ax.set_ylim(npix//2-1024,npix//2+1024)
    ax.set_xlim(npix//2-1024,npix//2+1024)
    print("Saving 'Moon.{0:.3f}GHz.polfrac.png'...".format(crfreq))
    plt.savefig("Moon.{0:.3f}GHz.polfrac.png".format(crfreq))
