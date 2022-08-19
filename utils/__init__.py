import numpy as np
def imstd(polmap, nbox=50):
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

def eval_mauch_beam(npix, scale, crfreq):
    x = np.arange(-npix//2,+npix//2) * scale / 60 * crfreq
    mauchbeam = 1 - \
                0.3514 * np.abs(x**2) / 10**3 + \
                0.5600 * np.abs(x**2)**2 / 10**7 - \
                0.0474 * np.abs(x**2)**3 / 10**10 + \
                0.00078 * np.abs(x**2)**4/10**13 + \
                0.00019 * np.abs(x**2)**5 / 10**16
    return mauchbeam