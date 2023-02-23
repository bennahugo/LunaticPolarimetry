#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as plt
import pandas 
from mpl_toolkits.basemap import Basemap
import sys

for ff, title in (zip(["aips_magdip.txt", "tony_igrf.txt", "casa_igrf.txt"],["AIPS MAGDIP", "ALBUS IGRFv13", "CASA IGRFv12"])):
    csv = pandas.read_csv(ff)
    lat = csv["LAT"]
    lon = csv["LON"]
    bfield = csv["STRENGTH(nT)"]
    # interpret as lon x lat map, c order
    nlon = len(np.unique(lon))
    nlat = len(np.unique(lat))
    bfield = bfield.values.reshape(nlon, nlat).T

    xx, yy = np.meshgrid(np.rad2deg(np.unique(lon)), np.rad2deg(np.unique(lat)))
    def fmt(x):
        return f"{x:.0f} nT"
    # create new figure, axes instances.
    fig=plt.figure(figsize=(10, 6))
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    # setup mercator map projection.
    m = Basemap(projection='mill',lon_0=0.)
    m.drawcoastlines()
    m.fillcontinents()
    # draw parallels
    m.drawparallels(np.arange(-89.9999,89.9999,36),labels=[1,1,0,1])
    # draw meridians
    m.drawmeridians(np.arange(-179.9999,179.9999,36),labels=[1,1,0,1])
    CS = m.contour(xx, yy, bfield, latlon=True, levels=20,)
    ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)
    plt.title(title)
    plt.show()
