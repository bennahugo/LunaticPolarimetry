#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as plt
import pandas 
from mpl_toolkits.basemap import Basemap
import sys

loaded_maps = {}

for ff, title, degs, transpose, negated_axis  in (zip(["aips_magdip.txt", "aips_igrf13.txt", "tony_igrf.txt", "casa_igrf.txt", "rmextract_emm.txt"],
                                                      ["AIPS MAGDIP","AIPS IGRFv13", "ALBUS IGRFv13", "CASA IGRFv12", "RMExtract WMM"],
                                                      [False,False,False,False,False],
                                                      [False,False,False,False,False],
                                                      [False,True,False,False,False])):
    csv = pandas.read_csv(ff)
    lat = np.deg2rad(csv["LAT"]) if degs else csv["LAT"]
    lon = np.deg2rad(csv["LON"]) if degs else csv["LON"]
    bfield = csv["STRENGTH(nT)"]
    # interpret as lon x lat map, c order
    nlon = len(np.unique(lon))
    nlat = len(np.unique(lat))
    if transpose:
        bfield = bfield.values.reshape(nlat, nlon)
    else:
        bfield = bfield.values.reshape(nlon, nlat).T
    xx, yy = np.meshgrid(np.rad2deg(np.unique(lon)), np.rad2deg(np.unique(lat)))
    if negated_axis:
        bfield = bfield[::-1,:]
    plt.figure(figsize=(10,10))
    plt.imshow(bfield)
    plt.savefig(ff + "bfieldheat.png")
    plt.close()
    def fmt(x):
        return f"{x:.0f} nT"
    # create new figure, axes instances.
    fig=plt.figure(figsize=(10, 6))
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    # setup mercator map projection.
    m = Basemap(projection='mill',lon_0=0.)
    #m.drawcoastlines()
    m.fillcontinents()
    # draw parallels
    m.drawparallels(np.linspace(-89,89,6),labels=[1,1,0,1])
    # draw meridians
    m.drawmeridians(np.linspace(-179,179,9),labels=[1,1,0,1])    
    CS = m.contour(xx, yy, bfield, latlon=True, levels=20, cmap='tab20b')
    ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)
    plt.title(title)
    plt.savefig(ff + ".png")
    plt.close()

    loaded_maps[title] = bfield
    loaded_maps.setdefault("xx", xx)
    loaded_maps.setdefault("yy", yy)

bfieldv13 = loaded_maps["AIPS IGRFv13"]
for ff, title in zip(["aips_magdip.txt", "aips_igrf13.txt", "casa_igrf.txt", "rmextract_emm.txt"],
                     ["AIPS MAGDIP", "AIPS IGRFv13", "CASA IGRFv12", "RMExtract WMM"]):
    xx = loaded_maps["xx"]
    yy = loaded_maps["yy"]
    bfield = (loaded_maps[title] - bfieldv13) * 100. / (bfieldv13.max() - bfieldv13.min())
    fig=plt.figure(figsize=(10, 6))
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    # setup mercator map projection.
    m = Basemap(projection='mill',lon_0=0.)
    #m.drawcoastlines()
    m.fillcontinents()
    # draw parallels
    #m.drawparallels(np.linspace(-89,89,6),labels=[1,1,0,1])
    # draw meridians
    #m.drawmeridians(np.linspace(-179,179,9),labels=[1,1,0,1])    
    CS = m.contour(xx, yy, bfield, latlon=True, levels=20, cmap='tab20b')
    def fmt(x):
        return f"{x:.0f} %"
    ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)
    plt.title(title)
    plt.savefig(ff + ".diff.png")
    plt.close()