#!/usr/bin/env python3

# This script does something similar to CASA fixvis
# but is more general in that you can ask it to compute
# UVW coordinates for a special non-sidereal (non-fixed ra/dec) body
# you must pull the following script (to the same root directory you are running from) 
# from XOVA before running this script
# https://raw.githubusercontent.com/ratt-ru/xova/master/xova/apps/xova/fixvis.py

import ephem
import numpy as np
import datetime
import argparse
import logging
from pyrap.tables import table as tbl
from pyrap.tables import taql
from pyrap.quanta import quantity
from pyrap.measures import measures
from astropy.coordinates import SkyCoord
from astropy import units
import pytz
import utils.fixvis as fixvis

def create_logger():
    """ Create a console logger """
    log = logging.getLogger("UVW corrector")
    cfmt = logging.Formatter(('%(name)s - %(asctime)s %(levelname)s - %(message)s'))
    log.setLevel(logging.DEBUG)
    log.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(cfmt)

    log.addHandler(console)

    return log, console, cfmt

log, log_console_handler, log_formatter = create_logger()

parser = argparse.ArgumentParser(description="UVW computer for moving target for MeerKAT")
parser.add_argument("ms", type=str, help="Database to correct")
parser.add_argument("--field", "-f", dest="field", type=int, default=0, help="Field index to correct")
parser.add_argument("--specialEphem", "-se", dest="ephem", default=None, type=str, help="Use special ephemeris body as defined in PyEphem")
parser.add_argument("--doPlot", "-dp", dest="plot", action="store_true", help="Make plots for specified field")
parser.add_argument("--simulate", "-s", dest="sim", action="store_true", help="Simulate only -- make no modifications to database")
parser.add_argument("--chunksize", "-cs", type=int, dest="chunksize", default=1000, help="Chunk size in rows")
parser.add_argument("--verbose", "-v", action="store_true", dest="verbose", help="Increase verbosity")

args = parser.parse_args()

if args.plot:
    from matplotlib import pyplot as plt
    import matplotlib.dates as mdates
    log.info("Enabling plotting")

meerkat = ephem.Observer()
meerkat.lat = "-30:42:47.41"
meerkat.long = "21:26:38.0"
meerkat.elevation = 1054
meerkat.epoch = ephem.J2000

with tbl(args.ms, ack=False) as t:
    with taql("select * from $t where FIELD_ID=={}".format(args.field)) as tt:
        def __to_datetimestr(t):
            dt = datetime.datetime.utcfromtimestamp(quantity("{}s".format(t)).to_unix_time())
            return dt.strftime("%Y/%m/%d %H:%M:%S")
        dbtime = tt.getcol("TIME_CENTROID")
        start_time_Z = __to_datetimestr(dbtime.min())
        end_time_Z = __to_datetimestr(dbtime.max())
        log.info("Observation spans '{}' and '{}' UTC".format(
                 start_time_Z, end_time_Z))
dm = measures()
meerkat.date = start_time_Z
st = meerkat.date
meerkat.date = end_time_Z
et = meerkat.date
TO_SEC = 3600*24.0
uniqdbtime = np.sort(np.unique(dbtime))
mintimedelta = np.min(np.abs(uniqdbtime[1:] - uniqdbtime[:-1]))
nstep = int(np.round((float(et)*TO_SEC - float(st)*TO_SEC) / (mintimedelta/10.0)))
log.info("Computing RADEC in {} steps of {}s each".format(nstep, mintimedelta/10.0))
timecoord = time = np.linspace(st,et,nstep)
timecoorddt = list(map(lambda x: ephem.Date(x).datetime(), time))

with tbl(args.ms+"::ANTENNA", ack=False) as t:
    anames = t.getcol("NAME")
    apos = t.getcol("POSITION")
    aposdm = list(map(lambda pos: dm.position('itrf',*[ quantity(x,'m') for x in pos ]),
                      apos))

if args.ephem:
    with tbl(args.ms+"::FIELD", ack=False) as t:
        fieldnames = t.getcol("NAME")
    fieldEphem = getattr(ephem, args.ephem, None)()
    if not fieldEphem:
        raise RuntimeError("Body {} not defined by PyEphem".format(args.ephem))
    log.info("Overriding stored ephemeris in database '{}' field '{}' by special PyEphem body '{}'".format(
        args.ms, fieldnames[args.field], args.ephem))
else:
    with tbl(args.ms+"::FIELD", ack=False) as t:
        fieldnames = t.getcol("NAME")
        pos = t.getcol("PHASE_DIR")
    skypos = SkyCoord(pos[args.field][0,0]*units.rad, pos[args.field][0,1]*units.rad, frame="fk5")
    rahms = "{0:.0f}:{1:.0f}:{2:.5f}".format(*skypos.ra.hms)
    decdms = "{0:.0f}:{1:.0f}:{2:.5f}".format(skypos.dec.dms[0], abs(skypos.dec.dms[1]), abs(skypos.dec.dms[2]))
    fieldEphem = ephem.readdb(",f|J,{},{},0.0".format(rahms, decdms))
    log.info("Using coordinates of field '{}' for body: J2000, {}, {}".format(fieldnames[args.field],
                                                                              np.rad2deg(pos[args.field][0,0]),
                                                                              np.rad2deg(pos[args.field][0,1])))

az = np.zeros(nstep, dtype=np.float32)
el = az.copy()
ra = az.copy()
#racc = az.copy()
dec = az.copy()
#deccc = az.copy()

for ti, t in enumerate(time):
    meerkat.date = t
    t_iso8601 = meerkat.date.datetime().strftime("%Y-%m-%dT%H:%M:%S.%f")
    fieldEphem.compute(meerkat)
    az[ti] = fieldEphem.az
    el[ti] = fieldEphem.alt
    ra[ti] = fieldEphem.a_ra
    dec[ti] = fieldEphem.a_dec

if args.plot:
    def __angdiff(a, b):
        return ((a-b) + 180) % 360 - 180
    for axl, axd in zip(["Az", "El", "RA", "DEC"],
                        [az, el, ra, dec]):
        hfmt = mdates.DateFormatter('%H:%M')
        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(111)
        ax.set_xlabel("Time UTC")
        ax.set_ylabel("{} [deg]".format(axl))
        ax.plot(timecoorddt, np.rad2deg(axd))
        ax.xaxis.set_major_formatter(hfmt)
        ax.grid(True)
        plt.show()

if args.sim:
    log.warn("Simulating correction only -- no changes applied to data")

coordtimeunix = np.array(list(map(lambda x: x.replace(tzinfo=pytz.UTC).timestamp(), timecoorddt)))
nrowsput = 0
if args.plot:
    plt.figure()
    plt.xlabel("u [m]")
    plt.ylabel("v [m]")

with tbl(args.ms, ack=False, readonly=False) as t:
    with taql("select * from $t where FIELD_ID=={} and ANTENNA1!=ANTENNA2".format(args.field)) as tt:
        nrow = tt.nrows()
        nchunk = nrow // args.chunksize + int(nrow % args.chunksize > 0)
        for ci in range(nchunk):
            cl = ci * args.chunksize
            crow = min(nrow - ci * args.chunksize, args.chunksize)

            def __casa_to_unixtime(t):
                dt = quantity("{}s".format(t)).to_unix_time()
                return dt
            mstimecentroid = tt.getcol("TIME_CENTROID", startrow=cl, nrow=crow)
            msuniqtime = np.unique(mstimecentroid)
            # expensive quanta operation -- do only for unique values
            uniqtimemsunix = np.array(list(map(__casa_to_unixtime, msuniqtime)))
            timemsunixindex = np.array(list(map(lambda t: np.argmin(np.abs(msuniqtime-t)),
                                                mstimecentroid)))
            timemsunix = uniqtimemsunix[timemsunixindex]
            coordmaperr = np.array(list(map(lambda x: np.min(np.abs(x - coordtimeunix)), timemsunix)))
            if args.verbose:
                log.info("Max temporal err on RADEC sampling: {0:.4f}s".format(coordmaperr.max()))
                log.info("Min temporal err on RADEC sampling: {0:.4f}s".format(coordmaperr.min()))
                log.info("Mean temporal err on RADEC sampling: {0:.4f}s".format(np.mean(coordmaperr)))

            coordmap = np.array(list(map(lambda x: np.argmin(np.abs(x - coordtimeunix)), timemsunix)))

            a1 = tt.getcol("ANTENNA1", startrow=cl, nrow=crow)
            a2 = tt.getcol("ANTENNA2", startrow=cl, nrow=crow)
            msuvw = tt.getcol("UVW", startrow=cl, nrow=crow)
            newuvw = np.zeros_like(msuvw)
            adjusted_uvw = 0
            for ms_uniqt in msuniqtime:
                sel = mstimecentroid == ms_uniqt
                assert np.unique(coordmap[sel]).size == 1
                icoord = np.unique(coordmap[sel])[0]

                ira = ra[icoord]
                idec = dec[icoord]
                fringestopctr = np.array([[ira, idec]])
                padded_uvw = fixvis.synthesize_uvw(station_ECEF=apos,
                                                   time=mstimecentroid[sel],
                                                   a1=a1[sel],
                                                   a2=a2[sel],
                                                   phase_ref=fringestopctr)
                newuvw[sel] = fixvis.dense2sparce_uvw(a1=a1[sel],
                                                      a2=a2[sel],
                                                      time=mstimecentroid[sel],
                                                      ddid=np.zeros_like(a1[sel]),
                                                      padded_uvw=padded_uvw["UVW"])
                adjusted_uvw += newuvw[sel].size
            assert adjusted_uvw == msuvw.size
            if args.plot:
                #seldb = np.logical_and(a1 == 0, a2 == 32)
                seldb = np.ones_like(a1, dtype=bool)
                #plt.title("uvplot for baseline {}&{}".format(anames[0], anames[32]))

                plt.scatter(msuvw[seldb][:,0],
                            msuvw[seldb][:,1], label="katpoint" if ci==0 else "_nolegend_", facecolor="b", marker="x")
                plt.scatter(newuvw[seldb][:,0],
                            newuvw[seldb][:,1], label="casacore" if ci==0 else "_nolegend_", facecolor="r", alpha=0.3)
            if not args.sim:
                tt.putcol("UVW", newuvw, startrow=cl, nrow=crow)
            log.info("\t{} chunk {}/{}".format("Corrected" if not args.sim else "Simulated", ci+1, nchunk))

            nrowsput += crow
    assert nrow == nrowsput
    if args.plot:
        plt.legend()

        plt.title("uvplot for field '{}'".format(fieldnames[args.field]))
        plt.show()


