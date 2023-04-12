#!/usr/bin/env python3
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

def create_logger():
    """ Create a console logger """
    log = logging.getLogger("Scan Info")
    cfmt = logging.Formatter(('%(name)s - %(asctime)s %(levelname)s - %(message)s'))
    log.setLevel(logging.DEBUG)
    log.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(cfmt)

    log.addHandler(console)

    return log, console, cfmt

log, log_console_handler, log_formatter = create_logger()

parser = argparse.ArgumentParser(description="Scan Info")
parser.add_argument("ms", type=str, help="Database to correct")
parser.add_argument("--field", "-f", dest="field", type=int, default=0, help="Field index to correct")
parser.add_argument("--doPlot", "-dp", dest="plot", action="store_true", help="Make plots for specified field")
parser.add_argument("--parangstep", "-pas", type=float, dest="stepsize", default=1., help="Parallactic angle correction step size in minutes")
args = parser.parse_args()

if args.plot:
    from matplotlib import pyplot as plt
    import matplotlib.dates as mdates
    log.info("Enabling plotting")


with tbl(args.ms, ack=False) as t:
    with taql("select * from $t where FIELD_ID=={}".format(args.field)) as tt:
        def __to_datetimestr(t):
            dt = datetime.datetime.utcfromtimestamp(quantity("{}s".format(t)).to_unix_time())
            return dt.strftime("%Y/%m/%d %H:%M:%S")
        dbtime = tt.getcol("TIME_CENTROID")
        scans = np.unique(tt.getcol("SCAN_NUMBER"))
        start_time_Z = __to_datetimestr(dbtime.min())
        end_time_Z = __to_datetimestr(dbtime.max())
        log.info("Observation spans '{}' and '{}' UTC".format(
                 start_time_Z, end_time_Z))

dm = measures()
with tbl(args.ms+"::ANTENNA", ack=False) as t:
    anames = t.getcol("NAME")
    apos = t.getcol("POSITION")
    aposdm = list(map(lambda pos: dm.position('itrf',*[ quantity(x,'m') for x in pos ]),
                      apos))
    aposctr = dm.position('itrf', *[quantity(x, 'm') for x in np.mean(apos, axis=0)])
    dm.doframe(aposctr)
    aposctr_wgs84 = dm.measure(aposctr, 'wgs84')

arrelev = aposctr_wgs84['m2']['value']
arrlat = aposctr_wgs84['m1']['value']
arrlon = aposctr_wgs84['m0']['value']

facility = ephem.Observer()
facility.lat = arrlat
facility.long = arrlon
facility.elevation = arrelev
facility.epoch = ephem.J2000

facility.date = start_time_Z
st = facility.date
facility.date = end_time_Z
et = facility.date
TO_SEC = 3600*24.0
nstep = int(np.round((float(et)*TO_SEC - float(st)*TO_SEC) / (args.stepsize*60.)))
timepa = time = np.linspace(st,et,nstep)
timepadt = list(map(lambda x: ephem.Date(x).datetime(), time))
timepaunix = np.array(list(map(lambda x: x.replace(tzinfo=pytz.UTC).timestamp(), timepadt)))

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
arraypa = az.copy()
ha = az.copy()
pa = np.zeros((len(anames), nstep), np.float32)

zenith = dm.direction('AZELGEO','0deg','90deg')
for ti, t in enumerate(time):
    facility.date = t
    t_iso8601 = facility.date.datetime().strftime("%Y-%m-%dT%H:%M:%S.%f")
    fieldEphem.compute(facility)
    az[ti] = fieldEphem.az
    el[ti] = fieldEphem.alt
    ra[ti] = fieldEphem.a_ra
    dec[ti] = fieldEphem.a_dec
    ha[ti] = facility.sidereal_time() - fieldEphem.g_ra
    arraypa[ti] = fieldEphem.parallactic_angle()
    # compute PA per antenna
    field_centre = dm.direction('J2000', quantity(ra[ti],"rad"), quantity(dec[ti],"rad"))
    dm.do_frame(dm.epoch("UTC", quantity(t_iso8601)))
    #dm.doframe(aposdm[0])
    #field_centre = dm.measure(dm.direction('moon'), "J2000")
    #racc[ti] = field_centre["m0"]["value"]
    #deccc[ti] = field_centre["m1"]["value"]
    for ai in range(len(anames)):
       dm.doframe(aposdm[ai])
       pa[ai, ti] = dm.posangle(field_centre,zenith).get_value("rad")
if args.plot:
    def __angdiff(a, b):
        return ((a-b) + 180) % 360 - 180
    for axl, axd in zip(["Az", "El", "RA", "DEC", "ParAng"],
                        [az, el, ra, dec, pa]):
        hfmt = mdates.DateFormatter('%H:%M')
        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(111)
        ax.set_xlabel("Time UTC")
        ax.set_ylabel("{} [deg]".format(axl))
        if axl == "ParAng":
            ax.errorbar(timepadt,
                        np.rad2deg(np.mean(axd, axis=0)),
                        capsize=2,
                        yerr=0.5*__angdiff(np.rad2deg(axd.max(axis=0)),
                                           np.rad2deg(axd.min(axis=0))), label="CASACORE")
            plt.plot(timepadt, np.rad2deg(arraypa), label="PyEphem")
        else:
            ax.plot(timepadt, np.rad2deg(axd))
        ax.xaxis.set_major_formatter(hfmt)
        ax.grid(True)
        plt.show()

    with tbl(args.ms+"::FEED", ack=False) as t:
        receptor_aid = t.getcol("ANTENNA_ID")
        if len(receptor_aid) != len(anames):
            raise RuntimeError("Receptor angles not all filed for the antennas in the ::FEED keyword table")
        receptor_angles = dict(zip(receptor_aid, t.getcol("RECEPTOR_ANGLE")[:,0]))
        if args.fa is not None:
            receptor_angles[...] = float(args.fa)
            log.info("Overriding F Jones angle to {0:.3f} for all antennae".format(float(args.fa)))
        else:
            log.info("Applying the following feed angle offsets to parallactic angles:")
            for ai, an in enumerate(anames):
                log.info("\t {0:s}: {1:.3f} degrees".format(an, np.rad2deg(receptor_angles.get(ai, 0.0))))

    raarr = np.empty(len(anames), dtype=int)
    for aid in range(len(anames)):
        raarr[aid] = receptor_angles[aid]

with tbl(args.ms, ack=False, readonly=False) as t:
    for s in scans:
        with taql("select * from $t where FIELD_ID=={} and SCAN_NUMBER=={}".format(args.field, s)) as tt:
           def __casa_to_unixtime(t):
               dt = quantity("{}s".format(t)).to_unix_time()
               return dt
           mstimecentroid = tt.getcol("TIME_CENTROID")
           msuniqtime = np.unique(mstimecentroid)
           # expensive quanta operation -- do only for unique values
           uniqtimemsunix = np.array(list(map(__casa_to_unixtime, msuniqtime)))
           timemsunixindex = np.array(list(map(lambda t: np.argmin(np.abs(msuniqtime-t)),
                                               mstimecentroid)))
           timemsunix = uniqtimemsunix[timemsunixindex]

           pamap = np.array(list(map(lambda x: np.argmin(np.abs(x - timepaunix)), timemsunix)))[len(timemsunix)//2]
           log.info("SCAN {}".format(s))
           log.info("\tElev {:.0f} deg".format(np.rad2deg(el[pamap])))
           log.info("\tZenith {:.0f} deg".format(90. - np.rad2deg(el[pamap])))
           log.info("\tAz {:.0f} deg".format(np.rad2deg(az[pamap])))
           log.info("\tParAng {:.0f} deg".format(np.rad2deg(arraypa[pamap])))
           log.info("\tHA {:.0f} deg".format(np.rad2deg(ha[pamap])))
           log.info("\tMJD {:.3f}".format(timemsunix[len(timemsunix)//2] / (3600.*24.)))

