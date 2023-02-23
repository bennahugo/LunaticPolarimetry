from pyrap.tables import table as tbl
from pyrap.tables import taql
import numpy as np
from matplotlib import pyplot as plt
import sys
import logging
import functools
from utils.fixvis import baseline_index
import numpy.ma as ma
import argparse

def create_logger():
    """ Create a console logger """
    log = logging.getLogger("SEFD computer")
    cfmt = logging.Formatter(('%(name)s - %(asctime)s %(levelname)s - %(message)s'))
    log.setLevel(logging.DEBUG)
    log.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(cfmt)

    log.addHandler(console)

    return log, console, cfmt

log, log_console_handler, log_formatter = create_logger()

def __padbuffer(buffer, expbuffer, rowsel, exposure, data, time, blindx):
    datarsel = data[rowsel, :, :]
    exprsel = exposure[rowsel]
    for ti, tt in enumerate(np.unique(time[rowsel])):
        tsel = time[rowsel] == tt
        for bl in range(NBL):
            sel = np.logical_and(tsel,
                                 blindx[rowsel] == bl)
            if np.sum(sel) == 0: continue
            buffer[ti, bl, :, :] = datarsel[sel, :, :]
            expbuffer[ti, bl] = exprsel[sel]

parser = argparse.ArgumentParser(description="SEFD computer -- program to compute SEFD from visibility noise")
parser.add_argument("--ddid", dest="DDID", default=0, type=int, help="Select DDID - usually just equal to SPW")
parser.add_argument("--scan", dest="SCAN", default="*", type=str, help="Select by SCAN NUMBER")
parser.add_argument("--fieldid", "-fi", dest="FIELD", default=0, type=int, help="FIELD id to compute SEFD for")
parser.add_argument("--chunksize", dest="CHUNKSIZE", default=10000, type=int, help="Chunk size")
parser.add_argument("--datacolumn", dest="COLUMN", default="CORRECTED_DATA", type=str, help="Data column to read (should be flux calibrated)")
parser.add_argument("--modeldatacolumn", dest="SUBCOLUMN", default="MODEL_DATA", type=str, help="Data column to read (should be flux calibrated)")
parser.add_argument("--disablesubevenodd", dest="DOSUBEVENODD", action="store_false", help="By default even and odd timestamps (decorreclated) are subtracted to further remove sky flux contribution, this disables this behaviour")
parser.add_argument("--vmin", dest="VMIN", default=0, type=float, help="Plot vmin (default 0 Jy)")
parser.add_argument("--vmax", dest="VMAX", default=900, type=float, help="Plot vmin (default 900 Jy)")

parser.add_argument("ms", type=str, help="Database")

args = parser.parse_args()

ONLYCHUNK = None
VMIN = args.VMIN
VMAX = args.VMAX
FIELD = args.FIELD
DDID = args.DDID
COLUMN = args.COLUMN
SUBCOLUMN = None if args.SUBCOLUMN.strip() == "" else args.SUBCOLUMN.strip()
CHUNKSIZE = args.CHUNKSIZE
DOSUBEVENODD = args.DOSUBEVENODD
SCANSEL = args.SCAN
log.info(f"Will select scans matching '{SCANSEL}'")
if DOSUBEVENODD:
    log.info("Will estimate noise by subtracting even and odd timestamps")
if SUBCOLUMN:
    log.info(f"Subtracting column {SUBCOLUMN} from {COLUMN}")
else:
    log.info(f"Will read column {COLUMN}")
ms = args.ms

with tbl(ms+"::FIELD", ack=False) as t:
    if FIELD < 0 or FIELD >= t.nrows():
        raise RuntimeError("Invalid field selected")
    fnames = t.getcol("NAME")
    log.info(f"Will compute for field '{fnames[FIELD]}'")

with tbl(ms+"::ANTENNA", ack=False) as t:
    anames = t.getcol("NAME")
    NANT = len(anames)
    NBL = NANT * (NANT - 1) // 2 + NANT
    log.info(f"{NANT} antennae in this database")

with tbl(ms+"::DATA_DESCRIPTION", ack=False) as t:
    if DDID < 0 or DDID >= t.nrows():
        raise RuntimeError("Invalid DDID selected")
    spwsel = t.getcol("SPECTRAL_WINDOW_ID")[DDID]

with tbl(ms+"::SPECTRAL_WINDOW", ack=False) as t:
    chan_freqs = t.getcol("CHAN_FREQ")[spwsel]
    chan_width = t.getcol("CHAN_WIDTH")[spwsel]
    NCHAN = chan_freqs.size
    log.info("Will compute for SPW {0:d} ({3:d} channels): {1:.2f} to {2:.2f} MHz".format(
        spwsel, chan_freqs.min()*1e-6, chan_freqs.max()*1e-6, NCHAN))

noisesum = np.zeros((NCHAN, 4), dtype=np.float64)
M = np.zeros((NCHAN, 4), dtype=np.int64)

with tbl(ms, ack=False, readonly=True) as t:
    scanseltaql = f"AND SCAN_NUMBER=={SCANSEL}" if SCANSEL != "*" else ""
    with taql("select * from $t where FIELD_ID=={} {}".format(FIELD, scanseltaql)) as tt:
        nrow = tt.nrows()
        nchunk = nrow // CHUNKSIZE + int(nrow % CHUNKSIZE > 0)
        for ci in range(nchunk) if not ONLYCHUNK else ONLYCHUNK:
            log.info(f"Processing chunk {ci+1}/{nchunk}")
            cl = ci * CHUNKSIZE
            crow = min(nrow - ci * CHUNKSIZE, CHUNKSIZE)
            data = tt.getcol(COLUMN, startrow=cl, nrow=crow)
            time = tt.getcol("TIME", startrow=cl, nrow=crow)
            flag = tt.getcol("FLAG", startrow=cl, nrow=crow)
            a1 = tt.getcol("ANTENNA1", startrow=cl, nrow=crow)
            a2 = tt.getcol("ANTENNA2", startrow=cl, nrow=crow)
            exposure = tt.getcol("EXPOSURE", startrow=cl, nrow=crow)

            # flag also autos
            autos = a1 == a2
            flag[autos] = True

            data[flag] = np.nan
            if SUBCOLUMN:
                model = tt.getcol(SUBCOLUMN, startrow=cl, nrow=crow)
                model[flag] = np.nan
                data = data - model
            # visibilities are stored without a factor of 2 (see SMIRNOV I)
            # this means HH and VV both store the full observed power of
            # an unpolarized source. To get physical noise estimates
            # we need to remove this software convention
            data /= 2.0
            uniqa = np.unique(np.concatenate([a1, a2]))
            blindx = baseline_index(a1, a2, np.max(uniqa) + 1)

            if data.shape[2] != 4:
                raise RuntimeError("Data must be full correlation")
            uniqt = np.unique(time)
            if uniqt.size < 2: continue # need odd and even timestamps
            event = functools.reduce(lambda a, b: np.logical_or(a,b),
                                     map(lambda ut: time == ut,
                                         uniqt[0::2]))
            oddt = functools.reduce(lambda a, b: np.logical_or(a,b),
                                    map(lambda ut: time == ut,
                                        uniqt[1::2]))
            # truncate to same size of elements for even and odd times
            sel = np.array(list(zip(event, oddt)))
            sele = sel[:, 0]
            selo = sel[:, 1]
            if DOSUBEVENODD:
                NTIME = np.unique(time[sele]).size
                # rebuffer vis onto ntime x nbl x nchan x 4 padded buffers
                evendata = np.zeros((NTIME, NBL, NCHAN, 4), dtype=data.dtype) * np.nan
                odddata = np.zeros_like(evendata) * np.nan
                evenexp = np.zeros((NTIME, NBL), dtype=exposure.dtype)
                oddexp = np.zeros_like(evenexp)

                __padbuffer(evendata.view(), evenexp.view(), sele, exposure, data, time, blindx)
                __padbuffer(odddata.view(), oddexp.view(), selo, exposure, data, time, blindx)
                noise = (evendata - odddata).real # can either do it for real or imag correlator
                tblexp = np.repeat(np.max(np.array([evenexp, oddexp]), axis=0), (NCHAN) * 4).reshape(NTIME, NBL, NCHAN, 4)
                Mi = np.sum(np.logical_and(np.logical_not(np.isnan(evendata)),
                                           np.logical_not(np.isnan(odddata))),
                            axis=(0, 1), dtype=np.int64)
            else:
                NTIME = np.unique(time).size
                databuff = np.zeros((NTIME, NBL, NCHAN, 4), dtype=data.dtype) * np.nan
                expbuff = np.zeros((NTIME, NBL), dtype=exposure.dtype) * np.nan
                rowsel = np.ones_like(time, dtype=np.bool)
                __padbuffer(databuff.view(), expbuff.view(), rowsel, exposure, data, time, blindx)
                noise = databuff.real # can either do it for real or imag correlator
                tblexp = np.repeat(expbuff, (NCHAN) * 4).reshape(NTIME, NBL, NCHAN, 4)
                Mi = np.sum(np.logical_not(np.isnan(databuff)),
                            axis=(0, 1), dtype=np.int64)
            M[:,:] += Mi


            tblvar = noise**2
            tblchwidth = np.tile(chan_width.repeat(4), (NTIME * NBL, 1)).reshape(NTIME, NBL, NCHAN, 4)
            blvarsum = np.nansum(2 * # two ants in the baseline
                                 tblvar *
                                 tblexp *
                                 tblchwidth,
                                 axis=(0,1), dtype=np.float64)
            noisesum = np.nansum(np.array([noisesum, blvarsum]), axis=0)


SEFD = np.sqrt(noisesum / M)
SEFD[M == 0] = np.nan
plt.figure()
plt.plot(chan_freqs*1e-6, SEFD[:, 0], label="vertical", linestyle="--")
plt.plot(chan_freqs*1e-6, SEFD[:, 3], label="horizontal", linestyle="--")
plt.plot(chan_freqs*1e-6, np.sqrt(0.5 * (SEFD[:, 0]**2 + SEFD[:, 3]**2)), label="Stokes I", linewidth=1, color="k")
plt.legend()
plt.xlabel("Frequency [MHz]")
plt.ylabel("SEFD [Jy]")
plt.ylim(VMIN,VMAX)
plt.grid(True)
plt.show()

with open("sefd.txt", "w+") as f:
    f.write("FREQ[MHz]\tVertical SEFD[JY]\tHorizontal SEFD[JY]\tStokes I SEFD[JY]\n")
    for nu, sefdvert, sefdhorz, sefdi in zip(chan_freqs*1e-6,SEFD[:, 0],SEFD[:, 3],np.sqrt(0.5 * (SEFD[:, 0]**2 + SEFD[:, 3]**2))):
        f.write(f'{nu}\t{sefdvert}\t{sefdhorz}\t{sefdi}\n')

