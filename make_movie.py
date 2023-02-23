#!/usr/bin/env python
# ian.heywood@physics.ox.ac.uk
# adapted by bhugo@sarao.ac.za

import glob
import logging
import os
import random
import numpy
import string

from astropy.io import fits
from astropy.time import Time
from multiprocessing import Pool
from PIL import Image,ImageDraw,ImageFont

fontPath = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
sans15  =  ImageFont.truetype ( fontPath, 32 )
sans24  =  ImageFont.truetype ( fontPath, 42 )


def generate_temp(k=16):
    tmpfits = 'temp_'+''.join(random.choices(string.ascii_uppercase + string.digits, k=k))+'.fits'
    return tmpfits


def make_png(ff,i):

    tmpfits = generate_temp()

    logging.info(' | File '+str(i)+' | Input image '+ff)
    logging.info(' | File '+str(i)+' | Temp image  '+tmpfits)

    os.system('mShrink '+ff+' '+tmpfits+' 2')

    input_hdu = fits.open(ff)[0]
    hdr = input_hdu.header
    map_date = hdr.get('DATE-OBS')
    ctrfreq = hdr["CRVAL3"]
    bmaj = hdr["BMAJ"]
    bmin = hdr["BMIN"]
    t_mjd = Time(map_date, format='isot', scale='ut1').mjd
    tt = map_date+' | '+str(t_mjd)
#   pp = str(i).zfill(4)+'_'+ff.replace('.fits','.png')
    pp = 'pic_'+str(i).zfill(4)+'.png'
#   syscall = 'mViewer -ct 0 -gray '+ff+' -0.0004 0.0008 -out '+pp
    logging.info(' | File '+str(i)+' | PNG         '+pp)
    syscall = 'mViewer -ct 0 -gray '+tmpfits+' -0.02552 0.1316 -out '+pp
    os.system(syscall)
    logging.info(' | File '+str(i)+' | Time        '+tt)
    img = Image.open(pp)
    # crop to centre
    CROPSIZE = 1750
    img = img.crop((img.size[0]//2 - CROPSIZE//2,  img.size[1]//2 - CROPSIZE//2, img.size[0]//2 + CROPSIZE//2, img.size[1]//2 + CROPSIZE//2))
    # ffmpeg codec does not like odd numbers much like certain friends... trim the last row or column if needed
    img = img.crop((0,0,img.size[0] - (img.size[0] % 2), img.size[1] - (img.size[1] % 2)))
    xx,yy = img.size
    draw = ImageDraw.Draw(img)
    TITLE = f"MeerKAT UHF Band [{ctrfreq*1e-6:.1f} MHz]"
    draw.text((0.03*xx,0.03*yy),'Karoo skies - Moon',fill=('white'),font=sans24)
    draw.text((0.03*xx,0.81*yy),TITLE,fill=('white'),font=sans15)
    draw.text((0.03*xx,0.84*yy),'Frame : '+str(i).zfill(len(str(nframes)))+' / '+str(nframes),fill=('white'),font=sans15)
    draw.text((0.03*xx,0.87*yy),'Time  : '+tt,fill=('white'),font=sans15)
    draw.text((0.03*xx,0.90*yy),'Image : '+ff,fill=('white'),font=sans15)
    draw.text((0.03*xx,0.93*yy),f'Resolution : {bmaj*3600.:.1f}" x {bmin*3600.:.1f}"',fill=('white'),font=sans15)
    draw.text((0.03*xx,0.96*yy),f'Stokes : I',fill=('white'),font=sans15)
    img.save(pp)
    os.system('rm '+tmpfits)
    logging.info(' | File '+str(i)+' | Done')

if __name__ == '__main__':

    logfile = 'make_movie.log'
    logging.basicConfig(filename=logfile, level=logging.DEBUG, format='%(asctime)s |  %(message)s', datefmt='%d/%m/%Y %H:%M:%S ')


    fitslist = sorted(glob.glob('*-t*-image.fits'))
    ids = numpy.arange(0,len(fitslist))
    nframes = len(fitslist)
    j = 8

    pool = Pool(processes=j)
#    pool.map(make_png,fitslist)
    pool.starmap(make_png,zip(fitslist,ids))

    frame = '2340x2340'
    fps = 10
    opmovie = fitslist[0].split('-t')[0]+'.mp4'
    os.system('ffmpeg -r '+str(fps)+' -f image2 -s '+frame+' -i pic_%0004d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p '+opmovie)
