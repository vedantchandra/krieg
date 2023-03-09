import numpy as np
from astropy.io import ascii
from astropy.table import Table
import glob
import astropy
import os
import argparse

with open('/n/home03/vchandra/outerhalo/09_sdss5/pipeline/control/redux.txt', 'r') as file:
    redux = file.read().replace('\n','')


selcol = ['PROGRAMNAME', 'FIELDQUALITY', 'GAIA_G', 'FIRSTCARTON', 'RACAT', 'DECCAT', 
         'CATALOGID', 'FIELD', 'NEXP', 'EXPTIME', 'AIRMASS', 'HEALPIX', 'MJD_FINAL', 
         'SPEC_FILE', 'MJD', 'FIBER_RA', 'FIBER_DEC', 'SN_MEDIAN_ALL', 'SPECOBJID',
         'MOON_DIST', 'MOON_PHASE', "CARTON_TO_TARGET_PK", 'carton']


parser = argparse.ArgumentParser()
parser.add_argument('--getspall',help='catalog to use as input',type=int,default=0)
args = parser.parse_args()

getspall = bool(int(args.getspall))

#####################################
###### DOWNLOAD SPALL-LITE ###############
#####################################

outspall = '/n/holyscratch01/conroy_lab/vchandra/sdss5/catalogs/spAll-lite-%s.fits.gz' % redux

if getspall:

    try:
        os.remove(outspall)
        print('spall already existed, deleting and re-downloading')
    except OSError:
        pass


    cmd = 'wget --user sdss5 --password panoPtic-5 --no-check-certificate https://data.sdss5.org/sas/ipl-2/spectro/boss/redux/%s/spAll-lite-%s.fits.gz -O %s' % (redux, redux, outspall)
    os.system(cmd)

else:
    print('using existing spall file: %s' % outspall)

#####################################
###### REDUCE TO HALO SPALL ###############
#####################################


spall = Table.read(outspall)

spall['carton'] = [str(spall['FIRSTCARTON'][ii]).strip() for ii in range(len(spall))]

with open('/n/home03/vchandra/outerhalo/09_sdss5/pipeline/control/halocartons.txt', 'r') as file:
    halocartons = file.read().splitlines()

with open('/n/home03/vchandra/outerhalo/09_sdss5/pipeline/control/ncarton.txt', 'w') as f:
    f.write('%i' % len(halocartons))

print('there are %i cartons' % len(halocartons))

selected = np.repeat(False, len(spall))
for cart in halocartons:
    incarton = (spall['carton'] == cart)
    print('%s has %i stars' % (cart, np.sum(incarton)))

    selected |= incarton

print('there are %i halo stars in total' % np.sum(selected))

halo = spall[selected][selcol]

print('writing table...')

print('spAll-halo has %i rows' % len(halo))

halo.write('/n/holyscratch01/conroy_lab/vchandra/sdss5/catalogs/spAll_halo_%s.fits' % redux, overwrite = True)

print('done!')