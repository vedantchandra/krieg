import numpy as np
from astropy.io import ascii
from astropy.table import Table
import glob
import astropy
import os

with open('/n/home03/vchandra/outerhalo/09_sdss5/pipeline/control/redux.txt', 'r') as file:
    redux = file.read().replace('\n','')


selcol = ['PROGRAMNAME', 'FIELDQUALITY', 'GAIA_G', 'FIRSTCARTON', 'RACAT', 'DECCAT', 
         'CATALOGID', 'FIELD', 'NEXP', 'EXPTIME', 'AIRMASS', 'HEALPIX', 'MJD_FINAL', 
         'SPEC_FILE', 'MJD', 'FIBER_RA', 'FIBER_DEC', 'SN_MEDIAN_ALL', 'SPECOBJID',
         'MOON_DIST', 'MOON_PHASE', "CARTON_TO_TARGET_PK", 'carton']


spall = Table.read('/n/holyscratch01/conroy_lab/vchandra/sdss5/catalogs/spAll-lite-%s.fits.gz' % redux)

spall['carton'] = [str(spall['FIRSTCARTON'][ii]).strip() for ii in range(len(spall))]

with open('/n/home03/vchandra/outerhalo/09_sdss5/pipeline/control/halocartons.txt', 'r') as file:
    halocartons = file.read().splitlines()

with open('/n/home03/vchandra/outerhalo/09_sdss5/pipeline/control/ncarton.txt', 'w') as f:
    f.write('%i' % len(halocartons))

print('there are %i cartons' % len(halocartons))

selected = np.repeat(False, len(spall))
for cart in halocartons:
    selected |= (spall['carton'] == cart)

print('there are %i halo stars in total' % np.sum(selected))

halo = spall[selected][selcol]

print('writing table...')

print('spAll-halo has %i rows' % len(halo))

halo.write('/n/holyscratch01/conroy_lab/vchandra/sdss5/catalogs/spAll_halo.fits', overwrite = True)

print('done!')