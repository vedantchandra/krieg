from astropy.table import Table
import astropy
import glob
import os
from tqdm import tqdm
import numpy as np
import glob



with open('/n/home03/vchandra/outerhalo/09_sdss5/pipeline/control/redux.txt', 'r') as file:
	redux = file.read().replace('\n','')

infiles = glob.glob('/n/holyscratch01/conroy_lab/vchandra/sdss5/catalogs/xgall/*%s*.fits' % redux)

acat = Table()

for file in infiles:
	tab = Table.read(file)
	acat = astropy.table.vstack((acat, tab))
	print(len(acat))


# Reconcile missing spectra

print('checking how many ACAT stars have spectra...')
datadir = '/n/holyscratch01/conroy_lab/vchandra/sdss5/'
specfiles = glob.glob(datadir + 'spectra/%s/*/*.fits' % redux)
acatfiles = [datadir + 'spectra/%s/%s/%s' % (redux, row['carton'], row['SPEC_FILE'].strip()) for row in acat]
has_spec = np.isin(specfiles, acatfiles)

print('%i stars out of %i in ACAT have downloaded spectra...' % (np.sum(has_spec), len(acat)))

# remove bad columns

acat.remove_columns(['UNWISE_FRACFLUX', 'UNWISE_FLAGS', 'UNWISE_INFO_FLAGS'])

acat['ACAT_ID'] = np.arange(len(acat))
print('writing acat...')
acat.write('/n/holyscratch01/conroy_lab/vchandra/sdss5/catalogs/mwmhalo_acat_%s.fits' % redux, overwrite = True)

print('making clean acat..')

clean = (
    (acat['SN_MEDIAN_ALL'] > 10)
)

acat_clean = acat[clean]

print('%i stars in clean acat' % len(acat_clean))
acat_clean.write('/n/holyscratch01/conroy_lab/vchandra/sdss5/catalogs/mwmhalo_clean_acat_%s.fits' % redux, overwrite = True)

print('making xh3 acat...')

rcat = Table.read('/n/holystore01/LABS/conroy_lab/Lab/h3/catalogs/rcat_V4.0.5.latest_MSG.h5')

import astropy

for key in list(rcat.columns):
    rcat.rename_column(key, 'h3_' + key)

rcat['GAIAEDR3_ID'] = rcat['h3_GAIAEDR3_ID']

xh3 = astropy.table.join(acat, rcat, keys = 'GAIAEDR3_ID')

print('there are %i stars in common with H3' % len(xh3))

xh3.write(datadir + 'catalogs/mwmhalo_xh3_acat_%s.fits' % redux, overwrite = True)
