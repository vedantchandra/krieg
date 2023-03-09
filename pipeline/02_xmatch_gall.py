# xmatches supplied clean table to all-band photometry on Cannon

import numpy as np
from astropy.io import ascii
from astropy.table import Table
import glob
import astropy
import os
from astropy import units as u
from astropy.coordinates import SkyCoord
import sys

print('starting script')
with open('/n/home03/vchandra/outerhalo/09_sdss5/pipeline/control/redux.txt', 'r') as file:
	redux = file.read().replace('\n','')
#infile = '/n/holyscratch01/conroy_lab/vchandra/mage/MS/ms_input/source_ids.txt'

infile = '/n/holyscratch01/conroy_lab/vchandra/sdss5/catalogs/spAll_halo_%s.fits' % redux
outfolder = '/n/holyscratch01/conroy_lab/vchandra/sdss5/catalogs/xgall/'

try:
	os.mkdir(outfolder)
except:
	print('folder exists!')

# XMATCH Charlie's Gaia EDR3 photometric tables to my Gaia giants sample
# Loop over Charlie's files, match my file, save as h5

charlie_cats = '/n/holystore01/LABS/conroy_lab/Lab/gaia/edr3/gall2/catalogs/'
charlie_cats = glob.glob(charlie_cats + '*')

galidx = int(sys.argv[1])

bigcat = charlie_cats[galidx]

table = Table.read(infile)

outtab = Table();

max_sep = 3.0 * u.arcsec
c = SkyCoord(ra=table['RACAT'] * u.degree, dec = table['DECCAT']*u.degree)

# for bigcat in charlie_cats:
print('-----------')
print('Big Catalog: %s' % bigcat.split('/')[-1])

bs = bigcat.split('/')[-1].split('b')[-1].split('_')
b1 = float(bs[0])
b2 = float(bs[1][:5])
bs = np.arange(b1, b2 + 1).astype(int)

print(b1, b2)

bigtable = Table.read(bigcat)

catalog = SkyCoord(ra=bigtable['RA'] * u.degree, dec = bigtable['DEC'] * u.degree)
idx, d2d, d3d = c.match_to_catalog_sky(catalog)
sep_constraint = d2d < max_sep
c_matches = table[sep_constraint]
cat_matches = bigtable[idx[sep_constraint]]
tab = astropy.table.hstack((c_matches, cat_matches))

outtab = astropy.table.vstack((outtab, tab))

del bigtable
del tab

print('output table has %i rows' % len(outtab))
print('writing output!')
outtab.write(outfolder + 'mwmhalo_xgall_%s_%i.fits' % (redux, galidx), overwrite = True)
