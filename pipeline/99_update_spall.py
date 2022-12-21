from astropy.table import Table
import urllib.request
import time
import sys
import os
import numpy as np

with open('/n/home03/vchandra/outerhalo/09_sdss5/pipeline/redux.txt', 'r') as file:
	redux = file.read().replace('\n','')

print(redux)

auth_user = 'sdss5'
auth_passwd = 'panoptic-5'

inpath = 'https://data.sdss5.org/sas/sdsswork/bhm/boss/spectro/redux/%s/spAll-lite-%s.fits.gz' % (redux, redux)
outpath = '/n/holyscratch01/conroy_lab/vchandra/sdss5/catalogs/spAll-lite-%s.fits.gz' % redux

print('executing request...')
passman = urllib.request.HTTPPasswordMgrWithDefaultRealm()
passman.add_password(None, inpath, auth_user, auth_passwd)
authhandler = urllib.request.HTTPBasicAuthHandler(passman)
opener = urllib.request.build_opener(authhandler)
urllib.request.install_opener(opener)
urllib.request.urlretrieve(inpath, outpath)
print('done!')