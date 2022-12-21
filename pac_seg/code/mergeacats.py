from astropy.table import Table,vstack
import glob,sys
import numpy as np

import socket
hostname = socket.gethostname()
if hostname[:4] == 'holy':
     datadir = '/n/holystore01/LABS/conroy_lab/Lab/SEGUE/data/'
else:
     datadir = '/Users/pcargile/Astro/SEGUE/data/'

h3rcat = Table.read('/n/holystore01/LABS/conroy_lab/Lab/h3/catalogs/rcat_V4.0.4.d20220112_MSG.fits')

def run(survey=None):

    flist = glob.glob('{0}/{1}/*/acat.fits'.format(datadir,survey))

    print('Found {} tiles'.format(len(flist)))

    tablelist = []
    for ii,ff in enumerate(flist):
        print('... Reading in: {0} ({1}/{2})'.format(ff,ii+1,len(flist)))
        t = Table.read(ff,format='fits')
        if len(t) < 1:
            continue

        t['H3_ID'] = -1 * np.ones(len(t),dtype=int)
        for ii in range(len(t)):
            if t['GAIAEDR3_ID'][ii] in h3rcat['GAIAEDR3_ID']:
                ind = np.argwhere(h3rcat['GAIAEDR3_ID'] == t['GAIAEDR3_ID'][ii])
                t['H3_ID'][ii] = h3rcat['H3_ID'][ind[0]]

        t['SDSS_U'] = t['SDSS_U'] + 22.5
        t['SDSS_G'] = t['SDSS_G'] + 22.5
        t['SDSS_R'] = t['SDSS_R'] + 22.5
        t['SDSS_I'] = t['SDSS_I'] + 22.5
        t['SDSS_Z'] = t['SDSS_Z'] + 22.5

        tablelist.append(t)

    combtab = vstack(tablelist)
    combtab.write('{0}/../catalogs/{1}_acat.fits'.format(datadir,survey),format='fits',overwrite=True)

if __name__ == '__main__':

    survey = sys.argv[1]
    run(survey=survey)