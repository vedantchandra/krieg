import numpy as np
from astropy.table import Table,hstack,vstack,join
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import glob

from scipy import constants
speedoflight = constants.c / 1000.0

import socket
hostname = socket.gethostname()
if hostname[:4] == 'holy':
     datadir = '/n/holystore01/LABS/conroy_lab/Lab/SEGUE/'
else:
     datadir = '/Users/pcargile/Astro/SEGUE/data/'


def getdata(survey=None,tileID=None,index=0):
     # read in tile
     segue_tilefile  = glob.glob('{0}/{1}/{2}/spPlate*.fits'.format(datadir,survey,tileID))[0]
     segue_acatfile  = '{0}/{1}/{2}/acat.fits'.format(datadir,survey,tileID)

     segue_tile  = fits.open(segue_tilefile)
     segue_acat  = Table.read(segue_acatfile,format='fits')

     header = segue_tile[0].header

     # pull acat data
     phot = segue_acat[segue_acat['SEGUE_INDEX'] == index]

     if len(phot) == 0:
          phot = 'SKY'

     # pull individual spectrum parts
     flux = segue_tile[0].data[index]
     ivar = segue_tile[1].data[index]
     andmask = segue_tile[2].data[index]
     ormask  = segue_tile[3].data[index]

     # LSF dispersion in km/s (delta(log(lambda)) = 1e-4 for SEGUE)
     lsf = (np.log(10) * header['CD1_1'] * segue_tile[4].data[index]) * speedoflight 

     # calculate wave array
     wave = [10.0**header['CRVAL1']]
     for ii in range(len(flux)):
          if ii == 0:
               continue
          wave.append(10.0**(header['CD1_1'] + np.log10(wave[ii-1])))
     wave = np.array(wave)

     return {'phot':phot,'spec':[wave,flux,ivar,andmask,ormask,lsf]}