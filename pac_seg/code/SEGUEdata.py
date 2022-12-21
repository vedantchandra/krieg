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
     datadir = '/n/holystore01/LABS/conroy_lab/Lab/SEGUE/data'
else:
     datadir = '/Users/pcargile/Astro/SEGUE/data/'

class SegueData(object):
     """docstring for SegueData"""
     def __init__(self, survey=None, tileID=None, mjd=None):
          super(SegueData, self).__init__()

          print('... looking for {0} {1} {2} in {3}'.format(survey,tileID,mjd,datadir))

          # read in tile
          if mjd == None:
               segue_tilefile  = glob.glob('{0}/{1}/{2}/spPlate*.fits'.format(datadir,survey,tileID))[0]
          else:
               segue_tilefile  = glob.glob('{0}/{1}/{2}/spPlate*{3}*.fits'.format(datadir,survey,tileID,mjd))[0]               
          segue_acatfile  = '{0}/{1}/{2}/acat.fits'.format(datadir,survey,tileID)

          self.segue_tile  = fits.open(segue_tilefile)
          self.segue_st    = Table.read(segue_tilefile,format='fits')
          self.segue_acat  = Table.read(segue_acatfile,format='fits')

          self.header = self.segue_tile[0].header
          
          self.nobj = len(self.segue_acat)
          
     def getdata(self,index=None,GaiaID=None,FiberID=None):

          # pull the acat data
          if index is not None:
               phot = self.segue_acat[index]
          if GaiaID is not None:
               phot = self.segue_acat[self.segue_acat['GAIAEDR3_ID'] == GaiaID]
          if FiberID is not None:
               phot = self.segue_acat[self.segue_acat['FIBERID'] == FiberID]
          try:
               assert phot
          except AssertionError:
               print('Warning: User must pass selection criteria for star')
               raise

          phot = np.array(phot).squeeze()

          # convert SDSS phot from nanomags
          phot['SDSS_U'] = phot['SDSS_U'] + 22.5
          phot['SDSS_G'] = phot['SDSS_G'] + 22.5
          phot['SDSS_R'] = phot['SDSS_R'] + 22.5
          phot['SDSS_I'] = phot['SDSS_I'] + 22.5
          phot['SDSS_Z'] = phot['SDSS_Z'] + 22.5

          # find which spectrum is associated with this acat row
          spec_ind = np.argwhere(
               (self.segue_st['RA'] == phot['PLUG_RA']) & (self.segue_st['DEC'] == phot['PLUG_DEC']))[0][0]

          # pull individual spectrum parts
          flux    = self.segue_tile[0].data[spec_ind]
          ivar    = self.segue_tile[1].data[spec_ind]
          andmask = self.segue_tile[2].data[spec_ind]
          ormask  = self.segue_tile[3].data[spec_ind]

          # LSF dispersion in km/s (delta(log(lambda)) = 1e-4 for SEGUE)
          lsf = (np.log(10) * self.header['CD1_1'] * self.segue_tile[4].data[spec_ind]) * speedoflight 

          # calculate wave array
          wave = [10.0**self.header['CRVAL1']]
          for ii in range(len(flux)):
               if ii == 0:
                    continue
               wave.append(10.0**(self.header['CD1_1'] + np.log10(wave[ii-1])))
          wave = np.array(wave)

          return {'phot':np.array(phot),'spec':[wave,flux,ivar,andmask,ormask,lsf]}
