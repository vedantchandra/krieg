import sys,os,glob,shutil,gzip
from datetime import datetime
import numpy as np
from astropy.table import Table
import h5py
import argparse

from scipy.interpolate import interp1d
from scipy import constants
speedoflight = constants.c / 1000.0
fwhm_sigma = 2.0*np.sqrt(2.0*np.log(2.0))

from h3py.data.read_hecto import PointingData

# from PayneSw.fitting import genmod
# from Payne.fitting.fitutils import airtovacuum

from minesweeper import genmod
from minesweeper.fastMISTmod import GenMIST 

from SEGUEdata import SegueData

import socket,os
hostname = socket.gethostname()
if hostname[:4] == 'holy':
    holypath   = os.environ.get('HOLYSCRATCH')
    specNN = '{}/conroy_lab/pacargile/ThePayne/train/optfal/v256/modV0_spec_LinNet_R5K_WL445_565.h5'.format(holypath)
    contNN = '{}/conroy_lab/pacargile/ThePayne/train/optfal/v256/modV0_cont_LinNet_R12K_WL445_565.h5'.format(holypath)
    NNtype = 'LinNet'
    photNN = '{}/conroy_lab/pacargile/ThePayne/SED/VARRV/'.format(holypath)
    SBlib = '{}/conroy_lab/pacargile/CKC/ckc_R500.h5'.format(holypath)
    MISTgrid = '{}/conroy_lab/pacargile/MIST/MIST_2.0_spot_EEPtrk_small.h5'.format(holypath)
    datadir = '/n/holystore01/LABS/conroy_lab/Lab/SEGUE/data/'
    outdir  = '/n/holyscratch01/conroy_lab/pacargile/SEGUE/'
else:
    specNN = '/Users/pcargile/Astro/ThePayne/train_grid/optfal/v256/modV0_spec_LinNet_R5K_WL445_565.h5'
    contNN = '/Users/pcargile/Astro/ThePayne/train_grid/optfal/v256/modV0_cont_LinNet_R12K_WL445_565.h5'
    NNtype = 'LinNet'
    photNN = '/Users/pcargile/Astro/GITREPOS/ThePayne/data/photANN/'
    SBlib = '/Users/pcargile/Astro/ckc/ckc_R500.h5'
    MISTgrid = '/Users/pcargile/Astro/MIST/MIST_v2.0_spot/MIST_2.0_spot_EEPtrk_small.h5'
    datadir = '/Users/pcargile/Astro/SEGUE/data/'
    outdir  = '/Users/pcargile/Astro/SEGUE/'


CATFILTERARR = ({
     'PS_g':'PS_G',
     'PS_r':'PS_R',
     'PS_i':'PS_I',
     'PS_z':'PS_Z',
     'PS_y':'PS_Y',
     '2MASS_J':'TMASS_J',
     '2MASS_H':'TMASS_H',
     '2MASS_Ks':'TMASS_K',
     'WISE_W1':'UNWISE_W1',
     'WISE_W2':'UNWISE_W2',
     'SDSS_u':'SDSS_U',
     'SDSS_g':'SDSS_G',
     'SDSS_r':'SDSS_R',
     'SDSS_i':'SDSS_I',
     'SDSS_z':'SDSS_Z',
     'GaiaEDR3_G':'GAIAEDR3_G',
     'GaiaEDR3_BP':'GAIAEDR3_BP',
     'GaiaEDR3_RP':'GAIAEDR3_RP',
     })



class mkfcatfn(object):
     """docstring for mkspeccat"""
     def __init__(self, **kwargs):
          super(mkfcatfn, self).__init__()

          self.rcatfile = kwargs.get('RCAT','./temp.fits')
          self.fcatfile = kwargs.get('FCAT',None)
          self.verbose = kwargs.get('verbose',False)

          print('... initializing mkfcatfn')

          self.rcat = Table.read(self.rcatfile,format='fits')

          if self.fcatfile is None:
               self.fcatfile = self.rcatfile.replace('rcat','fcat').replace('fits','h5')

          self.output = self.fcatfile

          self.catfilterarr = CATFILTERARR
          self.photbands = list(self.catfilterarr.keys())

          # self.GM = genmod.GenMod()
          # self.GM._initspecnn(nnpath='{0}/conroy_lab/pacargile/ThePayne/Hecto_FAL/YSTANN.h5'.format(holypath),NNtype='YST1')
          # self.GM._initphotnn(self.photbands,nnpath='{0}/conroy_lab/pacargile/ThePayne/SED/VARRV/'.format(holypath))

          self.GM = genmod.GenMod() 
          self.GM._initspecnn(
               nnpath=specNN,
               NNtype=NNtype,
               Cnnpath=contNN)
          self.GM._initphotnn(
               self.photbands,
               nnpath=photNN,
               )

          self.survey = kwargs.get('survey',None)

          print('... Running mkfcat at {0}'.format(datetime.now()))
          sys.stdout.flush()
          sys.stderr.flush()
          self.__call__()
          print('... Finished running mkfcat at {0}'.format(datetime.now()))

     def compiledata(self,starinfo):
          # initialize output dict
          output = {}

          # read in SEGUE data
          SD = SegueData(survey=starinfo['survey'],tileID=starinfo['tileID'],mjd=starinfo['mjd'])
          data = SD.getdata(GaiaID=starinfo['gaiaID'])

          ###
          phot = {}
          for kk in self.catfilterarr.keys():
               pp = self.catfilterarr[kk]
               if ((data['phot'][pp] > 5.0) &
                    (data['phot'][pp] < 90.0) & 
                    (np.abs(data['phot'][pp+'_ERR']) < 90.0) & 
                    ~np.isnan(data['phot'][pp]) & 
                    ~np.isnan(data['phot'][pp+'_ERR'])
                    ):
                    filtersys = kk.split('_')[0]
                    filtername = kk.split('_')[1]

                    if filtersys == 'PS':
                         if data['phot'][pp] <= 14.0:
                              continue
                         photdata = data['phot'][pp]
                         photerr = np.sqrt((data['phot'][pp+'_ERR']**2.0)+((0.02**2.0)))
                    elif filtersys == '2MASS':
                         if data['phot'][pp] <= 5.0:
                              continue
                         photdata = data['phot'][pp]
                         photerr = np.sqrt((data['phot'][pp+'_ERR']**2.0)+((0.05**2.0)))
                    elif (filtersys == 'WISE') or (filtersys == 'UNWISE'):
                         if data['phot'][pp] <= 8.0:
                              continue
                         photdata = data['phot'][pp]
                         photerr = np.sqrt((data['phot'][pp+'_ERR']**2.0)+((0.05**2.0)))
                    elif filtersys == 'SDSS':
                         if data['phot'][pp] <= 12.0:
                              continue
                         if filtername == 'u':
                              photdata = data['phot'][pp] - 0.04
                              photerr = np.sqrt((data['phot'][pp+'_ERR']**2.0)+((0.05**2.0)))
                         elif filtername == 'i':
                              photdata = data['phot'][pp] + 0.02
                              photerr = np.sqrt((data['phot'][pp+'_ERR']**2.0)+((0.02**2.0)))
                         else:
                              photdata = data['phot'][pp]
                              photerr = np.sqrt((data['phot'][pp+'_ERR']**2.0)+((0.02**2.0)))
                    elif filtersys == 'GaiaEDR3':
                         if filtername == 'G':
                              photdata = data['phot'][pp+'_CORRECTED']
                              photerr = np.sqrt((data['phot'][pp+'_ERR']**2.0)+((0.05**2.0)))
                         if filtername == 'BP':
                              photdata = data['phot'][pp]
                              photerr = np.sqrt((data['phot'][pp+'_ERR']**2.0)+((0.05**2.0)))
                         if filtername == 'RP':
                              photdata = data['phot'][pp]
                              photerr = np.sqrt((data['phot'][pp+'_ERR']**2.0)+((0.05**2.0)))
                    else:
                         photdata = data['phot'][pp]
                         photerr = data['phot'][pp+'_ERR']

                    phot[kk] = [float(photdata),float(photerr)]

          output['obs_phot']  = {x:phot[x][0] for x in phot.keys()}
          output['obs_ephot'] = {x:phot[x][1] for x in phot.keys()}

          sedpars = ([
               starinfo['Teff'],
               starinfo['log(g)'],
               starinfo['[Fe/H]'],
               starinfo['[a/Fe]'],
               starinfo['log(R)'],
               starinfo['Dist']*1000.0,
               starinfo['Av'],
               ])
          sed = self.GM.genphot(sedpars)
          output['mod_phot'] = sed
          
          ####
          specdata = data['spec']
          spec = {}
          cond = specdata[2] != 0.0
          spec['WAVE']   = specdata[0][cond]
          spec['FLUX']   = specdata[1][cond]
          spec['E_FLUX'] = 1.0/np.sqrt(specdata[2][cond])
          spec['LSF']  = specdata[-1][cond]

          cond = np.isfinite(spec['FLUX']) & np.isfinite(spec['E_FLUX']) & (spec['LSF'] > 0.0)
          spec['WAVE']   = spec['WAVE'][cond]
          spec['FLUX']   = spec['FLUX'][cond]
          spec['E_FLUX'] = spec['E_FLUX'][cond]
          spec['LSF']    = spec['LSF'][cond]

          # create the WRESL array
          spec['WRESL'] = (spec['WAVE'] * spec['LSF']) / speedoflight

          cond = (spec['WAVE'] > 4750.0) & (spec['WAVE'] < 5500.0)
          spec['WAVE']   = spec['WAVE'][cond]
          spec['FLUX']   = spec['FLUX'][cond]
          spec['E_FLUX'] = spec['E_FLUX'][cond]
          spec['WRESL']  = spec['WRESL'][cond]

          medflux = np.median(spec['FLUX'])
          # spec['FLUX']   = spec['FLUX']/medflux
          # spec['E_FLUX'] = spec['E_FLUX']/medflux

          output['obs_wave'] = spec['WAVE']
          output['obs_flux'] = spec['FLUX']
          output['obs_eflux'] = spec['E_FLUX']
          output['obs_wresl'] = spec['WRESL']

          specpars = ([
               starinfo['Teff'],
               starinfo['log(g)'],
               starinfo['[Fe/H]'],
               starinfo['[a/Fe]'],
               starinfo['Vrad'],
               starinfo['Vrot'],
               np.nan,
               spec['WRESL'],
               starinfo['pc_0'],
               starinfo['pc_1'],
               starinfo['pc_2'],
               starinfo['pc_3'],
               ])

          specmod = self.GM.genspec(specpars,outwave=spec['WAVE'],modpoly=True)
          specmod_wave = specmod[0]
          specmod_flux = specmod[1]
          specmod_flux_m = specmod[1]*medflux

          output['mod_wave'] = specmod_wave
          output['mod_flux'] = specmod_flux_m
          output['norm_mod_flux'] = specmod_flux

          output['resid'] = output['obs_flux'] / output['mod_flux']
          
          modwave_s = output['mod_wave']*(1.0-(starinfo['Vrad']/speedoflight))
          output['mod_wave_rest'] = modwave_s
          
          return output

     def __call__(self):

          with h5py.File(self.output,'w') as outfile:
               photbandslist = np.array([x.encode('ascii') for x in self.photbands])
               outfile.create_dataset('photbands',data=photbandslist)

               rcat_i = self.rcat[np.isfinite(self.rcat['lnZ'])]

               for ii,rcat_ii in enumerate(rcat_i):
                    if self.verbose:
                         print('... Working on: GaiaEDR3 ID = {0} ({1}/{2})'.format(rcat_ii['GAIAEDR3_ID'],ii,len(rcat_i)))
                         sys.stdout.flush()

                    starinfo = {}
                    if self.survey == None:
                         starinfo['survey'] = 'SEGUE'
                    starinfo['tileID']  = rcat_ii['PLATE']
                    starinfo['mjd']     = rcat_ii['MJD']
                    starinfo['gaiaID']  = rcat_ii['GAIAEDR3_ID']

                    # best fit parameters
                    starinfo['Teff']   = rcat_ii['Teff']
                    starinfo['log(g)'] = rcat_ii['logg']
                    starinfo['[Fe/H]'] = rcat_ii['FeH']
                    starinfo['[a/Fe]'] = rcat_ii['aFe']
                    starinfo['Vrad']   = rcat_ii['Vrad']
                    starinfo['Vrot']   = rcat_ii['Vrot']
                    starinfo['pc_0']   = rcat_ii['pc_0']
                    starinfo['pc_1']   = rcat_ii['pc_1']
                    starinfo['pc_2']   = rcat_ii['pc_2']
                    starinfo['pc_3']   = rcat_ii['pc_3']

                    starinfo['log(R)'] = rcat_ii['logR']
                    starinfo['Dist']   = rcat_ii['Dist']
                    starinfo['Av']     = rcat_ii['Av']
                    
                    outputdata = self.compiledata(starinfo)
                    
                    for par in ['obs','mod']:
                         starinfo['par'] = par

                         outfile.create_dataset(
                              '{tileID}/{mjd}/{gaiaID}/{par}_wave'.format(**starinfo),
                              data=outputdata[par+'_wave'],compression='gzip')

                         outfile.create_dataset(
                              '{tileID}/{mjd}/{gaiaID}/{par}_flux'.format(**starinfo),
                              data=outputdata[par+'_flux'],compression='gzip')

                         if par == 'obs':
                              outfile.create_dataset(
                                   '{tileID}/{mjd}/{gaiaID}/{par}_eflux'.format(**starinfo),
                                   data=outputdata[par+'_eflux'],compression='gzip')

                              outfile.create_dataset(
                                   '{tileID}/{mjd}/{gaiaID}/{par}_wresl'.format(**starinfo),
                                   data=outputdata[par+'_wresl'],compression='gzip')

                         phot_i = []
                         for kk in self.photbands:
                              try:
                                   phot_i.append(outputdata[par+'_phot'][kk])
                              except KeyError:
                                   phot_i.append(np.nan)

                         outfile.create_dataset(
                              '{tileID}/{mjd}/{gaiaID}/{par}_phot'.format(**starinfo),
                              data=np.array(phot_i),compression='gzip')
                         
                    outfile.create_dataset(
                         '{tileID}/{mjd}/{gaiaID}/resid'.format(**starinfo),
                         data=outputdata['resid'],compression='gzip')

                    outfile.create_dataset(
                         '{tileID}/{mjd}/{gaiaID}/mod_wave_rest'.format(**starinfo),
                         data=outputdata['mod_wave_rest'],compression='gzip')

                    outfile.create_dataset(
                         '{tileID}/{mjd}/{gaiaID}/norm_mod_flux'.format(**starinfo),
                         data=outputdata['norm_mod_flux'],compression='gzip')


if __name__ == '__main__':
     parser = argparse.ArgumentParser()
     parser.add_argument('--RCAT',help='rcat path',type=str,default='./tmp.fits')
     parser.add_argument('--verbose', help='verbose sdtout',type=bool,choices=[True,False],default=False)
     args = parser.parse_args()
     mkfcatfn(**vars(args))
