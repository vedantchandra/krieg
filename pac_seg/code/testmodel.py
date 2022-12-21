from jax import jit,lax,jacfwd
from jax import random as jrandom
import jax.numpy as jnp

import sys,itertools,os,argparse
from datetime import datetime

import numpy as np
from astropy.table import Table

from quantiles import quantile

from misty.predict import GenModJax as GenMIST
from Payne.jax.genmod import GenMod
from Payne.jax.fitutils import airtovacuum
# from Payne.fitting.genmod import GenMod

from scipy import constants
speedoflight = constants.c / 1000.0

fwhm_sigma = 2.0*np.sqrt(2.0*np.log(2.0))

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

import socket
hostname = socket.gethostname()
if hostname[:4] == 'holy':
    holypath   = os.environ.get('HOLYSCRATCH')
    photNN = '{0}/conroy_lab/pacargile/ThePayne/SED/VARRV/'.format(holypath)
    specNN = '{0}/conroy_lab/pacargile/ThePayne/Hecto_FAL/YSTANN_wvt.h5'.format(holypath)
    mistNN = '{0}/conroy_lab/pacargile/MISTy/models/modV2.h5'.format(holypath)
    GBSdir = '{0}/conroy_lab/pacargile/GBS/'.format(holypath)
    NNtype = 'YST2'
else:
    # specNN = '/Users/pcargile/Astro/ThePayne/YSdata/YSTANN.h5'
    # NNtype = 'YST1'
    specNN = '/Users/pcargile/Astro/ThePayne/train_grid/optfal/v256/modV0_spec_LinNet_R5K_WL445_565.h5'
    contNN = '/Users/pcargile/Astro/ThePayne/train_grid/optfal/v256/modV0_cont_LinNet_R12K_WL445_565.h5'
    NNtype = 'LinNet'
    photNN = '/Users/pcargile/Astro/GITREPOS/ThePayne/data/photANN/'
    mistNN = '/Users/pcargile/Astro/MISTy/train/v512/v1/modV1.h5'
    GBSdir = '/Users/pcargile/Astro/GBS/'

photbands = ({
    'GAIAEDR3_G':'GaiaEDR3_G',
    'GAIAEDR3_BP':'GaiaEDR3_BP',
    'GAIAEDR3_RP':'GaiaEDR3_RP',
    'PS_G':'PS_g',
    'PS_R':'PS_r',
    'PS_I':'PS_i',
    'PS_Z':'PS_z',
    'PS_Y':'PS_y',
    'TMASS_J':'2MASS_J',
    'TMASS_H':'2MASS_H',
    'TMASS_K':'2MASS_Ks',
    'UNWISE_W1':'WISE_W1',
    'UNWISE_W2':'WISE_W2',
    'SDSS_U':'SDSS_u',
    'SDSS_G':'SDSS_g',
    'SDSS_R':'SDSS_r',
    'SDSS_I':'SDSS_i',
    'SDSS_Z':'SDSS_z',
    })

from SEGUEdata import SegueData

def getdata(survey=None,tileID=None,index=None,GaiaID=None,FiberID=None):

    # read in SEGUE data
    SD = SegueData(survey=survey,tileID=tileID)

    if index is not None:
        data = SD.getdata(index=index)
    if GaiaID is not None:
        data = SD.getdata(GaiaID=GaiaID)
    if FiberID is not None:
        data = SD.getdata(FiberID=FiberID)

    try:
        assert data['phot']
    except AssertionError:
        print('Warning: User did not pass a valid selection criteria')
        raise

    starname = 'GaiaEDR3_{}'.format(data['phot']['GAIAEDR3_ID'])

    print('-- Running survey={0} tile={1}'.format(survey,tileID))
    print('   ... GaiaEDR3 ID = {0}'.format(data['phot']['GAIAEDR3_ID']))
    print('   ... SEGUE FIBER ID = {0}'.format(data['phot']['FIBERID']))
    print('   ... Plug RA / Dec = {0} / {1}'.format(data['phot']['PLUG_RA'],data['phot']['PLUG_DEC']))

    # get RV estimate
    EZ      = data['phot']['ELODIE_Z']
    E_RV    = EZ * speedoflight
    sspp_RV = data['phot']['RV_ADOP']
    if sspp_RV > -9999.0:
        RVest = sspp_RV
        RVest_src = 'SSPP'
    elif E_RV != 0.0:
        RVest = E_RV
        RVest_src = 'ELODIE'
    else:
        RVest = np.nan
        RVest_src = 'NULL'

    # get Parallax
    parallax = float(data['phot']['GAIAEDR3_PARALLAX_CORRECTED'])
    parallax_error = float(data['phot']['GAIAEDR3_PARALLAX_ERROR'])

    # get max Av estimate
    Ebv = float(data['phot']['EBV'])
    Av = 3.1 * Ebv * 0.86

    print('   ... RV estimate = {0} ({1})'.format(RVest,RVest_src))
    print('   ... GaiaEDR3 Parallax = {0:n} +/- {1:n}'.format(parallax,parallax_error))
    print('   ... SFD Av = {0:n}'.format(Av))

    print('    ... Building Phot')
    phot = {}
    filtarr = []
    for pp in photbands.keys():
        if ((data['phot'][pp] > 5.0) &
            (data['phot'][pp] < 90.0) & 
            (np.abs(data['phot'][pp+'_ERR']) < 90.0) & 
            ~np.isnan(data['phot'][pp]) & 
            ~np.isnan(data['phot'][pp+'_ERR'])
            ):
            filtersys = photbands[pp].split('_')[0]
            filtername = photbands[pp].split('_')[1]

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

            filtarr.append(photbands[pp])
            phot[photbands[pp]] = [float(photdata),float(photerr)]


    print('    ... Building Spec')
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

    # cond = (spec['WAVE'] > 3850.0) & (spec['WAVE'] < 8900.0)
    # spec['WAVE']   = spec['WAVE'][cond]
    # spec['FLUX']   = spec['FLUX'][cond]
    # spec['E_FLUX'] = spec['E_FLUX'][cond]
    # spec['WRESL']  = spec['WRESL'][cond]

    cond = (spec['WAVE'] > 4500.0) & (spec['WAVE'] < 5650.0)
    # cond = (spec['WAVE'] > 5150.0) & (spec['WAVE'] < 5300.0)
    spec['WAVE']   = spec['WAVE'][cond]
    spec['FLUX']   = spec['FLUX'][cond]
    spec['E_FLUX'] = spec['E_FLUX'][cond]
    spec['WRESL']  = spec['WRESL'][cond]

    # # mask out Na doublet due to ISM absorption
    # cond = (spec['WAVE'] < 5850.0) | (spec['WAVE'] > 5950.0)
    # spec['WAVE']   = spec['WAVE'][cond]
    # spec['FLUX']   = spec['FLUX'][cond]
    # spec['E_FLUX'] = spec['E_FLUX'][cond]
    # spec['WRESL']  = spec['WRESL'][cond]

    # # mask out telluric features
    # cond = (spec['WAVE'] < 7500.0) | (spec['WAVE'] > 7700.0)
    # spec['WAVE']   = spec['WAVE'][cond]
    # spec['FLUX']   = spec['FLUX'][cond]
    # spec['E_FLUX'] = spec['E_FLUX'][cond]
    # spec['WRESL']  = spec['WRESL'][cond]

    medflux = np.median(spec['FLUX'])
    spec['FLUX']   = spec['FLUX']/medflux
    spec['E_FLUX'] = spec['E_FLUX']/medflux


    return ({
        'starname':starname,
        'spec':[spec['WAVE'],spec['FLUX'],spec['E_FLUX']],
        'lsf':spec['WRESL'],
        'phot':phot,'filtarr':filtarr,
        'parallax':[parallax,parallax_error],
        'RVest':RVest,
        'SFD_Av':Av})

data = getdata(survey='SEGUE',tileID=1660,index=0)

specwave_in,specflux_in,speceflux_in = data['spec']
specwave_in  = jnp.asarray(specwave_in,dtype=float)
specflux_in  = jnp.asarray(specflux_in,dtype=float)
speceflux_in = jnp.asarray(speceflux_in,dtype=float)

specwave_in = airtovacuum(specwave_in)

phot_in    = jnp.asarray([data['phot'][xx][0] for xx in data['filtarr']],dtype=float)
photerr_in = jnp.asarray([data['phot'][xx][1] for xx in data['filtarr']],dtype=float)

rng_key = jrandom.PRNGKey(0)
rng = np.random.default_rng()

GM = GenMod()
GM._initspecnn(nnpath=specNN, Cnnpath=contNN, NNtype=NNtype)
GM._initphotnn(data['filtarr'],nnpath=photNN)
GMIST = GenMIST.modpred(nnpath=mistNN,nntype='LinNet',normed=True)

genspecfn = GM.genspec
genphotfn = GM.genphot
genMISTfn = GMIST.getMIST

teff = 5770.0
logg = 4.44
feh = 0.0
afe = 0.0
vrad = 0.0
vrot = 2.0
vmic = 1.0
instr = data['lsf']

pc0 = 1.0
pc1 = 0.0
pc2 = 0.0
pc3 = 0.0

specpars = [teff,logg,feh,afe,vrad,vrot,vmic,instr]
specpars += [pc0,pc1,pc2,pc3]

specmod_est = genspecfn(specpars,outwave=specwave_in,modpoly=True)

fig,ax = plt.subplots(1,1)
ax.plot(specmod_est[0],specmod_est[1])
fig.savefig('test.pdf')