from minesweeper import fitstar
from astropy.table import Table
import sys,argparse,os,glob
import numpy as np

from scipy import constants
speedoflight = constants.c / 1000.0

fwhm_sigma = 2.0*np.sqrt(2.0*np.log(2.0))

import socket,os
hostname = socket.gethostname()
if hostname[:4] == 'holy':
    holypath   = '/n/holyscratch01'
    # specNN = '{}/conroy_lab/pacargile/ThePayne/Hecto_FAL/lowres/YSTANN_4000_7000_spec.h5'.format(holypath)
    # contNN = '{}/conroy_lab/pacargile/ThePayne/Hecto_FAL/lowres/YSTANN_4000_7000_cont.h5'.format(holypath)
    specNN = '{}/conroy_lab/pacargile/ThePayne/train/optfal/v256/modV0_spec_LinNet_R5K_WL445_565.h5'.format(holypath)
    contNN = '{}/conroy_lab/pacargile/ThePayne/train/optfal/v256/modV0_cont_LinNet_R12K_WL445_565.h5'.format(holypath)
    NNtype = 'LinNet'
    photNN = '{}/conroy_lab/pacargile/ThePayne/SED/VARRV/'.format(holypath)
    MISTgrid = '{}/conroy_lab/pacargile/MIST/MIST_2.0_spot_EEPtrk_small.h5'.format(holypath)
    seguedir = '/n/holystore01/LABS/conroy_lab/Lab/SEGUE/'
    outdir  = '/n/holyscratch01/conroy_lab/pacargile/SEGUE/'
else:
    # specNN = '/Users/pcargile/Astro/ThePayne/YSdata/lowres/YSTANN_4000_7000_spec.h5'
    # specNN = '/Users/pcargile/Astro/ThePayne/YSdata/YSTANN.h5'
    # contNN = '/Users/pcargile/Astro/ThePayne/YSdata/lowres/YSTANN_4000_7000_cont.h5'
    specNN = '/Users/pcargile/Astro/ThePayne/train_grid/optfal/v256/modV0_spec_LinNet_R5K_WL445_565.h5'
    contNN = '/Users/pcargile/Astro/ThePayne/train_grid/optfal/v256/modV0_cont_LinNet_R12K_WL445_565.h5'
    NNtype = 'LinNet'
    photNN = '/Users/pcargile/Astro/GITREPOS/ThePayne/data/photANN/'
    MISTgrid = '/Users/pcargile/Astro/MIST/MIST_v2.0_spot/MIST_2.0_spot_EEPtrk_small.h5'
    seguedir = '/Users/pcargile/Astro/SEGUE/'
    outdir  = '/Users/pcargile/Astro/SEGUE/'


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

def run(survey=None,tileID=None,index=None,GaiaID=None,FiberID=None,mjd=None,version='VX'):

    # read in SEGUE data
    SD = SegueData(survey=survey,tileID=tileID,mjd=mjd)

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


    samplefile = 'segue_gaiaID_{GAIAID}_fibID_{FIBID}_plate_{PLATEID}_mjd_{MJD}_{VER}_samp.dat'.format(
            FIBID=data['phot']['FIBERID'],
            GAIAID=data['phot']['GAIAEDR3_ID'],
            PLATEID=data['phot']['PLATE'],
            MJD=data['phot']['MJD'],
            VER=version)

    outputfile = '{OUTDIR}/samples/{SAMPLEFILE}'.format(
            OUTDIR=outdir,
            SAMPLEFILE=samplefile)

    print('-- Running survey={0} tile={1} mjd={2}'.format(survey,tileID,data['phot']['MJD']))
    print('   ... GaiaEDR3 ID = {0}'.format(data['phot']['GAIAEDR3_ID']))
    print('   ... SEGUE FIBER ID = {0}'.format(data['phot']['FIBERID']))
    print('   ... Plug RA / Dec = {0} / {1}'.format(data['phot']['PLUG_RA'],data['phot']['PLUG_DEC']))
    sys.stdout.flush()

    # get RV estimate
    EZ      = data['phot']['ELODIE_Z']
    E_RV    = EZ * speedoflight
    sspp_RV = data['phot']['RV_ADOP']
    if (np.abs(E_RV-sspp_RV) < 50.0):
        RVest = sspp_RV
        RVrange = 150.0
        RVest_src = 'SSPP'
    else:
        RVest = 0.0
        RVrange = 500.0
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
    usedphot = {}
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

            usedphot[photbands[pp]] = pp
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

    # cond = (spec['WAVE'] > 4000.0) & (spec['WAVE'] < 7000.0)
    # cond = (spec['WAVE'] > 5150.0) & (spec['WAVE'] < 5300.0)
    # cond = (spec['WAVE'] > 4455.0) & (spec['WAVE'] < 5645.0)
    # cond = (spec['WAVE'] > 5000.0) & (spec['WAVE'] < 5500.0)
    cond = (spec['WAVE'] > 4750.0) & (spec['WAVE'] < 5500.0)
    spec['WAVE']   = spec['WAVE'][cond]
    spec['FLUX']   = spec['FLUX'][cond]
    spec['E_FLUX'] = spec['E_FLUX'][cond]
    spec['WRESL']  = spec['WRESL'][cond]

    # # mask out H-beta
    # cond = (spec['WAVE'] < 4800.0) | (spec['WAVE'] > 4900.0)
    # spec['WAVE']   = spec['WAVE'][cond]
    # spec['FLUX']   = spec['FLUX'][cond]
    # spec['E_FLUX'] = spec['E_FLUX'][cond]
    # spec['WRESL']  = spec['WRESL'][cond]

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

    print('    ... Building Input Dict')
    # build input dict
    inputdict = {}

    inputdict['specANNpath'] = specNN
    inputdict['contANNpath'] = contNN
    inputdict['NNtype'] = NNtype
    inputdict['photANNpath'] = photNN
    inputdict['MISTpath'] = MISTgrid
    inputdict['isochrone_prior'] = True
    inputdict['ageweight'] = True
    inputdict['mistdif'] = True
    inputdict['Rvfree'] = False

    inputdict['phot'] = phot

    inputdict['spec'] = {}
    inputdict['spec']['obs_wave']  = spec['WAVE']
    inputdict['spec']['obs_flux']  = spec['FLUX']
    inputdict['spec']['obs_eflux'] = spec['E_FLUX']
    inputdict['spec']['modpoly'] = True
    inputdict['spec']['convertair'] = False

    maxdist = 1000.0/(parallax-10.0*parallax_error)
    if maxdist < 0.0:
        maxdist = np.inf
    mindist = 1000.0/(parallax+10.0*parallax_error)

    print('    ... Building Sampler Options')

    # set parameter for sampler
    inputdict['sampler'] = {}
    inputdict['sampler']['samplertype'] = 'Static'
    inputdict['sampler']['samplemethod'] = 'rwalk'
    # inputdict['sampler']['samplemethod'] = 'unif'
    inputdict['sampler']['npoints'] = 500
    # inputdict['sampler']['npoints'] = 150
    inputdict['sampler']['samplerbounds'] = 'multi'
    inputdict['sampler']['flushnum'] = 250
    inputdict['sampler']['delta_logz_final'] = 0.01
    inputdict['sampler']['bootstrap'] = 0
    inputdict['sampler']['walks'] = 25
    # inputdict['sampler']['walks'] = 3

    ############## Priors ################
    print('    ... Building Prior Options')
    inputdict['priordict'] = {}

    inputdict['priordict']['EEP'] = {'pv_uniform':[100,808]}
    inputdict['priordict']['initial_Mass'] = {'pv_uniform':[0.2,3.0]}
    inputdict['priordict']['initial_[Fe/H]'] = {'pv_uniform':[-4.0,0.5]}
    inputdict['priordict']['initial_[a/Fe]'] = {'pv_uniform':[-0.2,0.6]}
    inputdict['priordict']['Dist']   = {'pv_uniform':[max([1.0,mindist]),min([maxdist,200000.0])]}
    inputdict['priordict']['Av']     = {'pv_tgaussian':[0.0,3.0*Av,Av,Av*0.15]}

    inputdict['priordict']['Age'] = {'uniform':[1.0,17.0]}
    inputdict['priordict']['Parallax'] =  {'gaussian':[parallax,parallax_error]}

    numpoly = 4
    coeffarr = [[1.0,0.5],[0.0,0.5],[0.0,0.5]] +[[0.0,0.25] for ii in range(numpoly-3)]
    inputdict['priordict']['blaze_coeff'] = coeffarr
    inputdict['priordict']['Vrad'] = {'pv_uniform':[RVest - RVrange, RVest + RVrange]}
    # inputdict['priordict']['Vrot'] = {'pv_tgaussian':[0.0, 200.0, 25.0, 25.0]}
    # inputdict['priordict']['Vrot'] = {'pv_uniform':[0.0, 100.0]}
    inputdict['priordict']['Vrot'] = {'pv_beta':[1.05,1.5,0.0,250.0]}
    inputdict['priordict']['Inst_R'] = {'fixed':spec['WRESL']}
    # inputdict['priordict']['Inst_R'] = {'fixed':3000.0}

    inputdict['priordict']['IMF'] = {'IMF_type':'Kroupa'}
    inputdict['priordict']['GAL'] = {'lb_coords':[float(data['phot']['L']),float(data['phot']['B'])]}

    inputdict['priordict']['GALAGE'] = {}
    inputdict['priordict']['GALAGE']['lb_coords'] = [float(data['phot']['L']),float(data['phot']['B'])]
    inputdict['priordict']['GALAGE']['pars'] = ({
        'thin': {'min':1.0,'max':14.0},
        'thick':{'min':6.0,'max':14.0,'mean':10.0,'sigma':2.0},
        'halo': {'min':8.0,'max':14.0,'mean':12.0,'sigma':2.0},
        })

    # inputdict['priordict']['VROT'] = {'giant':{'a':-10.0,'c':7.0,'n':1.0},'dwarf':{'a':-10.0,'c':10.0,'n':0.4}}

    inputdict['output'] = outputfile
    print('    ... Writing to: {}'.format(inputdict['output']))

    print('---------------')
    if 'spec' in inputdict.keys():
        print('    SPEC:')
        print('       Median Spec Flux: ')
        print('          {0}'.format(np.median(inputdict['spec']['obs_flux']*medflux)))
        print('       Median Spec Err_Flux:')
        print('          {0}'.format(np.median(inputdict['spec']['obs_eflux']*medflux)))
        print('       Fitting Wavelengths:')
        print('          {0} -- {1}'.format(
            min(inputdict['spec']['obs_wave']),
            max(inputdict['spec']['obs_wave'])))
        print('       Min Resolution of Spectrum: {0:n}'.format(np.min(spec['WAVE']/(spec['WRESL']))))
        print('       Max Resolution of Spectrum: {0:n}'.format(np.max(spec['WAVE']/(spec['WRESL']))))        
        print('       Mean Resolution of Spectrum: {0:n}'.format(np.median(spec['WAVE']/(spec['WRESL']))))
        print('       Fitting w/ ANN: {0}'.format(inputdict['NNtype']))

    print('---------------')
    if 'phot' in inputdict.keys():
        print('    PHOT:')
        for kk in inputdict['phot'].keys():
            print('       {0} = {1} +/- {2}'.format(kk,inputdict['phot'][kk][0],inputdict['phot'][kk][1]))

    print('---------------')
    print('    PRIORS:')
    for kk in inputdict['priordict'].keys():
      if kk == 'blaze_coeff':
           pass
      elif kk in ['IMF','GAL','GALAGE','VROT','VTOT','ALPHA']:
           print('       Turned on {0} prior'.format(kk))
      else:
           try:
                for kk2 in inputdict['priordict'][kk].keys():
                     if (kk2 == 'uniform') or (kk2 == 'pv_uniform'):
                          print('       {0}: min={1} max={2}'.format(kk,inputdict['priordict'][kk][kk2][0],inputdict['priordict'][kk][kk2][1]))
                     if (kk2 == 'gaussian') or (kk2 == 'pv_gaussian'):
                          print('       {0}: N({1},{2})'.format(kk,inputdict['priordict'][kk][kk2][0],inputdict['priordict'][kk][kk2][1]))
                     if (kk2 == 'tgaussian') or (kk2 == 'pv_tgaussian'):
                          print('       {0}: N({1},{2}) [{3} - {4}]'.format(
                               kk,inputdict['priordict'][kk][kk2][2],inputdict['priordict'][kk][kk2][3],
                               inputdict['priordict'][kk][kk2][0],inputdict['priordict'][kk][kk2][1]))
                     if kk2 == 'pv_exp':
                          print('       {0}: EXP({1},{2})'.format(kk,inputdict['priordict'][kk][kk2][0],inputdict['priordict'][kk][kk2][1]))
                     if kk2 == 'pv_loguniform':
                          print('       {0}: log(uniform({1},{2}))'.format(kk,inputdict['priordict'][kk][kk2][0],inputdict['priordict'][kk][kk2][1]))
                     if kk2 == 'fixed':
                          if kk == 'Inst_R':
                            if not isinstance(inputdict['priordict'][kk][kk2],float):
                                print('       {0}: fixed'.format(kk))
                            else:
                                print('       {0}: fixed({1})'.format(kk,inputdict['priordict'][kk][kk2]))
                          else:
                            print('       {0}: fixed({1})'.format(kk,inputdict['priordict'][kk][kk2]))

           except:
                print('       {0}: {1}'.format(kk,inputdict['priordict'][kk]))

    print('--------------')


    sys.stdout.flush()
    FS = fitstar.FitMS()
    results = FS.run(inputdict=inputdict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--survey',help='SEGUE Survey ID',type=str,
        choices=['SEGUE','SEGUE_clusters'],default='SEGUE')
    parser.add_argument('--tileID',help='tile ID number',type=int,default=1660)
    parser.add_argument('--index',help='Index of star in acat',type=int,default=None)
    parser.add_argument('--GaiaID',help='Gaia EDR3 ID of star',type=int,default=None)
    parser.add_argument('--FiberID',help='Fiber ID of star',type=int,default=None)
    parser.add_argument('--mjd',help='MJD of plate to run',type=int,default=None)
    parser.add_argument("--version", "-v", help="run version",type=str,default='VX')
    args = parser.parse_args()
    run(
        survey=args.survey,
        tileID=args.tileID,
        index=args.index,
        GaiaID=args.GaiaID,
        FiberID=args.FiberID,
        mjd=args.mjd,
        version=args.version)
