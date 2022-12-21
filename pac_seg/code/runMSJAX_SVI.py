import numpyro
# numpyro.set_platform(platform='cpu')
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, autoguide, initialization
import numpyro.distributions as distfn
from numpyro.diagnostics import print_summary
from numpyro.distributions import constraints
from numpyro.contrib.control_flow import cond

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

import socket,os
hostname = socket.gethostname()
if hostname[:4] == 'holy':
    holypath   = os.environ.get('HOLYSCRATCH')
    photNN = '{0}/conroy_lab/pacargile/ThePayne/SED/VARRV/'.format(holypath)
    specNN = '{0}/conroy_lab/pacargile/ThePayne/Hecto_FAL/YSTANN_wvt.h5'.format(holypath)
    mistNN = '{0}/conroy_lab/pacargile/MISTy/models/modV2.h5'.format(holypath)
    datadir = '{}/conroy_lab/pacargile/SEGUE/data/'
    NNtype = 'YST2'
else:
    # specNN = '/Users/pcargile/Astro/ThePayne/YSdata/YSTANN.h5'
    # NNtype = 'YST1'
    specNN = '/Users/pcargile/Astro/ThePayne/train_grid/optfal/v256/modV0_spec_LinNet_R5K_WL445_565.h5'
    contNN = '/Users/pcargile/Astro/ThePayne/train_grid/optfal/v256/modV0_cont_LinNet_R12K_WL445_565.h5'
    NNtype = 'LinNet'
    photNN = '/Users/pcargile/Astro/GITREPOS/ThePayne/data/photANN/'
    mistNN = '/Users/pcargile/Astro/MISTy/train/v512/v1/modV1.h5'
    datadir = '/Users/pcargile/Astro/SEGUE/data/'

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

class IMF_Prior(distfn.Distribution):
    support = constraints.interval(0.5,3.0)
    def __init__(self,alpha_low=1.3, alpha_high=2.3, mass_break=0.5):
        """
        Apply a Kroupa-like broken IMF prior over the provided initial mass grid.
        Parameters
        ----------

        alpha_low : float, optional
            Power-law slope for the low-mass component of the IMF.
            Default is `1.3`.
        alpha_high : float, optional
            Power-law slope for the high-mass component of the IMF.
            Default is `2.3`.
        mass_break : float, optional
            The mass where we transition from `alpha_low` to `alpha_high`.
            Default is `0.5`.
        """
        # self.mass = mass
        super().__init__(batch_shape = (), event_shape=())

        self.alpha_low = alpha_low
        self.alpha_high = alpha_high
        self.mass_break = mass_break

        # Compute normalization.
        norm_low = mass_break ** (1. - alpha_low) / (alpha_high - 1.)
        norm_high = 0.08 ** (1. - alpha_low) / (alpha_low - 1.)  # H-burning limit
        norm_high -= mass_break ** (1. - alpha_low) / (alpha_low - 1.)
        norm = norm_low + norm_high
        self.lognorm = jnp.log(norm)
        
    def sample(self, key, sample_shape=()):
        raise NotImplementedError
         
    def log_prob(self, mass):
        """
        mgrid : `~numpy.ndarray` of shape (Ngrid)
            Grid of initial mass (solar units) the IMF will be evaluated over.
        Returns
        -------
        lnprior : `~numpy.ndarray` of shape (Ngrid)
            The corresponding unnormalized ln(prior).
        """

        # # make sure mgrid is not a single float
        # if not isinstance(mgrid,Iterable):
        #     mgrid = jnp.array([mgrid])

        # # Initialize log-prior.
        # lnprior = jnp.zeros_like(mgrid) - jnp.inf

        # # Low mass.
        # low_mass = (mgrid <= mass_break) & (mgrid > 0.08)
        # lnprior[low_mass] = -alpha_low * jnp.log(mgrid[low_mass])

        # # High mass.
        # high_mass = mgrid > mass_break
        # lnprior[high_mass] = (-alpha_high * jnp.log(mgrid[high_mass])
        #                       + (alpha_high - alpha_low) * jnp.log(mass_break))

        # lnprior = self.lnprior_high(mass)

        def lnprior_high(mass):
            return (-self.alpha_high * jnp.log(mass) 
                + (self.alpha_high - self.alpha_low) * jnp.log(self.mass_break))
        def lnprior_low(mass):
            return -self.alpha_low * jnp.log(mass)

        lnprior = lax.cond(mass > self.mass_break,lnprior_high,lnprior_low,mass)

        return lnprior - self.lognorm

def determineprior(parname,priorinfo):
    if priorinfo[0] == 'uniform':
        return numpyro.sample(parname,distfn.Uniform(*priorinfo[1]))
    if priorinfo[0] == 'normal':
        return numpyro.sample(parname,distfn.Normal(*priorinfo[1]))
    if priorinfo[0] == 'fixed':
        return numpyro.deterministic(parname,priorinfo[1])

# define the model
def model(
    specwave=None,specobs=None,specobserr=None,
    photobs=None,photobserr=None,
    genspecfn=None,
    genphotfn=None,
    genMISTfn=None,
    MISTpars=None,
    jMIST=None,
    parallax=None,
    filtarr=None,
    lsf=None,
    SFD_Av=None,
    RVest=None,
    vmicbool=None,
    eepprior=None,
    ):

    if eepprior is not None:
        eep_i = determineprior("eep",eepprior)
    else:
        eep_i  = numpyro.sample("eep", distfn.Uniform(200,800))

    # mass_i = numpyro.sample("initial_Mass",   distfn.Uniform(0.5,3.0))
    mass_i = numpyro.sample('initial_Mass', IMF_Prior())
    feh_i  = numpyro.sample("initial_[Fe/H]", distfn.Uniform(-3.0,0.25))
    afe_i  = numpyro.sample("initial_[a/Fe]", distfn.Uniform(-0.15,0.55))

    MISTpred = genMISTfn(
        eep=eep_i,
        mass=mass_i,
        feh=feh_i,
        afe=afe_i,
        verbose=False
        )

    # dlogAgedEEP = jMIST(jnp.array([eep_i,mass_i,feh_i,afe_i]))[4][0]
    # numpyro.factor("AgeWgt_log_prob", jnp.log(dlogAgedEEP))

    MISTdict = ({
        kk:pp for kk,pp in zip(
        MISTpars,MISTpred)
        })

    # numpyro.factor('AgePrior',
    #     distfn.Uniform(low=9.0,high=10.15,validate_args=True).log_prob(MISTdict['log(Age)']))
    numpyro.factor('AgePrior',
        distfn.ImproperUniform(constraints.interval(9.0,10.15),(),event_shape=(),validate_args=True).log_prob(
            MISTdict['log(Age)']))

    teff = 10.0**MISTdict['log(Teff)']
    logg = MISTdict['log(g)']
    feh  = MISTdict['[Fe/H]']
    afe  = MISTdict['[a/Fe]']

    vrad = numpyro.sample("vrad", distfn.Uniform(RVest-25.0, RVest+25.0))
    vrot = numpyro.sample("vrot", distfn.Uniform(0.01, 25.0))

    if vmicbool:
        # Ramirez, Allende Prieto, Lambert (2013)
        # vmic_p = 1.163 + (7.808E-4)*(teff-5800.0)-0.494*(logg-4.30)-0.050*feh
        # vmic  = numpyro.sample('vmic', distfn.TruncatedDistribution(
        #     distfn.Normal(loc=vmic_p,scale=0.12),low=0.6,high=2.9))
        vmic = numpyro.sample("vmic", distfn.Uniform(0.5, 3.0))
    else:
        vmic = numpyro.deterministic('vmic',1.0)

    pc0  = numpyro.sample('pc0', distfn.Uniform(0.5,1.5))
    pc1  = numpyro.sample('pc1', distfn.Normal(0.0,0.25))
    pc2  = numpyro.sample('pc2', distfn.Normal(0.0,0.25))
    pc3  = numpyro.sample('pc3', distfn.Normal(0.0,0.25))
        
    instr_scale = numpyro.sample('instr_scale', distfn.Uniform(0.5,3.0))
    instr = lsf * instr_scale

    specpars = [teff,logg,feh,afe,vrad,vrot,vmic,instr]
    specpars += [pc0,pc1,pc2,pc3]

    specmod_est = genspecfn(specpars,outwave=specwave,modpoly=True)
    specmod_est = jnp.asarray(specmod_est[1])

    # numpyro.sample("specobs",distfn.Normal(specmod_est, 0.01), obs=specobs)
    # numpyro.sample("specobs",distfn.Normal(specmod_est, specobserr), obs=specobs)
    specjitter = numpyro.sample("specjitter", distfn.HalfNormal(0.001))
    specsig = jnp.sqrt( (specobserr**2.0) + (specjitter**2.0) )
    # specsig = specobserr
    numpyro.sample("specobs",distfn.Normal(specmod_est, specsig), obs=specobs)

    logr = MISTdict['log(R)']
    dist = numpyro.sample("dist", distfn.Uniform(1.0, 200000.0))
    # av   = numpyro.sample("av", distfn.TruncatedNormal(low=1E-6,loc=0.0,scale=3.0*SFD_Av))
    av = numpyro.sample("av", distfn.Uniform(1E-6, 3.0*SFD_Av))

    numpyro.sample("para", distfn.Normal(1000.0/dist,parallax[1]), obs=parallax[0])

    photpars = jnp.asarray([teff,logg,feh,afe,logr,dist,av,3.1])
    photmod_est = genphotfn(photpars)
    photmod_est = jnp.asarray([photmod_est[xx] for xx in filtarr])

    # numpyro.sample("photobs",distfn.Normal(photmod_est, 0.1), obs=photobs)
    # numpyro.sample("photobs",distfn.Normal(photmod_est, photobserr), obs=photobs)
    photjitter = numpyro.sample("photjitter", distfn.HalfNormal(0.001))
    photsig = jnp.sqrt( (photobserr**2.0) + (photjitter**2.0) )
    # photsig = photobserr
    numpyro.sample("photobs",distfn.Normal(photmod_est, photsig), obs=photobs)
    

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

    starinfo = {}
    starinfo['starname'] = 'GaiaEDR3_{}'.format(data['phot']['GAIAEDR3_ID'])
    starinfo['FIBERID']  = data['phot']['FIBERID']
    starinfo['MJD']      = data['phot']['MJD']
    starinfo['PLATE']    = data['phot']['PLATE']
    starinfo['GAIAEDR3_ID'] = data['phot']['GAIAEDR3_ID']

    print('-- Running survey={0} tile={1}'.format(survey,tileID))
    print('   ... GaiaEDR3 ID = {0}'.format(data['phot']['GAIAEDR3_ID']))
    print('   ... SEGUE FIBER ID = {0}'.format(data['phot']['FIBERID']))
    print('   ... Plug RA / Dec = {0} / {1}'.format(data['phot']['PLUG_RA'],data['phot']['PLUG_DEC']))
    print('   ... l / b = {0} / {1}'.format(data['phot']['L'],data['phot']['B']))

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
        'starinfo':starinfo,
        'spec':[spec['WAVE'],spec['FLUX'],spec['E_FLUX']],
        'lsf':spec['WRESL'],
        'phot':phot,'filtarr':filtarr,
        'parallax':[parallax,parallax_error],
        'RVest':RVest,
        'SFD_Av':Av,
        'l':data['phot']['L'],
        'b':data['phot']['B']})


def run(survey=None,tileID=None,index=None,GaiaID=None,FiberID=None,version='VX',progress_bar=False):

    data = getdata(survey=survey,tileID=tileID,index=index,GaiaID=GaiaID,FiberID=FiberID)

    specwave_in,specflux_in,speceflux_in = data['spec']
    specwave_in  = jnp.asarray(specwave_in,dtype=float)
    specflux_in  = jnp.asarray(specflux_in,dtype=float)
    speceflux_in = jnp.asarray(speceflux_in,dtype=float)

    # specwave_in = airtovacuum(specwave_in)

    phot_in    = jnp.asarray([data['phot'][xx][0] for xx in data['filtarr']],dtype=float)
    photerr_in = jnp.asarray([data['phot'][xx][1] for xx in data['filtarr']],dtype=float)

    rng_key = jrandom.PRNGKey(0)
    rng = np.random.default_rng()

    GM = GenMod()
    GM._initspecnn(nnpath=specNN, Cnnpath=contNN, NNtype=NNtype)
    # GM._initphotnn(data['filtarr'],nnpath=photNN)
    GM._initphotnn(None,nnpath=photNN)
    GMIST = GenMIST.modpred(nnpath=mistNN,nntype='LinNet',normed=True)

    # jit a couple of functions
    genspecfn = jit(GM.genspec)
    genphotfn = jit(GM.genphot)
    genMISTfn = jit(GMIST.getMIST)

    def gMIST(pars):
        eep,mass,feh,afe = pars
        return genMISTfn(eep=eep,mass=mass,feh=feh,afe=afe)
    jgMIST = jit(gMIST)
    Jac_genMISTfn = jacfwd(jgMIST)

    sys.stdout.flush()
    starttime = datetime.now()
    print('... Working on {0} @ {1}...'.format(data['starinfo']['starname'],starttime))

    print('--------')
    print('MODELS:')
    print('--------')
    print('Spec NN: {}'.format(specNN))
    print('Cont NN: {}'.format(contNN))
    print('NN-type: {}'.format(NNtype))
    print('Phot NN: {}'.format(photNN))
    print('MIST NN: {}'.format(mistNN))
    print('--------')
    print('PHOT:')
    print('--------')
    for ii,ff in enumerate(data['filtarr']):
        print('{0} = {1} +/- {2}'.format(ff,phot_in[ii],photerr_in[ii]))
    print('--------')
    print('SPEC:')
    print('--------')
    print('number of pixels: {0}'.format(len(specwave_in)))
    print('min/max wavelengths: {0} -- {1}'.format(specwave_in.min(),specwave_in.max()))
    print('median flux: {0}'.format(np.median(specflux_in)))
    print('median flux error: {0}'.format(np.median(speceflux_in)))
    print('SNR: {0}'.format(np.median(specflux_in/speceflux_in)))
    print('--------')
    sys.stdout.flush()

    initpars = ({
        'eep':400,
        'initial_Mass':1.00,
        # 'initial_[Fe/H]':0.0,
        # 'initial_[a/Fe]':0.0,
        'dist':1000.0/data['parallax'][0],
        # 'av':0.01,
        # 'vrad':0.0,
        # 'vmic':1.0,
        'vrot':5.0,
        'pc0':1.0,
        'pc1':0.0,
        'pc2':0.0,
        'pc3':0.0,
        'instr_scale':1.0,
        'photjitter':1E-5,
        'specjitter':1E-5,            
        })

    modelkw = ({
        'specobs':specflux_in,
        'specobserr':speceflux_in, 
        'specwave':specwave_in,
        'parallax':data['parallax'],
        'photobs':phot_in,
        'photobserr':photerr_in,
        'filtarr':data['filtarr'],
        'genspecfn':genspecfn,
        'genphotfn':genphotfn,
        'genMISTfn':genMISTfn,
        'MISTpars':GMIST.modpararr,
        'jMIST':Jac_genMISTfn,
        'lsf':data['lsf'],
        'RVest':data['RVest'],
        'SFD_Av':data['SFD_Av'],
        'vmicbool':False,
        'eepprior':['uniform',(300,800)],
        })

    # optimizer = numpyro.optim.Adam(0.1)
    # optimizer = numpyro.optim.Adagrad(1.0)
    # optimizer = numpyro.optim.Minimize()
    optimizer = numpyro.optim.ClippedAdam(0.001)
    # optimizer = numpyro.optim.SM3(0.1)    
    # guide = autoguide.AutoLaplaceApproximation(model,init_loc_fn=initialization.init_to_value(values=initpars))
    # guide = autoguide.AutoNormal(model,init_loc_fn=initialization.init_to_value(values=initpars))
    guide = autoguide.AutoLowRankMultivariateNormal(model,init_loc_fn=initialization.init_to_value(values=initpars))
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(
        rng_key, 
        20000,
        **modelkw
        )

    params = svi_result.params
    posterior = guide.sample_posterior(rng_key, params, (5000,))
    print_summary({k: v for k, v in posterior.items() if k != "mu"}, 0.89, False)

    t = Table(posterior)

    # determine extra parameter from MIST
    extrapars = [x for x in GMIST.modpararr if x not in t.keys()] 
    for kk in extrapars + ['Teff','Age']:
        t[kk] = np.nan * np.ones(len(t),dtype=float)

    for t_i in t:
        MISTpred = genMISTfn(
            eep=t_i['eep'],
            mass=t_i['initial_Mass'],
            feh=t_i['initial_[Fe/H]'],
            afe=t_i['initial_[a/Fe]'],
            verbose=False
            )
        MISTdict = ({
            kk:pp for kk,pp in zip(
            GMIST.modpararr,MISTpred)
            })

        for kk in extrapars:
            t_i[kk] = MISTdict[kk]

        t_i['Teff'] = 10.0**(t_i['log(Teff)'])
        t_i['Age']  = 10.0**(t_i['log(Age)']-9.0)

    for kk in ['Teff','log(g)','[Fe/H]','[a/Fe]','Age']:
        pars = quantile(t[kk],[0.5,0.16,0.84])
        print('{0} = {1:f} +{2:f}/-{3:f}'.format(kk,pars[0],pars[2]-pars[0],pars[0]-pars[1]))


    outfile = './output/samp_fibID_{FIBID}_gaiaID_{GAIAID}_plate_{PLATEID}_mjd_{MJD}_{VER}.fits'.format(
        FIBID=data['starinfo']['FIBERID'],
        GAIAID=data['starinfo']['GAIAEDR3_ID'],
        PLATEID=data['starinfo']['PLATE'],
        MJD=data['starinfo']['MJD'],
        VER=version)
    print('... writing samples to {}'.format(outfile))
    t.write(outfile,format='fits',overwrite=True)
    print('... Finished {0} @ {1}...'.format(data['starinfo']['starname'],datetime.now()-starttime))
    sys.stdout.flush()

    return (svi,guide,svi_result)

if __name__ == '__main__':
    starttime = datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument('--survey',help='SEGUE Survey ID',type=str,
        choices=['SEGUE','SEGUE_clusters'],default='SEGUE')
    parser.add_argument('--tileID',help='tile ID number',type=int,default=1660)
    parser.add_argument('--index',help='Index of star in acat',type=int,default=None)
    parser.add_argument('--GaiaID',help='Gaia EDR3 ID of star',type=int,default=None)
    parser.add_argument('--FiberID',help='Fiber ID of star',type=int,default=None)
    parser.add_argument("--version", "-v", help="run version",type=str,default='VX')
    args = parser.parse_args()
    run(
        survey=args.survey,
        tileID=args.tileID,
        index=args.index,
        GaiaID=args.GaiaID,
        FiberID=args.FiberID,
        version=args.version)
    print('Total Runtime: {0}'.format(datetime.now()-starttime))