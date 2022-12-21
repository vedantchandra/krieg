from astropy.table import Table
from phaseafy import phaseafy
import numpy as np


t = Table.read('SEGUE_acat.fits',format='fits')

t['Dist'] = 1.0/t['GAIAEDR3_PARALLAX_CORRECTED']
t['Dist_lerr'] = t['Dist'] - (1.0/(t['GAIAEDR3_PARALLAX_CORRECTED'] + t['GAIAEDR3_PARALLAX_ERROR']))
t['Dist_uerr'] = (1.0/(t['GAIAEDR3_PARALLAX_CORRECTED'] - t['GAIAEDR3_PARALLAX_ERROR'])) - t['Dist']

t['Vrad'] = t['RV_ADOP']
t['Vrad_err'] = t['RV_ADOP_UNC']

cond = (
    (t['Dist'] > 0.0) & 
    (t['Dist_uerr'] > 0.0) & 
    (t['Dist_lerr'] > 0.0) & 
    np.isfinite(t['Vrad']) & 
    (t['Vrad_err'] > 0.0))

t['Dist'][~cond] = np.nan
t['Vrad'][~cond] = np.nan

# doing phase-a-fy
PH = phaseafy()

# add phaseafy stuff
phpar = ([
    'R_gal','X_gal','Y_gal','Z_gal',
    'Vx_gal','Vy_gal','Vz_gal',
    'Vr_gal','Vphi_gal','Vtheta_gal','V_tan','V_gsr',
    'Lx','Ly','Lz','Ltot'])
for POTN in PH.potentials.keys():
    phpar.append('E_kin_{0}'.format(POTN))
    phpar.append('E_pot_{0}'.format(POTN))
    phpar.append('E_tot_{0}'.format(POTN))
    phpar.append('circLz_{0}'.format(POTN))
    phpar.append('circLtot_{0}'.format(POTN))

for pp in phpar:
    t[pp] = np.nan * np.ones(len(t))
    t[pp+'_err'] = np.nan * np.ones(len(t))
t['phase_cov'] = np.nan * np.ones(len(t))

print('... Computing phase info')
for ii,t_i in enumerate(t):
    if cond[ii]:
        print('... Working on phase for : {0} ({1}/{2})'.format(t_i['GAIAEDR3_ID'],ii,len(t)))
        t_i = PH.calcphase(t_i,nsamples=50000,verbose=False)
t.write('SEGUE_seguephase.fits',overwrite=True)