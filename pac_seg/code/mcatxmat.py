import numpy as np
from astropy.table import Table,hstack,vstack,join
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import glob,argparse,os
import gc

import socket
hostname = socket.gethostname()
if hostname[:4] == 'holy':
     datadir = '/n/holystore01/LABS/conroy_lab/Lab/SEGUE/'
     mcatdir = '/n/holystore01/LABS/conroy_lab/Lab/gaia/edr3/gall2/catalogs/'
else:
     datadir = '/Users/pcargile/Astro/BOSS/data/'
     mcatdir = './'

mcatfiledict = ({
    +05.0:'gall2_EDR3_b+05.0_+06.0.fits.gz',
    +06.0:'gall2_EDR3_b+06.0_+07.0.fits.gz',
    +07.0:'gall2_EDR3_b+07.0_+08.0.fits.gz',
    +08.0:'gall2_EDR3_b+08.0_+09.0.fits.gz',
    +09.0:'gall2_EDR3_b+09.0_+10.0.fits.gz',
    +10.0:'gall2_EDR3_b+10.0_+12.0.fits.gz',
    +12.0:'gall2_EDR3_b+12.0_+15.0.fits.gz',
    +15.0:'gall2_EDR3_b+15.0_+20.0.fits.gz',
    +20.0:'gall2_EDR3_b+20.0_+25.0.fits.gz',
    +25.0:'gall2_EDR3_b+25.0_+35.0.fits.gz',
    +35.0:'gall2_EDR3_b+35.0_+45.0.fits.gz',
    +45.0:'gall2_EDR3_b+45.0_+90.0.fits.gz',
    -06.0:'gall2_EDR3_b-06.0_-05.0.fits.gz',
    -07.0:'gall2_EDR3_b-07.0_-06.0.fits.gz',
    -08.0:'gall2_EDR3_b-08.0_-07.0.fits.gz',
    -09.0:'gall2_EDR3_b-09.0_-08.0.fits.gz',
    -10.0:'gall2_EDR3_b-10.0_-09.0.fits.gz',
    -12.0:'gall2_EDR3_b-12.0_-10.0.fits.gz',
    -15.0:'gall2_EDR3_b-15.0_-12.0.fits.gz',
    -20.0:'gall2_EDR3_b-20.0_-15.0.fits.gz',
    -25.0:'gall2_EDR3_b-25.0_-20.0.fits.gz',
    -30.0:'gall2_EDR3_b-30.0_-25.0.fits.gz',
    -32.0:'gall2_EDR3_b-32.0_-30.0.fits.gz',
    -33.0:'gall2_EDR3_b-33.0_-32.0.fits.gz',
    -35.0:'gall2_EDR3_b-35.0_-33.0.fits.gz',
    -45.0:'gall2_EDR3_b-45.0_-35.0.fits.gz',
    -90.0:'gall2_EDR3_b-90.0_-45.0.fits.gz',
    })

bbins = list(mcatfiledict.keys())
bbins.sort()

mcatcol = ([
    'GAIAEDR3_ID',
    'RA',
    'DEC',
    'L',
    'B',
    'EBV',
    'PS_G',
    'PS_R',
    'PS_I',
    'PS_Z',
    'PS_Y',
    'SDSS_U',
    'SDSS_G',
    'SDSS_R',
    'SDSS_I',
    'SDSS_Z',
    'TMASS_J',
    'TMASS_H',
    'TMASS_K',
    'WISE_W1',
    'WISE_W2',
    'UNWISE_W1',
    'UNWISE_W2',
    'GAIAEDR3_G',
    'GAIAEDR3_BP',
    'GAIAEDR3_RP',
    'PS_G_ERR',
    'PS_R_ERR',
    'PS_I_ERR',
    'PS_Z_ERR',
    'PS_Y_ERR',
    'SDSS_U_ERR',
    'SDSS_G_ERR',
    'SDSS_R_ERR',
    'SDSS_I_ERR',
    'SDSS_Z_ERR',
    'TMASS_J_ERR',
    'TMASS_H_ERR',
    'TMASS_K_ERR',
    'WISE_W1_ERR',
    'WISE_W2_ERR',
    'UNWISE_W1_ERR',
    'UNWISE_W2_ERR',
    'GAIAEDR3_G_ERR',
    'GAIAEDR3_BP_ERR',
    'GAIAEDR3_RP_ERR',
    'GAIAEDR3_RA',
    'GAIAEDR3_DEC',
    'GAIAEDR3_RA_ERROR',
    'GAIAEDR3_DEC_ERROR',
    'GAIAEDR3_PARALLAX',
    'GAIAEDR3_PARALLAX_ERROR',
    'GAIAEDR3_PARALLAX_OVER_ERROR',
    'GAIAEDR3_PMRA',
    'GAIAEDR3_PMDEC',
    'GAIAEDR3_PMRA_ERROR',
    'GAIAEDR3_PMDEC_ERROR',
    'GAIAEDR3_PSEUDOCOLOUR',
    'GAIAEDR3_PSEUDOCOLOUR_ERROR',
    'GAIAEDR3_NU_EFF_USED_IN_ASTROMETRY',
    'GAIAEDR3_ASTROMETRIC_PARAMS_SOLVED',
    'GAIAEDR3_PHOT_BP_RP_EXCESS_FACTOR',
    'GAIAEDR3_VISIBILITY_PERIODS_USED',
    'GAIAEDR3_RUWE',
    'GAIAEDR3_IPD_GOF_HARMONIC_AMPLITUDE',
    'GAIAEDR3_G_CORRECTED',
    'GAIAEDR3_PARALLAX_CORRECTED',
    'GAIAEDR3_PHOT_BP_RP_EXCESS_FACTOR_CORRECTED',
    'GAIAEDR3_PARALLAX_PMRA_CORR',
    'GAIAEDR3_PARALLAX_PMDEC_CORR',
    'GAIAEDR3_PMRA_PMDEC_CORR',
    'GAIAEDR3_RA_DEC_CORR',
    'GAIAEDR3_RA_PARALLAX_CORR',
    'GAIAEDR3_RA_PMRA_CORR',
    'GAIAEDR3_RA_PMDEC_CORR',
    'GAIAEDR3_DEC_PARALLAX_CORR',
    'GAIAEDR3_DEC_PMRA_CORR',
    'GAIAEDR3_DEC_PMDEC_CORR',
    ])

ssppcol  = ([
    'FLAG',
    'G_MAG',
    'TEFF_ADOP',
    'TEFF_ADOP_UNC',
    'LOGG_ADOP',
    'LOGG_ADOP_UNC',
    'FEH_ADOP',
    'FEH_ADOP_UNC',
    'AFE',
    'AFE_UNC',
    'DIST_DWARF',
    'DIST_TO',
    'DIST_GIANT',
    'DIST_AGB',
    'DIST_FHB',
    'DIST_AP',
    'RV_FLAG',
    'RV_ADOP',
    'RV_ADOP_UNC',
    'INSPECT',
    'SURVEY',
    'SEGUE1_TARGET1',
    'SEGUE1_TARGET2',
    'SEGUE2_TARGET1',
    'SEGUE2_TARGET2',
    'MP_FLAG',
    ])

zbestcol = ([
    'PLATE',
    'TILE',
    'MJD',
    'FIBERID',
    'OBJID',
    'OBJTYPE',
    'PLUG_RA',
    'PLUG_DEC',
    'CLASS',
    'SUBCLASS',
    'Z',
    'Z_ERR',
    'RCHI2',
    'DOF',
    'ZWARNING',
    'SN_MEDIAN',
    'ELODIE_TEFF',
    'ELODIE_LOGG',
    'ELODIE_FEH',
    'ELODIE_Z',
    'ELODIE_RCHI2',
    'ELODIE_DOF',
    ])

def mat(survey=None,tileID=None,verbose=True,clobber=True):
    # check if clobber is set, and if not then check if acat already exisits
    acatfile = '{0}/{1}/{2}/acat.fits'.format(datadir,survey,tileID)
    if not clobber:
        if os.path.exists(acatfile):
            if verbose:
                print('... ACAT Already Exists, skipping!')
                return

    # read in tile
    if verbose:
        print('... Reading in data')
    segue_tilefile  = glob.glob('{0}/{1}/{2}/spPlate*.fits'.format(datadir,survey,tileID))[0]
    segue_ssppfile  = glob.glob('{0}/{1}/{2}/ssppOut*.fit'.format(datadir,survey,tileID))[0]
    segue_Zbestfile = glob.glob('{0}/{1}/{2}/spZbest*.fits'.format(datadir,survey,tileID))[0]

    segue_tile  = Table.read(segue_tilefile,format='fits')
    segue_sspp  = Table.read(segue_ssppfile,format='fits')
    segue_Zbest = Table.read(segue_Zbestfile,format='fits')

    segue_tile['SEGUE_INDEX'] = range(len(segue_tile))

    # build SEGUE SkyCoords
    if verbose:
        print('... Building SEGUE SkyCoords')
    segue_SC = SkyCoord(ra=segue_tile['RA']*u.degree, dec=segue_tile['DEC']*u.degree)

    # figure out B bins
    if verbose:
        print('... Finiding which mcats are needed')
    bind = np.digitize(segue_SC.galactic.b,bbins*u.deg)

    # build lists to map over
    binnedlist = ([
        [bbins[x-1],segue_tile[bind == x],segue_sspp[bind == x],segue_Zbest[bind == x]] 
        for x in np.unique(bind)])
    if verbose:
        print('... Need {} mcat files'.format(len(binnedlist)))

    for nind,seldata in enumerate(binnedlist):
        mcatfile = mcatdir + mcatfiledict[seldata[0]]
        if verbose:
            print('... Read in {}'.format(mcatfile))
        mcat = Table.read(mcatfile,format='fits')
        # print('... len(mcat) = {}'.format(len(mcat)))
        # mcat = mcat[mcatcol]

        if verbose:
            print('... Generate SkyCoords')
        segue_t = seldata[1]
        segue_s = seldata[2]
        segue_z = seldata[3]
        mcat_c  = SkyCoord(ra=mcat['GAIAEDR3_RA']*u.deg,dec=mcat['GAIAEDR3_DEC']*u.deg)
        segue_c = SkyCoord(ra=segue_t['RA']*u.deg,dec=segue_t['DEC']*u.deg)

        if verbose:
            print('... Do search')
        idxsegue, idxmcat, d2d, d3d = mcat_c.search_around_sky(segue_c, 5.0*u.arcsec)

        segue_tm = segue_t[np.unique(idxsegue)]
        segue_sm = segue_s[np.unique(idxsegue)]
        segue_zm = segue_z[np.unique(idxsegue)]
        segue_matches = segue_t[idxsegue]

        mcat_matches = mcat[idxmcat]
        mcat_matches['mcat_matsep'] = d2d * 3600.0 

        outcols = {}
        for x in zbestcol:
            outcols[x] = []
        for x in ssppcol:
            outcols[x] = []
        for x in mcatcol:
            outcols[x] = []
        outcols['SEGUE_INDEX'] = []
        outcols['mcat_matsep'] = []

        for ii,ind_i in enumerate(segue_tm['SEGUE_INDEX']):

            # check to make sure segue source isn't a SKY fiber
            if segue_tm['OBJTYPE'][ii].replace(' ','') == 'SKY':
                continue

            ind = np.argwhere(segue_matches['SEGUE_INDEX'] == ind_i)
            mcat_a = mcat_matches[ind]

            # select the correct match mcat_a -> mcat_i
            if segue_sm['G_MAG'][ii] > 0.0:
                selind = np.argmin(np.abs((segue_sm['G_MAG'][ii]+(3.1*segue_sm['EBV'][ii]))-(mcat_a['SDSS_G']+22.5)))
            else:
                sepdist = np.sqrt(
                    (segue_sm['RA'][ii]-mcat_a['GAIAEDR3_RA'])**2.0 + 
                    (segue_sm['DEC'][ii]-mcat_a['GAIAEDR3_DEC'])**2.0                     
                    )
                selind = np.argmin(sepdist)
            mcat_i = mcat_a[selind]

            for x in zbestcol:
                outcols[x].append(segue_zm[x][ii])
            for x in ssppcol:
                outcols[x].append(segue_sm[x][ii])
            for x in mcatcol:
                outcols[x].append(mcat_i[x])

            outcols['mcat_matsep'].append(mcat_i['mcat_matsep'])
            outcols['SEGUE_INDEX'].append(ind_i)

        outcat_i = Table(outcols)

        if nind == 0:
            outcat = outcat_i.copy()
        else:
            outcat = vstack([outcat,outcat_i])

        # manual clean up
        del mcat
        del segue_c
        del mcat_c
        
        del segue_t
        del segue_s
        del segue_z

        del segue_tm
        del segue_sm
        del segue_zm
        del segue_matches

        try:
            del mcat_i
            del mcat_a
            del mcat_matches
        except UnboundLocalError:
            pass

        del outcols
        del outcat_i

        gc.collect()

    if verbose:
        print('... writing acat file: {}'.format(acatfile))
    outcat.write(acatfile,format='fits',overwrite=True)

    del segue_tile
    del segue_Zbest
    del segue_sspp
    del segue_SC
    del outcat
    gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--survey',help='SEGUE Survey ID',type=str,
        choices=['SEGUE-1','SEGUE-2','SEGUE_clusters'],default='SEGUE-1')
    parser.add_argument('--tileID',help='tile ID number',type=int,default=1660)

    args = parser.parse_args()
    mat(**vars(args))
