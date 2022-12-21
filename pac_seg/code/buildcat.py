from astropy.table import Table
import sys,argparse
import numpy as np
from datetime import datetime

import socket,os
hostname = socket.gethostname()
if hostname[:4] == 'holy':
    seguedir = '/n/holystore01/LABS/conroy_lab/Lab/SEGUE/'
    outdir  = '/n/holyscratch01/conroy_lab/pacargile/SEGUE/'
else:
    seguedir = '/Users/pcargile/Astro/SEGUE/'
    outdir  = '/Users/pcargile/Astro/SEGUE/'

def run(catalog=None,version='VX',runtype='MSG',clobber=False):

    # takes in table of GaiaEDR3 IDs
    runlist = Table.read(catalog,format='fits')

    # Finds all references in acat of this star
    acat = Table.read('{0}/catalogs/SEGUE_acat.fits'.format(seguedir),format='fits')
    acat = acat[np.in1d(acat['GAIAEDR3_ID'],runlist['GAIAEDR3_ID'])]

    nnr = 0

    if not clobber:
        # checks to see if star has been run
        removerows = []
        for ii,acat_i in enumerate(acat):
            parsfile = 'segue_gaiaID_{GAIAID}_fibID_{FIBID}_plate_{PLATEID}_mjd_{MJD}_{VER}_{RUNTYPE}.pars'.format(
                    FIBID=acat_i['FIBERID'],
                    GAIAID=acat_i['GAIAEDR3_ID'],
                    PLATEID=acat_i['PLATE'],
                    MJD=acat_i['MJD'],
                    VER=version,
                    RUNTYPE=runtype)
            outputfile = '{SEGUEDIR}/results/{VERSION}/pars/{PARSFILE}'.format(
                    SEGUEDIR=seguedir,
                    VERSION=version,
                    PARSFILE=parsfile)

            if os.path.exists(outputfile):
                # print('... found {}, not running again'.format(parsfile))
                removerows.append(ii)
                nnr = nnr + 1
        acat.remove_rows(removerows)
        print('... Found {} that will not run again.'.format(nnr))

    # writes rows to file which contains FIBERID, PLATE, MJD, GAIAEDR3_ID
    out = Table()
    out['FIBERID']     = acat['FIBERID']    
    out['PLATE']       = acat['PLATE']      
    out['MJD']         = acat['MJD']        
    out['GAIAEDR3_ID'] = acat['GAIAEDR3_ID']

    # writes out list of stars to run to a fits file
    runtime = str(datetime.now()).replace(' ','_')
    outfile = '{0}/runlists/runlist_{1}_{2}_{3}.fits'.format(outdir,version,runtype,runtime)
    out.write(outfile,format='fits',overwrite=True)
    print('... Built {0} cat for {1} stars'.format(outfile,len(out)))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog',help='catalog to use as input',type=str,default=None)
    parser.add_argument('--version',help='Version of run',type=str,default='V0')
    parser.add_argument('--runtype',help='TP or MSG',type=str,default='MSG')
    args = parser.parse_args()

    run(catalog=args.catalog,version=args.version,runtype=args.runtype,clobber=False)

