from astropy.table import Table
import sys,argparse,os

import runMS
import compmod
import corner

import socket,os
hostname = socket.gethostname()
if hostname[:4] == 'holy':
    datadir = '{}/conroy_lab/pacargile/SEGUE/data/'
else:
    datadir = '/Users/pcargile/Astro/SEGUE/data/'


def run(survey=None,tileID=None,ind=None,mjd=None,version='V0'):

    mattab = Table.read('{0}/../catalogs/plate{1}.dat'.format(datadir,tileID),format='ascii')
    mattab_i = mattab[ind]
    FiberID = mattab_i['FIBERID']

    runMS.run(survey=survey,tileID=tileID,FiberID=FiberID,version=version)
    compmod.run(survey=survey,tileID=tileID,FiberID=FiberID,version=version)
    corner.run(survey=survey,tileID=tileID,FiberID=FiberID,version=version)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--survey',help='SEGUE Survey ID',type=str,
        choices=['SEGUE','SEGUE_clusters'],default='SEGUE')
    parser.add_argument('--tileID',help='tile ID number',type=int,default=1660)
    parser.add_argument('--ind',help='Index of star',type=int,default=None)
    parser.add_argument('--version',help='Version of run',type=str,default='V0')
    args = parser.parse_args()

    run(survey=args.survey,tileID=args.tileID,ind=args.ind)

