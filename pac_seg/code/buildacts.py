import os,sys,argparse
from datetime import datetime
import sys
from astropy.table import Table


import socket
hostname = socket.gethostname()
if hostname[:4] == 'holy':
     datadir = '/n/holystore01/LABS/conroy_lab/Lab/SEGUE/'
else:
     datadir = '/Users/pcargile/Astro/SEGUE/data/'

import mcatxmat
def build(survey=None,tileID=None):
    print('-- Working on {0} / {1}'.format(survey,tileID))
    sys.stdout.flush()
    starttime = datetime.now()
    mcatxmat.mat(survey=survey,tileID=tileID,verbose=True,clobber=True)
    print(' -- Finished {0} / {1} : Time = {2}'.format(survey,tileID,datetime.now()-starttime))
    sys.stdout.flush()

def run(survey=None,tilelist=None):
    if tilelist is None:
        # find all tiles in survey
        flist = next(os.walk('{0}/{1}'.format(datadir,survey)))[1]
        # turn tileIDs into ints 
        flist = [int(x) for x in flist]
    else:
        ffile = Table.read(tilelist,format='ascii')
        flist = [int(x) for x in ffile['tileID']]        

    print('Found following tiles ({0}):'.format(len(flist)))
    print(flist)
    sys.stdout.flush()

    f = lambda x : build(survey=survey,tileID=x)

    list(map(f,flist))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--survey',help='SEGUE Survey ID',type=str,
        choices=['SEGUE-1','SEGUE-2','SEGUE_clusters'],default='SEGUE-1')
    parser.add_argument('--tilelist',help='file containing tileIDs to run',type=str,default=None)
    args = parser.parse_args()
    run(**vars(args))
