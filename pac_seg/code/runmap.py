import runcat
from multiprocessing import Pool

def runmap(ind):
    survey = 'SEGUE'
    catalog = '/n/holyscratch01/conroy_lab/pacargile/SEGUE/catalogs/SEGUE_H3.fits'
    runcat.run(survey=survey,catalog=catalog,ind=ind,version='V0')


if __name__ == '__main__':
    try:
        # determine the number of cpu's and make sure we have access to them all
        numcpus = open('/proc/cpuinfo').read().count('processor\t:')
        import os
        os.system("taskset -p -c 0-{NCPUS} {PID}".format(NCPUS=numcpus-1,PID=os.getpid()))
    except IOError:
        pass

    with Pool(48) as p:
        list(p.map(runmap,range(0,1000)))