from uberMS import runSVI,runNUTS
import numpy as np

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


SVI = runSVI.sviMS(specNN=specNN,photNN=photNN,mistNN=mistNN,contNN=contNN)
# NUTS = runNUTS.nutsMS(specNN=specNN,photNN=photNN,mistNN=mistNN,contNN=contNN)

from runMSJAX_SVI import getdata

survey  = 'SEGUE'
tileID  = 1664 
FiberID = 460

data = getdata(survey=survey,tileID=tileID,FiberID=FiberID)

print('--------')
print('PHOT:')
print('--------')
for ii,ff in enumerate(data['filtarr']):
    print('{0} = {1} +/- {2}'.format(ff,*data['phot'][ff]))
print('--------')
print('SPEC:')
print('--------')
print('number of pixels: {0}'.format(len(data['spec'][0])))
print('min/max wavelengths: {0} -- {1}'.format(data['spec'][0].min(),data['spec'][0].max()))
print('median flux: {0}'.format(np.median(data['spec'][1])))
print('median flux error: {0}'.format(np.median(data['spec'][2])))
print('SNR: {0}'.format(np.median(data['spec'][1]/data['spec'][2])))
print('--------')

indict = {}

indict['data'] = {}
indict['data']['spec'] = data['spec']
indict['data']['phot'] = data['phot']
indict['data']['parallax'] = data['parallax']

initpars = ({
    'eep':400,
    'initial_Mass':1.00,
    'initial_[Fe/H]':0.0,
    'initial_[a/Fe]':0.0,
    'dist':1000.0/data['parallax'][0],
    'av':0.01,
    # 'vrad':0.0,
    # 'vmic':1.0,
    # 'vstar':5.0,
    'pc0':1.0,
    'pc1':0.0,
    'pc2':0.0,
    'pc3':0.0,
    'instr_scale':1.0,
    'photjitter':1E-5,
    'specjitter':1E-5,            
    })
indict['initpars'] = initpars

indict['priors'] = {}
indict['priors']['lsf_array'] = data['lsf']
indict['priors']['initial_Mass'] = 'IMF'
indict['priors']['dist'] = ['GALAGE',{'l':data['l'],'b':data['b']}]

indict['svi'] = {'steps':10000}

indict['outfile'] = 'test.fits'

SVI.run(indict)
# NUTS.run(indict)