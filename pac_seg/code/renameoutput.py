import glob
import os

archive_dir = '/n/holystore01/LABS/conroy_lab/Lab/SEGUE/results/V1.0/'

# cornerplt_list  = glob.glob(archive_dir+'plots/*corner.png')
# compmodplt_list = glob.glob(archive_dir+'plots/*compmod.png')
pars_list = glob.glob(archive_dir+'pars/*.pars' )
# samples_list = glob.glob(archive_dir+'samples/*_samp.dat*')

# for crf in cornerplt_list:
#     oldfile = crf
#     newfile = crf.replace('_corner.png','_MSG_corner.png')
#     os.rename(oldfile,newfile)

# for crf in compmodplt_list:
#     oldfile = crf
#     newfile = crf.replace('_compmod.png','_MSG_compmod.png')
#     os.rename(oldfile,newfile)

totlen = len(pars_list)    
for ii,crf in enumerate(pars_list):
    if ii % 1000 == 0:
        print('{0}/{1}'.format(ii+1,totlen),end='\r')
    oldfile = crf
    newfile = crf.replace('.pars','_MSG.pars')
    os.rename(oldfile,newfile)

# for crf in samples_list:
#     oldfile = crf
#     newfile = crf.replace('_samp.dat.gz','_MSG_samp.dat.gz')
#     os.rename(oldfile,newfile)
