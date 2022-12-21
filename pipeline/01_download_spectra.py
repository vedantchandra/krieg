from astropy.table import Table
import urllib.request
import time
import sys
import os
import numpy as np

with open('/n/home03/vchandra/outerhalo/09_sdss5/pipeline/control/redux.txt', 'r') as file:
	redux = file.read().replace('\n','')

try:
	os.mkdir('/n/holyscratch01/conroy_lab/vchandra/sdss5/spectra/%s/' % redux)
except:
	print('spectro redux directory already exists')

halo = Table.read('/n/holyscratch01/conroy_lab/vchandra/sdss5/catalogs/spAll_halo.fits')

with open('/n/home03/vchandra/outerhalo/09_sdss5/pipeline/control/halocartons.txt', 'r') as file:
    halocartons = file.read().splitlines()
    
print('There are %i spectra from %i cartons' % (len(halo), len(halocartons)))
def make_boss_download_location(r):
	#field, mjd, spec_file = row["FIELD","MJD","SPEC_FILE"]
	return f"https://data.sdss5.org/sas/sdsswork/bhm/boss/spectro/redux/%s/spectra/lite/{r['FIELD']:06}/{r['MJD']}/{r['SPEC_FILE'].strip()}" % redux
auth_user = 'sdss5'
auth_passwd = 'panoptic-5'
outfolder = '/n/holyscratch01/conroy_lab/vchandra/sdss5/spectra/%s/' % redux
start = time.time()
		

# SELECT CARTON AND DOWNLOAD'
print(halocartons)
carton_idx = int(sys.argv[1])
print(carton_idx)
carton = halocartons[carton_idx]
print(carton)
tab = halo[halo['carton'] == carton]

cart = carton.strip()

print('downloading %s carton: %i spectra' % (cart, len(tab)))

outdir = outfolder + cart + '/'

try:
	os.mkdir(outdir)
except:
	print('%s dir already exists' % cart)
	
download_names = [
		make_boss_download_location(row) for row in tab
	]

outfile_names = [
		os.path.join(outdir, spec_file.strip()) for spec_file in tab["SPEC_FILE"]
	]


for i, (inpath, outpath) in enumerate(zip(download_names, outfile_names)):
	
	success = False

	if i % 100 == 0:
		print('%i done...' % i)   

	if os.path.exists(outpath):
		#print('spectrum %s exists, skipping...' % outpath)
		continue
	ntry = 0
	while not success:
		ntry += 1
		try:
			passman = urllib.request.HTTPPasswordMgrWithDefaultRealm()
			passman.add_password(None, inpath, auth_user, auth_passwd)
			authhandler = urllib.request.HTTPBasicAuthHandler(passman)
			opener = urllib.request.build_opener(authhandler)
			urllib.request.install_opener(opener)
			urllib.request.urlretrieve(inpath, outpath)
			success = True
		except Exception as e:
			print(e)
			print('spectrum %s failed, retrying...' % outpath)
			if ntry == 3:
				print('skipping %s...' % outpath)
				break
			else:
				pass
				#time.sleep(1)

print(f"Total took {time.time()-start:.1f}s")
