import time
import matplotlib.pyplot as plt
from pyAMaFiL.mag_field_wrapper import MagFieldWrapper
from pyampp.util.lff import mf_lfff
maglib = MagFieldWrapper()

potfile = '11312_hmi.M_720s.20111010_085818.W120N23CR.CEA.POT.sav'
# maglib.load_cube(potfile)
# energy_pot = maglib.energy
# print('Potential energy: ' + str(energy_pot) + ' erg')

# bndfile='11312_hmi.M_720s.20111010_085818.W120N23CR.CEA.BND.sav'

from scipy.io import readsav
# filename = bndfile
filename = potfile
sav_data = readsav(filename, python_dict = True)
box = sav_data.get('box', sav_data.get('pbox'))

box = maglib._as_dict(box[0])
# # box['BX'] = box['BX'][0]
# # box['BY'] = box['BY'][0]
# # box['BZ'] = box['BZ'][0]


bndfile = 'bnddata_20240517.pkl'
inputfile = 'inputdata_20240517.pkl'

bndfile = 'bnddata.pkl'
inputfile = 'inputdata.pkl'

import pickle

with open(bndfile, 'rb') as f:
 bnddata = pickle.load(f)

maglib_lff = mf_lfff()

maglib_lff.set_field(bnddata)

res = maglib_lff.LFFF_cube(96)

fig, axs = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True)
axs[0].imshow(bnddata,cmap='gray',origin='lower')
axs[1].imshow(res['bz'][:,:,2],cmap='gray',origin='lower')


from pyAMaFiL.mag_field_wrapper import MagFieldWrapper

t0 = time.time()
maglib = MagFieldWrapper()

maglib.load_cube_vars(res['bx'], res['by'], res['bz'], box['DR'])
# maglib.load_cube_vars(bx_lff, by_lff, bz_lff, box['DR'])
res_nlf = maglib.NLFFF()

print(f'Time taken to compute NLFFF solution 1 : {time.time() - t0} seconds')

fig, axs = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True)
axs[0].imshow(bnddata,cmap='gray',origin='lower')
axs[1].imshow(res_nlf['bz'][:,10,:],cmap='gray',origin='lower')


with open(inputfile, 'rb') as f:
 bx_lff,by_lff,bz_lff = pickle.load(f)

fig, axs = plt.subplots(ncols=3, nrows=3, sharex=True, sharey=True)
axs = axs.ravel()
# axs[0].imshow(bnddata,cmap='gray',origin='lower')
# axs[0].imshow(bz_lff[:,:,0],cmap='gray',origin='lower')
axs[0].imshow(res['bx'][:,:,0],cmap='gray',origin='lower',vmax=1500,vmin=-1500)
axs[1].imshow(bx_lff[:,:,0],cmap='gray',origin='lower',vmax=1500,vmin=-1500)
axs[2].imshow(res['bx'][:,:,0] - bx_lff[:,:,0],cmap='gray',origin='lower',vmax=500,vmin=-500)

axs[3].imshow(res['by'][:,:,0],cmap='gray',origin='lower',vmax=1500,vmin=-1500)
axs[4].imshow(by_lff[:,:,0],cmap='gray',origin='lower',vmax=1500,vmin=-1500)
axs[5].imshow(res['by'][:,:,0] - by_lff[:,:,0],cmap='gray',origin='lower',vmax=500,vmin=-500)

axs[6].imshow(res['bz'][:,:,0],cmap='gray',origin='lower',vmax=1500,vmin=-1500)
axs[7].imshow(bz_lff[:,:,0],cmap='gray',origin='lower',vmax=1500,vmin=-1500)
axs[8].imshow(res['bz'][:,:,0] - bz_lff[:,:,0],cmap='gray',origin='lower',vmax=500,vmin=-500)

axs[0].set_title('lfff')
axs[1].set_title('data')
axs[2].set_title('diff')

t1 = time.time()
maglib = MagFieldWrapper()

maglib.load_cube_vars(bx_lff, by_lff, bz_lff, box['DR'])
# maglib.load_cube_vars(res['bx'], by_lff, bz_lff, box['DR'])
res_nlf2 = maglib.NLFFF()
print(f'Time taken to compute NLFFF solution 2: {time.time() - t1} seconds')




# In [2]: t1 = time.time()
#    ...: maglib = MagFieldWrapper()
#    ...:
#    ...: # maglib.load_cube_vars(bx_lff, by_lff, bz_lff, box['DR'])
#    ...: maglib.load_cube_vars(res['bx'], res['by'], bz_lff, box['DR'])
#    ...: res_nlf2 = maglib.NLFFF()
#    ...: print(f'Time taken to compute NLFFF solution: {time.time() - t1} seconds')
#    ...:
#    ...:
# Time taken to compute NLFFF solution: 10.426038026809692 seconds

# In [3]: t1 = time.time()
#    ...: maglib = MagFieldWrapper()
#    ...:
#    ...: # maglib.load_cube_vars(bx_lff, by_lff, bz_lff, box['DR'])
#    ...: maglib.load_cube_vars(res['bx'], by_lff, bz_lff, box['DR'])
#    ...: res_nlf2 = maglib.NLFFF()
#    ...: print(f'Time taken to compute NLFFF solution 2: {time.time() - t1} seconds')
#    ...:

# Time taken to compute NLFFF solution 2: 47.814260959625244 seconds

# In [4]:

# In [4]: t1 = time.time()
#    ...: maglib = MagFieldWrapper()
#    ...:
#    ...: # maglib.load_cube_vars(bx_lff, by_lff, bz_lff, box['DR'])
#    ...: maglib.load_cube_vars(bx_lff, res['by'], bz_lff, box['DR'])
#    ...: res_nlf2 = maglib.NLFFF()
#    ...: print(f'Time taken to compute NLFFF solution 2: {time.time() - t1} seconds')
#    ...:
# Time taken to compute NLFFF solution 2: 59.9284029006958 seconds
