import matplotlib.pyplot as plt
import os

from pyampp.gxbox.boxutils import read_b3d_h5

## load the B3D data
workdir = '/Users/fisher/myworkspace/'
b3dfile64 = os.path.join(workdir, 'b3d_data_20231014T160000_dim64x64x64.h5')
b3dfile96 = os.path.join(workdir, 'b3d_data_20231014T160000_dim96x96x96.h5')
b3dfile128 = os.path.join(workdir, 'b3d_data_20231014T160000_dim128x128x128.h5')

b3dbox64 = read_b3d_h5(b3dfile64)
b3dbox96 = read_b3d_h5(b3dfile96)
b3dbox128 = read_b3d_h5(b3dfile128)

zidx=0
fig, axs = plt.subplots(3, 3, figsize=(8, 8))
axs[0,0].imshow(b3dbox64['nlfff']['bx'][:, :,zidx])
axs[0,0].set_title('bx zidx:{} dim:64')
axs[0,1].imshow(b3dbox96['nlfff']['bx'][:, :,zidx])
axs[0,1].set_title('bx zidx:{} dim:96')
axs[0,2].imshow(b3dbox128['nlfff']['bx'][:, :,zidx])
axs[0,2].set_title('bx zidx:{} dim:128')

axs[1,0].imshow(b3dbox64['nlfff']['by'][:, :,zidx])
axs[1,0].set_title('by zidx:{} dim:96')
axs[1,1].imshow(b3dbox96['nlfff']['by'][:, :,zidx])
axs[1,1].set_title('by zidx:{} dim:96')
axs[1,2].imshow(b3dbox128['nlfff']['by'][:, :,zidx])
axs[1,2].set_title('by zidx:{} dim:128')

axs[2,0].imshow(b3dbox64['nlfff']['bz'][:, :,zidx])
axs[2,0].set_title('bz zidx:{} dim:64')
axs[2,1].imshow(b3dbox96['nlfff']['bz'][:, :,zidx])
axs[2,1].set_title('bz zidx:{} dim:96')
axs[2,2].imshow(b3dbox128['nlfff']['bz'][:, :,zidx])
axs[2,2].set_title('bz zidx:{} dim:128')


fig.tight_layout()
fig.savefig(f'bfig-3d_compare.zidx{zidx}.jpg', dpi=300, bbox_inches='tight')