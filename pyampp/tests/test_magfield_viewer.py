import matplotlib.pyplot as plt

b3dtypes = ['pot', 'nlfff']
for bidx, b3dtype in enumerate(b3dtypes):
    if self.box.b3d[b3dtype] is not None:
        for k in ['bx', 'by', 'bz']:
            print(f'{b3dtype} {k}.shape: {self.box.b3d[b3dtype][k].shape}; type: {type(self.box.b3d[b3dtype][k])} ')

plt.close('all')
heights = [2, 10, 50, 190]
k='bz'
fig, axs = plt.subplots(nrows=2, ncols=len(heights), sharex=True, sharey=True)

for hidx, h in enumerate(heights):
    for bidx, b3dtype in enumerate(b3dtypes):
        ax = axs[bidx,hidx]
        if self.box.b3d[b3dtype] is not None:
            ax.imshow(self.box.b3d[b3dtype][k][:,:, h])
            ax.set_title(f'h={h} {b3dtype} {k}')
fig.tight_layout()
plt.show()

self.grid = pv.ImageData()
self.grid.dimensions = (len(x), len(y), len(z))
self.grid.spacing = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
self.grid.origin = (x.min(), y.min(), z.min())
