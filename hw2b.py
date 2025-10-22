import numpy as np
from numpy import isfinite
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from marvin.tools.cube import Cube
from marvin.tools.maps import Maps
cube = Cube(plateifu='8080-3702')
try:
    ha_wcs2d = cube.wcs.celestial
except AttributeError:
    ha_wcs2d = WCS(cube.header, naxis=2)
maps  = Maps(plateifu='8080-3702')
haflux = maps.emline_gflux_ha_6564
ha_data = np.array(haflux.value, dtype=float)
ivar = haflux.ivar
mask = haflux.mask
ha_data[(~np.isfinite(ha_data)) | (ivar <= 0) | (mask != 0)] = np.nan
hdul = fits.open("member.uid___A001_X12a3_X775.S02_sci.spw19.cube.I.pbcor.fits")
hoh  = hdul[0].header
bc   = np.squeeze(hdul[0].data)
bbc  = np.nan_to_num(bc, nan=0.0)
rf   = hoh["RESTFRQ"]
c    = 299792.458
delf = abs(hoh["CDELT3"])
vw   = c*(delf/rf)
m0_full = np.sum(bbc, axis=0) * vw
alma_wcs2d = WCS(hoh,naxis=2)
ha_on_alma, footprint = reproject_interp((ha_data, ha_wcs2d),
                                         alma_wcs2d,
                                         shape_out=m0_full.shape,
                                         return_footprint=True)
ha_on_alma = np.array(ha_on_alma, float)
ha_on_alma[~np.isfinite(ha_on_alma)] = np.nan
m0 = np.array(m0_full, float)
m0[~np.isfinite(m0)] = np.nan
ha_pos = ha_on_alma[np.isfinite(ha_on_alma) & (ha_on_alma > 0)]
if ha_pos.size:
    vmin, vmax = np.nanpercentile(ha_pos, [5, 99])
else:
    vmin, vmax = 0.0, 1.0

vmap = maps.getMap('stellar_vel')
v = np.array(vmap.value, float)     # km/s
ny, nx = m0_full.shape 
vmin1, vmax2 = (np.nanpercentile(ha_on_alma[np.isfinite(ha_on_alma)], [5, 99]) if ha_on_alma[np.isfinite(ha_on_alma)].size else (0, 1))
manga_wcs2d = cube.wcs.celestial
try:
    v_on_alma, _ = reproject_interp(
        (v, manga_wcs2d), alma_wcs2d,
        shape_out=(ny, nx),
        order='nearest-neighbor',          
        return_footprint=True
    )
except TypeError:
    v_on_alma, _ = reproject_interp(
        (v, manga_wcs2d), alma_wcs2d,
        shape_out=(ny, nx),
        order='nearest',                   
        return_footprint=True
    )
v_on_alma[~np.isfinite(v_on_alma)] = np.nan
finite = v_on_alma[np.isfinite(v_on_alma)]
vm = (np.nanpercentile(np.abs(finite), 99) if finite.size else 200)


smap = maps.getMap('stellar_sigma')        
sigma = np.array(smap.value, float)
try:
    sigma_on_alma, _ = reproject_interp(
        (sigma, manga_wcs2d), alma_wcs2d,
        shape_out=(ny, nx),
        order='nearest-neighbor',      
        return_footprint=True
    )
except TypeError:
    sigma_on_alma, _ = reproject_interp(
        (sigma, manga_wcs2d), alma_wcs2d,
        shape_out=(ny, nx),
        order='nearest',               
        return_footprint=True
    )
sigma_on_alma[~np.isfinite(sigma_on_alma)] = np.nan
finite = sigma_on_alma[np.isfinite(sigma_on_alma)]
smin, smax = (np.nanpercentile(finite, [5, 95]))

#part2
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(ha_on_alma, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
cbar = plt.colorbar(im, ax=ax); cbar.set_label(f'Hα flux ({haflux.unit})')
pos = m0[(m0 > 0) & np.isfinite(m0)]
lo, hi = np.nanpercentile(pos, [65, 99.5])   
levels = np.linspace(lo, hi, 15)             
cs = ax.contour(m0, levels=levels, colors='cyan', linewidths=1.4)
ax.clabel(cs, inline=True, fmt='%.3g', fontsize=7)
ax.set_xlabel('ALMA pixels (x)'); ax.set_ylabel('ALMA pixels (y)')
ax.set_title('Hα reprojected to ALMA grid')
ax.set_xlim(290, 380)
ax.set_ylim(290, 380) 
plt.savefig('ha_on_alma_with_co_m0_contours.png') 
plt.tight_layout()
plt.show()

#part3
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(
    v_on_alma, origin='lower', cmap='RdBu_r',
    vmin=-vm, vmax=+vm,
    interpolation='nearest',              
    extent=[0, nx, 0, ny]
)
plt.colorbar(im, ax=ax, label=f"Stellar velocity ({vmap.unit})")
ax.set_xlim(0, nx); ax.set_ylim(0, ny)
ax.set_aspect('equal', adjustable='box')
ax.set_title('MaNGA stellar velocity reprojected to ALMA grid')
plt.tight_layout()
plt.savefig('stellar_velocity_on_alma_grid.png')
plt.show()

#part4
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(
    sigma_on_alma, origin='lower', cmap='viridis',
    vmin=smin, vmax=smax,
    interpolation='nearest',          
    extent=[0, nx, 0, ny]
)
plt.colorbar(im, ax=ax, label=f"Stellar σ ({smap.unit})")
ax.set_xlim(0, nx); ax.set_ylim(0, ny)
ax.set_aspect('equal', adjustable='box') 
ax.set_title('MaNGA stellar velocity dispersion on ALMA grid')
plt.tight_layout()
plt.savefig('stellar_sigma_on_alma_grid.png', dpi=200)
plt.show()
