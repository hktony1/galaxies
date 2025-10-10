import numpy as np
import matplotlib.pyplot as plt
import scipy as sp 
from astropy.io import fits
hdul = fits.open("member.uid___A001_X12a3_X775.S02_sci.spw19.cube.I.pbcor.fits") #how to read 
hoh=hdul[0].header #header of heads or just the data ya know
CoD=hdul[0].data #Cube of Data
bc=np.squeeze(CoD) #better cube after removing the useless first dimension
bbc=np.nan_to_num(bc, nan=0) #better better cube after changing nan to zero so i can do math
rf=hoh["RESTFRQ"] #rest frequency
c=299792.458 #speed of light in km/s
delf=abs(hoh["CDELT3"]) #delta frequency
vw=c*(delf/rf) #velocity width of each channel
#how i found out what to slice out arrays between 0:42 and 630:671 are all nan
#deeper into the hw and i realize i spent way too much time on this part for it to be useless lol
for i in range(1918):
    if np.sum(bbc[i][42]) != 0.0:
        print(bbc[i][42])
        np.sum(bbc[i][630]) != 0.0
        print(bbc[i][630])
fc=bbc[:,43:629,43:629] #final cube with the slices
for i in range(586):
    if np.sum(fc[0,i,43])!=0.0:
        print(0,i)
m0=np.sum(fc,axis=0)*vw #moment 0 map in Jy/beam * km/s
plt.figure()
plt.imshow(m0,cmap='inferno',origin='lower',vmin=-0.3,vmax=0.5)
plt.colorbar(label='Jy/beam * km/s')
plt.title('Moment 0 Map')
plt.xlim([220,360])
plt.ylim([220,360])
plt.xlabel('RA Pixels')
plt.ylabel('DEC Pixels')
plt.savefig('moment0_map.png')
plt.show()
nc=fc.shape[0] #number of channels
nca=np.arange(nc) #number of channels array
crval3=hoh["CRVAL3"] 
crpix3=hoh["CRPIX3"]
crdelt3=hoh["CDELT3"] 
f=crval3+(nca+1-crpix3)*crdelt3 #frequency array
vel=c*(rf-f)/rf #velocity array
velplus2=vel[:,None,None] #velocity array with two extra dimensions 
numerator=np.sum(velplus2*fc,axis=0) #numerator of moment 1 map
denmoinator=np.sum(fc,axis=0) #denominator of moment 1 map without the vw
m1=np.divide(numerator,denmoinator,where=(denmoinator!=0)) #moment 1 map in Hz
m1[denmoinator==0]=np.nan #setting the places with no data to nan
plt.figure()
plt.imshow(m1,cmap='coolwarm',origin='lower',vmin=6700,vmax=6800)
plt.colorbar(label='km/s')
plt.title('Moment 1 Map')
plt.xlim([220,360])
plt.ylim([220,360])
plt.xlabel('RA Pixels')
plt.ylabel('DEC Pixels')
plt.savefig('moment1_map.png')
plt.show()
m1plus1=m1[None,:,:] #moment 1 map with one extra dimension
ps=(velplus2-m1plus1)**2 #the parentheses part of the numerator of moment 2 map squared
numerator2=np.sum(ps*fc,axis=0) #numerator of moment 2 map
m2=np.sqrt(abs(np.divide(numerator2,denmoinator,where=(denmoinator!=0)))) #moment 2 map in km/s
m2[denmoinator==0]=np.nan #setting the places with no data to nan
plt.figure()
plt.imshow(m2,cmap='inferno',origin='lower',vmin=0,vmax=300)
plt.colorbar(label='km/s')
plt.title('Moment 2 Map')
plt.xlim([220,360])
plt.ylim([220,360])
plt.xlabel('RA Pixels')
plt.ylabel('DEC Pixels')
plt.savefig('moment2_map.png')
plt.show()
vbluemin = -150.0   #km/s relative to system
vbluemax = -100.0
#to get the sytemic velocity
m0_roi = m0[220:360, 220:360] #region of interest
m1_roi = m1[220:360, 220:360]
hi = np.nanpercentile(m0_roi, 75) #high threshold for systemic velocity
mask_hi = m0_roi > hi #mask high S/N pixels
v_sys = np.nanmedian(m1_roi[mask_hi]) if np.any(mask_hi) else np.nanmedian(m1_roi)
v_rel = vel - v_sys #velocity relative to systemic
chan_blue = (v_rel >= vbluemin) & (v_rel <= vbluemax) #array for blue channels
if np.any(chan_blue): #check if there are any blue channels
    m0_blue_full = np.sum(fc[chan_blue, :, :], axis=0) * vw   
else:
    m0_blue_full = np.full_like(m0, np.nan)
decA, decB = 250, 270   # region where the higest blue signal from m1 map
mask_box = np.zeros_like(m0_blue_full, dtype=bool) #mask for the box
mask_box[decA:decB, raA:raB] = True #setting the box to True
m0_blue_box = np.where(mask_box, m0_blue_full, np.nan) #blue wing in the box only
plt.figure()
bgmin = np.nanpercentile(m0[220:360, 220:360], 5) #min/max
bgmax = np.nanpercentile(m0[220:360, 220:360], 99) 
plt.imshow(m0, origin='lower', cmap='inferno', vmin=bgmin, vmax=bgmax)
vals = m0_blue_box[np.isfinite(m0_blue_box)] #values in the box
if vals.size: #check if there are any finite values
    levels = np.percentile(vals, [60, 75, 90]) #contour levels
    plt.contour(m0_blue_box, levels=levels, colors='cyan', linewidths=1.6) 
else:
    print("No finite blue-wing signal in the specified box.")
plt.colorbar(label='Jy/beam * km/s')
plt.title("Countor map of blue wing over Moment 0 Map")
plt.xlim([220,360])
plt.ylim([220,360])
plt.xlabel('RA Pixels'); plt.ylabel('DEC Pixels')
#plt.tight_layout()
plt.savefig('blue_box_on_m02.png')
plt.show()
