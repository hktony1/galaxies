import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
#WARNING NO LOOPS WILL BE USED SORRY ANNE
#data files
wavelengthfile='wavelength'
neg18m="0g-18m"
neg1m="0g-1m"
neg02m="0g-02m"
nognom="0g0m"
nog08m="0g08m"
neg130g0m="-130g0m"
neg1g0m="-1g0m"
neg030g0m="-030g0m"
pos1g0m="1g0m"
#how to read
wavelengthdata= pd.read_csv(wavelengthfile, header=None)
neg18mdata= pd.read_csv(neg18m, header=None)
neg1mdata= pd.read_csv(neg1m, header=None)
neg02mdata= pd.read_csv(neg02m, header=None)
nognomdata= pd.read_csv(nognom, header=None)
nog08mdata= pd.read_csv(nog08m, header=None)
neg130g0mdata= pd.read_csv(neg130g0m, header=None)
neg1g0mdata= pd.read_csv(neg1g0m, header=None)
neg030g0mdata= pd.read_csv(neg030g0m, header=None)
pos1g0mdata= pd.read_csv(pos1g0m, header=None)
#flux data
wavelength= wavelengthdata.iloc[0].to_numpy()
neg18mflux = neg18mdata.iloc[0].to_numpy()
neg1mflux = neg1mdata.iloc[0].to_numpy()
neg02mflux = neg02mdata.iloc[0].to_numpy()
nognomflux = nognomdata.iloc[0].to_numpy()
nog08mflux = nog08mdata.iloc[0].to_numpy()
neg130g0mflux = neg130g0mdata.iloc[0].to_numpy()
neg1g0mflux = neg1g0mdata.iloc[0].to_numpy()
neg030g0mflux = neg030g0mdata.iloc[0].to_numpy()
pos1g0mflux = pos1g0mdata.iloc[0].to_numpy()
#plotting the variable metalicity spectrums
plt.figure(figsize=(8,4))
plt.plot(wavelength,neg18mflux)
plt.plot(wavelength,neg1mflux)
plt.plot(wavelength,neg02mflux)
plt.plot(wavelength,nognomflux)
plt.plot(wavelength,nog08mflux)
plt.xlabel(r'Wavelength [$\mathrm{\AA}$]')
plt.ylabel(r"Flux [$L_\odot\,\mathrm{\AA}^{-1}$]")
plt.title("Constant  Gigayear Variable Metalicity Spectrums")
plt.xscale("log")
plt.yscale("log")
lines = plt.gca().get_lines()
plt.legend(
    (lines[0], lines[1], lines[2], lines[3], lines[4]),
    (
        r'$\log(Z/Z_\odot) = -1.8$',
        r'$\log(Z/Z_\odot) = -1.0$',
        r'$\log(Z/Z_\odot) = -0.2$',
        r'$\log(Z/Z_\odot) = 0.0$',
        r'$\log(Z/Z_\odot) = +0.8$',
    ),
    title="Metallicity",
    loc="best"
)
plt.tight_layout()
plt.savefig("Constant  Gigayear Variable Metalicity Spectrums", dpi=300)
plt.show()
#plotting the variable age spectrums
plt.figure(figsize=(8,4))
plt.plot(wavelength,neg130g0mflux)
plt.plot(wavelength,neg1g0mflux)
plt.plot(wavelength,neg030g0mflux)
plt.plot(wavelength,nognomflux)
plt.plot(wavelength,pos1g0mflux)
plt.xlabel(r'Wavelength [$\mathrm{\AA}$]')
plt.ylabel(r"Flux [$L_\odot\,\mathrm{\AA}^{-1}$]")
plt.title("Constant Metalicity Variable Age Spectrums")
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
lines = plt.gca().get_lines()
plt.legend(
    (lines[0], lines[1], lines[2], lines[3], lines[4]),
    (
        r'$t = 0.050\ \mathrm{Gyr}$',
        r'$t = 0.10\ \mathrm{Gyr}$',
        r'$t = 0.50\ \mathrm{Gyr}$',
        r'$t = 1.0\ \mathrm{Gyr}$',
        r'$t = 10.0\ \mathrm{Gyr}$',
    ),
    title="Stellar population age",
    loc="best"
)
plt.savefig("Constant Metalicity Variable Age Spectrums", dpi=300)
plt.show()
#part 1 b
#function to apply dust attenuation for a given A_V
central_flux = nognomflux
def apply_dust(flux, wavelength, AV):
    A_lambda = AV * (wavelength / 5500.0)**(-1.15)
    return flux * 10**(-0.4 * A_lambda)
#make spectra for A_V = 0, 0.5, 1.0, 1.5, 2.0 
flux_AV0  = apply_dust(central_flux, wavelength, 0.0)
flux_AV05 = apply_dust(central_flux, wavelength, 0.5)
flux_AV1  = apply_dust(central_flux, wavelength, 1.0)
flux_AV15 = apply_dust(central_flux, wavelength, 1.5)
flux_AV2  = apply_dust(central_flux, wavelength, 2.0)
plt.figure(figsize=(8,4))
plt.plot(wavelength, flux_AV0,  label=r"$A_V = 0.0$ mag")
plt.plot(wavelength, flux_AV05, label=r"$A_V = 0.5$ mag")
plt.plot(wavelength, flux_AV1,  label=r"$A_V = 1.0$ mag")
plt.plot(wavelength, flux_AV15, label=r"$A_V = 1.5$ mag")
plt.plot(wavelength, flux_AV2,  label=r"$A_V = 2.0$ mag")
plt.xlabel(r'Wavelength [$\mathrm{\AA}$]')
plt.ylabel(r"Flux [$L_\odot\,\mathrm{\AA}^{-1}$]")
plt.title("Effect of Dust Attenuation on Central Spectrum")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("Dust_Attenuation_Central_Spectrum", dpi=300)
plt.show()
#1 c
bessellB = pd.read_csv("bessell_B.dat", delim_whitespace=True, header=None, comment="#")
bessellV = pd.read_csv("bessell_V.dat", delim_whitespace=True, header=None, comment="#")
bessellR = pd.read_csv("bessell_R.dat", delim_whitespace=True, header=None, comment="#")
lam_B = bessellB[0].to_numpy()
T_B   = bessellB[1].to_numpy()
lam_V = bessellV[0].to_numpy()
T_V   = bessellV[1].to_numpy()
lam_R = bessellR[0].to_numpy()
T_R   = bessellR[1].to_numpy()
#use your 0g0m spectrum and scale it to normalize
scale_factor = 5e3   
scaled_flux = nognomflux * scale_factor
plt.figure(figsize=(8,4))
plt.plot(wavelength, scaled_flux, label=rf"Spectrum × {scale_factor:.0e}")
plt.plot(lam_B, T_B, label="B filter")
plt.plot(lam_V, T_V, label="V filter")
plt.plot(lam_R, T_R, label="R filter")
plt.xlabel(r'Wavelength [$\mathrm{\AA}$]')
plt.ylabel("Normalized Flux")
plt.title("Bessell B, V, R Filters on Spectrum")
plt.legend()
plt.tight_layout()
plt.savefig("BVR_filters_on_spectrum", dpi=300)
plt.show()
#1 d
def filt_flux(wave, flux, lam_filt, T_filt):
    #interpolate spectrum onto filter wavelength grid
    F_interp = np.interp(lam_filt, wave, flux, left=0.0, right=0.0)
    num = np.trapz(F_interp * T_filt, lam_filt)
    den = np.trapz(T_filt, lam_filt)
    return num / den
def mag(F):
    return -2.5 * np.log10(F)
#color for metallicity sequence
#constant age, variable log(Z/Zsun)
#metallicity spectra: neg18m, neg1m, neg02m, nognom, nog08m
FB_met1 = filt_flux(wavelength, neg18mflux, lam_B, T_B)
FV_met1 = filt_flux(wavelength, neg18mflux, lam_V, T_V)
FR_met1 = filt_flux(wavelength, neg18mflux, lam_R, T_R)
FB_met2 = filt_flux(wavelength, neg1mflux, lam_B, T_B)
FV_met2 = filt_flux(wavelength, neg1mflux, lam_V, T_V)
FR_met2 = filt_flux(wavelength, neg1mflux, lam_R, T_R)
FB_met3 = filt_flux(wavelength, neg02mflux, lam_B, T_B)
FV_met3 = filt_flux(wavelength, neg02mflux, lam_V, T_V)
FR_met3 = filt_flux(wavelength, neg02mflux, lam_R, T_R)
FB_met4 = filt_flux(wavelength, nognomflux, lam_B, T_B)
FV_met4 = filt_flux(wavelength, nognomflux, lam_V, T_V)
FR_met4 = filt_flux(wavelength, nognomflux, lam_R, T_R)
FB_met5 = filt_flux(wavelength, nog08mflux, lam_B, T_B)
FV_met5 = filt_flux(wavelength, nog08mflux, lam_V, T_V)
FR_met5 = filt_flux(wavelength, nog08mflux, lam_R, T_R)
mB_met1, mV_met1, mR_met1 = mag(FB_met1), mag(FV_met1), mag(FR_met1)
mB_met2, mV_met2, mR_met2 = mag(FB_met2), mag(FV_met2), mag(FR_met2)
mB_met3, mV_met3, mR_met3 = mag(FB_met3), mag(FV_met3), mag(FR_met3)
mB_met4, mV_met4, mR_met4 = mag(FB_met4), mag(FV_met4), mag(FR_met4)
mB_met5, mV_met5, mR_met5 = mag(FB_met5), mag(FV_met5), mag(FR_met5)
BV_met1 = mB_met1 - mV_met1
BV_met2 = mB_met2 - mV_met2
BV_met3 = mB_met3 - mV_met3
BV_met4 = mB_met4 - mV_met4
BV_met5 = mB_met5 - mV_met5
VR_met1 = mV_met1 - mR_met1
VR_met2 = mV_met2 - mR_met2
VR_met3 = mV_met3 - mR_met3
VR_met4 = mV_met4 - mR_met4
VR_met5 = mV_met5 - mR_met5
BV_met_list = [BV_met1, BV_met2, BV_met3, BV_met4, BV_met5]
VR_met_list = [VR_met1, VR_met2, VR_met3, VR_met4, VR_met5]
#color for age sequence
#constant metallicity, variable age
#age spectra: neg130g0m, neg1g0m, neg030g0m, nognom, pos1g0m
FB_age1 = filt_flux(wavelength, neg130g0mflux, lam_B, T_B)
FV_age1 = filt_flux(wavelength, neg130g0mflux, lam_V, T_V)
FR_age1 = filt_flux(wavelength, neg130g0mflux, lam_R, T_R)
FB_age2 = filt_flux(wavelength, neg1g0mflux, lam_B, T_B)
FV_age2 = filt_flux(wavelength, neg1g0mflux, lam_V, T_V)
FR_age2 = filt_flux(wavelength, neg1g0mflux, lam_R, T_R)
FB_age3 = filt_flux(wavelength, neg030g0mflux, lam_B, T_B)
FV_age3 = filt_flux(wavelength, neg030g0mflux, lam_V, T_V)
FR_age3 = filt_flux(wavelength, neg030g0mflux, lam_R, T_R)
FB_age4 = filt_flux(wavelength, nognomflux, lam_B, T_B)
FV_age4 = filt_flux(wavelength, nognomflux, lam_V, T_V)
FR_age4 = filt_flux(wavelength, nognomflux, lam_R, T_R)
FB_age5 = filt_flux(wavelength, pos1g0mflux, lam_B, T_B)
FV_age5 = filt_flux(wavelength, pos1g0mflux, lam_V, T_V)
FR_age5 = filt_flux(wavelength, pos1g0mflux, lam_R, T_R)
mB_age1, mV_age1, mR_age1 = mag(FB_age1), mag(FV_age1), mag(FR_age1)
mB_age2, mV_age2, mR_age2 = mag(FB_age2), mag(FV_age2), mag(FR_age2)
mB_age3, mV_age3, mR_age3 = mag(FB_age3), mag(FV_age3), mag(FR_age3)
mB_age4, mV_age4, mR_age4 = mag(FB_age4), mag(FV_age4), mag(FR_age4)
mB_age5, mV_age5, mR_age5 = mag(FB_age5), mag(FV_age5), mag(FR_age5)
BV_age1 = mB_age1 - mV_age1
BV_age2 = mB_age2 - mV_age2
BV_age3 = mB_age3 - mV_age3
BV_age4 = mB_age4 - mV_age4
BV_age5 = mB_age5 - mV_age5
VR_age1 = mV_age1 - mR_age1
VR_age2 = mV_age2 - mR_age2
VR_age3 = mV_age3 - mR_age3
VR_age4 = mV_age4 - mR_age4
VR_age5 = mV_age5 - mR_age5
BV_age_list = [BV_age1, BV_age2, BV_age3, BV_age4, BV_age5]
VR_age_list = [VR_age1, VR_age2, VR_age3, VR_age4, VR_age5]
#colors for dust sequence
FB_d0  = filt_flux(wavelength, flux_AV0,  lam_B, T_B)
FV_d0  = filt_flux(wavelength, flux_AV0,  lam_V, T_V)
FR_d0  = filt_flux(wavelength, flux_AV0,  lam_R, T_R)
FB_d05 = filt_flux(wavelength, flux_AV05, lam_B, T_B)
FV_d05 = filt_flux(wavelength, flux_AV05, lam_V, T_V)
FR_d05 = filt_flux(wavelength, flux_AV05, lam_R, T_R)
FB_d1  = filt_flux(wavelength, flux_AV1,  lam_B, T_B)
FV_d1  = filt_flux(wavelength, flux_AV1,  lam_V, T_V)
FR_d1  = filt_flux(wavelength, flux_AV1,  lam_R, T_R)
FB_d15 = filt_flux(wavelength, flux_AV15, lam_B, T_B)
FV_d15 = filt_flux(wavelength, flux_AV15, lam_V, T_V)
FR_d15 = filt_flux(wavelength, flux_AV15, lam_R, T_R)
FB_d2  = filt_flux(wavelength, flux_AV2,  lam_B, T_B)
FV_d2  = filt_flux(wavelength, flux_AV2,  lam_V, T_V)
FR_d2  = filt_flux(wavelength, flux_AV2,  lam_R, T_R)
mB_d0,  mV_d0,  mR_d0  = mag(FB_d0),  mag(FV_d0),  mag(FR_d0)
mB_d05, mV_d05, mR_d05 = mag(FB_d05), mag(FV_d05), mag(FR_d05)
mB_d1,  mV_d1,  mR_d1  = mag(FB_d1),  mag(FV_d1),  mag(FR_d1)
mB_d15, mV_d15, mR_d15 = mag(FB_d15), mag(FV_d15), mag(FR_d15)
mB_d2,  mV_d2,  mR_d2  = mag(FB_d2),  mag(FV_d2),  mag(FR_d2)
BV_d0  = mB_d0  - mV_d0
BV_d05 = mB_d05 - mV_d05
BV_d1  = mB_d1  - mV_d1
BV_d15 = mB_d15 - mV_d15
BV_d2  = mB_d2  - mV_d2
VR_d0  = mV_d0  - mR_d0
VR_d05 = mV_d05 - mR_d05
VR_d1  = mV_d1  - mR_d1
VR_d15 = mV_d15 - mR_d15
VR_d2  = mV_d2  - mR_d2
BV_dust_list = [BV_d0, BV_d05, BV_d1, BV_d15, BV_d2]
VR_dust_list = [VR_d0, VR_d05, VR_d1, VR_d15, VR_d2]
plt.figure(figsize=(6,5))
plt.plot(BV_age_list,  VR_age_list,  '-o', label='Age sequence')
plt.plot(BV_met_list,  VR_met_list,  '-s', label='Metallicity sequence')
plt.plot(BV_dust_list, VR_dust_list, '-^', label='Dust sequence')
plt.xlabel("B - V")
plt.ylabel("V - R")
plt.title("Color–color diagram: age, metallicity, dust")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ColorColor_Age_Metal_Dust", dpi=300)
plt.show()
#2
z = 0.047
hdul = fits.open("spec-0570-52266-0537.fits")
sdss_data = hdul[1].data
flux_sdss_obs = sdss_data["flux"] #observed frame flux
loglam_sdss   = sdss_data["loglam"] #log10 of observed frame wavelength
lam_sdss_obs  = 10**loglam_sdss #observed frame wavelengths 
#convert to rest frame wavelengths
lam_sdss_rest = lam_sdss_obs / (1.0 + z)
flux_sdss_rest = flux_sdss_obs*(1.0+z)
def filt_flux(lam_spec, flux_spec, lam_filt, T_filt):
    #interpolate SDSS spectrum onto the filter wavelength grid
    F_interp = np.interp(lam_filt, lam_spec, flux_spec, left=0.0, right=0.0)
    num = np.trapz(F_interp * T_filt, lam_filt)
    den = np.trapz(T_filt, lam_filt)
    return num / den
def mag(F):
    return -2.5 * np.log10(F)
FB = filt_flux(lam_sdss_rest, flux_sdss_rest, lam_B, T_B)
FV = filt_flux(lam_sdss_rest, flux_sdss_rest, lam_V, T_V)
FR = filt_flux(lam_sdss_rest, flux_sdss_rest, lam_R, T_R)
mB = mag(FB)
mV = mag(FV)
mR = mag(FR)
BV = mB - mV
VR = mV - mR
BV_sdss = BV
VR_sdss = VR
plt.figure(figsize=(6,5))
plt.plot(BV_age_list,  VR_age_list,  '-o', label='Age sequence')
plt.plot(BV_met_list,  VR_met_list,  '-s', label='Metallicity sequence')
plt.plot(BV_dust_list, VR_dust_list, '-^', label='Dust sequence')
plt.plot(BV_sdss, VR_sdss, marker='*', markersize=14, linestyle='None',
         label='SDSS galaxy')

plt.xlabel("B - V")
plt.ylabel("V - R")
plt.title("Color–color diagram with SDSS galaxy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ColorColor_with_SDSS_galaxy", dpi=300)
plt.show()
BV_age_arr = np.array(BV_age_list)
VR_age_arr = np.array(VR_age_list)
BV_met_arr = np.array(BV_met_list)
VR_met_arr = np.array(VR_met_list)
age_values   = np.array([0.05, 0.10, 0.50, 1.0, 10.0])      # Gyr
logZ_values  = np.array([-1.8, -1.0, -0.2, 0.0, 0.8]) 
dist2_age = (BV_age_arr - BV_sdss)**2 + (VR_age_arr - VR_sdss)**2
best_idx_age = np.argmin(dist2_age)
best_age = age_values[best_idx_age]
dist2_met = (BV_met_arr - BV_sdss)**2 + (VR_met_arr - VR_sdss)**2
best_idx_met = np.argmin(dist2_met)
best_logZ = logZ_values[best_idx_met]
#print best loz and best age
print(f"Best-fit age: {best_age:.2f} Gyr")
print(f"Best-fit log(Z/Z_sun): {best_logZ:.2f}")
