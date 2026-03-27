import numpy as np
from numpy import isfinite
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from scipy.stats import linregress
hdul= fits.open("xCOLDGASS_PubCat.fits") #how to open fits file
data=hdul[1].data #data
hoh= hdul[1].header #header
#Part 1
logmgas=data["LOGMH2"] # H2 mass in solar masses
mgas=10**logmgas # convert log mass to mass
logmstar=data["LOGMSTAR"] # stellar mass in solar masses
mstar=10**logmstar # convert log mass to mass
logmgasms=data["LOGMH2MS"] # H2 mass in main sequence solar masses
mgasms=10**logmgasms # convert log mass to mass
logratio=logmgas-logmstar # log ratio
ratio=mgas/mstar 
plt.figure()
plt.scatter(logmstar, logratio, label='Data Points')
plt.xlabel('log(MStar)')
plt.ylabel('MH2/Mstar')
plt.title('part 1')
plt.savefig('part1.png')
plt.show()
#Part 2 
logSFR= data["LOGSFR_BEST"] #log10 SFR [Msun/yr]
elogSFR= data["LOGSFR_ERR"] #error bars on log10 SFR [Msun/yr]
emgas= data["LOGMH2_ERR"] #error bars on log10 MH2 [Msun]
R50= data["R50KPC"]
# mask made from dectection from part 1.
det1 = (data["FLAG_CO"] == 1) & np.isfinite(logmgas) & np.isfinite(logmstar)
ok2 = (np.isfinite(logSFR) & np.isfinite(elogSFR) &
       np.isfinite(emgas) & np.isfinite(R50) & (R50 > 0))
m = det1 & ok2
A   = np.pi * R50[m]**2                         # kpc^2
x   = logmgas[m] - np.log10(A)                   # log10 Σ_gas  [Msun/kpc^2]
y   = logSFR[m] - np.log10(A)                   # log10 Σ_SFR  [Msun/yr/kpc^2]
xerr = emgas[m]                                  # dex (ignore area error)
yerr = elogSFR[m]                                # dex
slope, intercept, r_value, p_value, std_err = linregress(x, y)
x_fit = np.linspace(np.min(x), np.max(x), 100)
y_fit = intercept + slope * x_fit
plt.figure()
plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='.', ms=5, alpha=0.8, lw=0.5, capsize=0, label="data")
plt.plot(x_fit, y_fit, color='red', label=f'Best Fit: y={slope:.2f}x + {intercept:.2f}')
plt.xlabel("log10 Σ_gas [M$_\\odot$ kpc$^{-2}$]")
plt.ylabel("log10 Σ_SFR [M$_\\odot$ yr$^{-1}$ kpc$^{-2}$]")
plt.title("Kennicutt–Schmidt (xCOLD GASS, H$_2$)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('part2.png')
plt.show()
#part 3
mc = (data["FLAG_CO"] == 1) & np.isfinite(logmgas) & np.isfinite(logSFR) & np.isfinite(logmstar)
#τ in Gyr
log_tau_yr = (logmgas - logSFR)[mc]
tau_Gyr = 10**log_tau_yr / 1e9
sSFR_Gyr = 10**(logSFR[mc] - logmstar[mc]) * 1e9 # sSFR in Gyr^-1
#errors
elog_tau = np.sqrt(emgas[mc]**2 + elogSFR[mc]**2)
tau_Gyr_err = tau_Gyr * np.log(10.0) * elog_tau
plt.figure()
plt.loglog(tau_Gyr, sSFR_Gyr, '.', ms=6, alpha=0.85)
t_burst = 0.1  # Gyr = 100 Myr
tau_model = np.logspace(np.log10(np.min(tau_Gyr)), np.log10(np.max(tau_Gyr)), 300)
sSFR_model = (1.0 / tau_model) * (np.exp(-t_burst / tau_model) / (1.0 - np.exp(-t_burst / tau_model)))
plt.loglog(tau_model, sSFR_model, '-', lw=2, label='Closed-box model (t = 0.1 Gyr)')
plt.legend(frameon=False)
plt.xlabel('depletion time τ Gyr')
plt.ylabel(r'$\mathrm{sSFR}$ Gyr$^{-1}$')
plt.title('Model vs Observation')
plt.tight_layout()
plt.savefig('modelvsobservation.png')
plt.show()
