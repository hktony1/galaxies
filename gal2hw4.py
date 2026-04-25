from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
filename = "COSMOS2020_FARMER_R1_v2.2_p3.fits"
hdul = fits.open(filename, memmap=True)
hdul.info()
data = hdul[1].data
logM = data["lp_mass_best"]
logSFR = data["lp_SFR_best"]
#mask finite values
mask = (
    np.isfinite(logM) &
    np.isfinite(logSFR) 
)
#check to see how many got masked
print("Total objects:", len(logM))
print("Objects after light plotting mask:", np.sum(mask))
plt.figure(figsize=(8, 6))
plt.scatter(logM[mask], logSFR[mask], s=1, alpha=0.03)
plt.xlabel(r"$\log_{10}(M_\star/M_\odot)$")
plt.ylabel(r"$\log_{10}(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})$")
plt.title("COSMOS2020 FARMER SFR vs Stellar Mass ")
plt.xlim(2.5, 12.5)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
#part b
logM = data["lp_mass_best"]
logSFR = data["lp_SFR_best"]
z = data["lp_zBEST"]
#mask finite values and local redshift z < 0.1
mask_b = (
    np.isfinite(logM) &
    np.isfinite(logSFR) &
    np.isfinite(z) &
    (z < 0.1) &
    (z >= 0)
)
print("Number of z < 0.1 objects:", np.sum(mask_b))
plt.figure(figsize=(8, 6))
plt.scatter(logM[mask_b], logSFR[mask_b], s=1, alpha=0.3, label="COSMOS2020 FARMER, z < 0.1")
# Renzini & Peng 2015 local SFMS
# log SFR = 0.76 log M* - 7.64
mgrid = np.linspace(2.5, 12, 300)
sfr_rp15 = 0.76 * mgrid - 7.64
plt.plot(
    mgrid,
    sfr_rp15,
    linewidth=3,
    color="black",
    label="Renzini & Peng 2015 local SFMS"
)
plt.xlabel(r"$\log_{10}(M_\star/M_\odot)$")
plt.ylabel(r"$\log_{10}(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})$")
plt.title(r"COSMOS2020 FARMER SFR vs Stellar Mass for $z<0.1$")
plt.xlim(2.5, 12.5)
plt.legend()
plt.tight_layout()
plt.show()
#part c
logM = data["lp_mass_best"]
logSFR = data["lp_SFR_best"]
z = data["lp_zBEST"]
model_flag = data["MODEL_FLAG"]
flag_combined = data["FLAG_COMBINED"]
lp_type = data["lp_type"]
lp_NbFilt = data["lp_NbFilt"]
lp_chi2_best = data["lp_chi2_best"]
#part b mask
mask_b = (
    np.isfinite(logM) &
    np.isfinite(logSFR) &
    np.isfinite(z) &
    (z >= 0) &
    (z < 0.1)
)
#mask/quality cuts using header for part c
mask_c = (
    mask_b &
    (model_flag == 0) &
    (flag_combined == 0) &
    (lp_type == 0) &
    (lp_NbFilt >= 3) &
    (lp_chi2_best != -99)
)
print("Part B objects, z < 0.1:", np.sum(mask_b))
print("Part C objects after quality cuts:", np.sum(mask_c))
print("Fraction kept:", np.sum(mask_c) / np.sum(mask_b))
plt.figure(figsize=(8, 6))
plt.scatter(
    logM[mask_c],
    logSFR[mask_c],
    s=1,
    alpha=0.35,
    label=r"COSMOS2020 FARMER, $z<0.1$, quality cuts"
)
# Renzini & Peng 2015 local SFMS
mgrid = np.linspace(5, 12, 300)
sfr_rp15 = 0.76 * mgrid - 7.64
plt.plot(
    mgrid,
    sfr_rp15,
    linewidth=3,
    color="black",
    label="Renzini & Peng 2015 local SFMS"
)
plt.xlabel(r"$\log_{10}(M_\star/M_\odot)$")
plt.ylabel(r"$\log_{10}(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})$")
plt.title(r"COSMOS2020 FARMER SFR vs Stellar Mass $z<0.1$ with Header Quality Cuts")
plt.xlim(2.5, 12.5)
plt.legend()
plt.tight_layout()
plt.show()
#part d
logM = data["lp_mass_best"]
logSFR = data["lp_SFR_best"]
z = data["lp_zBEST"]
lp_NbFilt = data["lp_NbFilt"]
lp_chi2_best = data["lp_chi2_best"]
# Renzini & Peng 2015 local SFMS residual
sfr_rp15_at_data = 0.76 * logM - 7.64
delta_sfr = logSFR - sfr_rp15_at_data
# histograms for cuts
plt.figure(figsize=(8, 5))
plt.hist(logM[mask_c], bins=80)
plt.xlabel(r"$\log_{10}(M_\star/M_\odot)$")
plt.ylabel("Number")
plt.title("Stellar Mass Distribution after Header Quality Cuts")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(logSFR[mask_c], bins=80)
plt.xlabel(r"$\log_{10}(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})$")
plt.ylabel("Number")
plt.title("SFR Distribution after Header Quality Cuts")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(lp_NbFilt[mask_c], bins=50)
plt.xlabel("Number of filters used in SED fit")
plt.ylabel("Number")
plt.title("Number of Filters Used")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(lp_chi2_best[mask_c], bins=80, range=(0, 100))
plt.xlabel(r"$\chi^2_{\rm best}$")
plt.ylabel("Number")
plt.title("Best-fit SED Chi-square")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(z_width[mask_c], bins=80, range=(0, 0.5))
plt.xlabel(r"$z_{\rm upper,68} - z_{\rm lower,68}$")
plt.ylabel("Number")
plt.title("Photometric Redshift Uncertainty Width")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(delta_sfr[mask_c], bins=80, range=(-5, 5))
plt.xlabel(r"$\Delta \log \mathrm{SFR}$ from Renzini & Peng 2015")
plt.ylabel("Number")
plt.title("Offset from Local SFMS")
plt.tight_layout()
plt.show()
delta_sfr = logSFR - (0.76 * logM - 7.64)
z_width = data["lp_zPDF_u68"] - data["lp_zPDF_l68"]
delta_sfr > -2.0
delta_sfr < 1.0
mask_d = (
    mask_c &

    # Remove low-mass 
    (logM > 6.5) &

    # Remove extreme SFR outliers
    (logSFR > -8) &
    (logSFR < 1.5) &

    # Require many filters
    (lp_NbFilt >= 26) &

    # Remove poor SED fits
    np.isfinite(lp_chi2_best) &
    (lp_chi2_best >= 0) &
    (lp_chi2_best < 10) &

    # Require reasonably constrained photo-z
    np.isfinite(z_width) &
    (z_width < 0.2) &

    # Remove objects far from the expected local SFMS
    np.isfinite(delta_sfr) &
    (delta_sfr > -2.0) &
    (delta_sfr < 1.0)
)
print("Part C objects:", np.sum(mask_c))
print("Part D objects:", np.sum(mask_d))
print("Fraction kept:", np.sum(mask_d) / np.sum(mask_c))
plt.scatter(
    logM[mask_d],
    logSFR[mask_d],
    s=1,
    alpha=0.45,
    label=r"Cleaned $z<0.1$ sample"
)
mgrid = np.linspace(6.5, 12, 300)
sfr_rp15 = 0.76 * mgrid - 7.64
plt.plot(
    mgrid,
    sfr_rp15,
    linewidth=3,
    color="black",
    label="Renzini & Peng 2015 local SFMS"
)
plt.xlabel(r"$\log_{10}(M_\star/M_\odot)$")
plt.ylabel(r"$\log_{10}(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})$")
plt.title(r"COSMOS2020 FARMER SFMS, $z<0.1$")
plt.xlim(2.5, 12.5)
plt.legend()
plt.tight_layout()
plt.show()
#part e
logM = data["lp_mass_best"]
logSFR = data["lp_SFR_best"]
z = data["lp_zBEST"]
model_flag = data["MODEL_FLAG"]
flag_combined = data["FLAG_COMBINED"]
lp_type = data["lp_type"]
lp_NbFilt = data["lp_NbFilt"]
lp_chi2_best = data["lp_chi2_best"]
z_width = data["lp_zPDF_u68"] - data["lp_zPDF_l68"]
#mask without Renzini & Peng residual cut here.
mask_e = (
    np.isfinite(logM) &
    np.isfinite(logSFR) &
    np.isfinite(z) &
    np.isfinite(z_width) &
    (z >= 0) &
    (z < 2.5) &
    (model_flag == 0) &
    (flag_combined == 0) &
    (lp_type == 0) &
    (lp_NbFilt >= 26) &
    (lp_chi2_best >= 0) &
    (lp_chi2_best < 10) &
    (logM > 6.5) &
    (logM < 12.5) &
    (logSFR > -8) &
    (logSFR < 4) &
    ((z_width / (1 + z)) < 0.15)
)
print("Objects in Part E cleaned sample:", np.sum(mask_e))
# Redshift bins from local to cosmic noon
z_bins = [
    (0.0, 0.5),
    (0.5, 1.0),
    (1.0, 1.5),
    (1.5, 2.0),
    (2.0, 2.5),
]
colors = ["blue", "green", "orange", "red", "purple"]
plt.figure(figsize=(9, 7))
legend_handles = []
for (zmin, zmax), color in zip(z_bins, colors):
    mask_bin = mask_e & (z >= zmin) & (z < zmax)
    x = logM[mask_bin]
    y = logSFR[mask_bin]
    print(f"{zmin:.1f} < z < {zmax:.1f}: {len(x)} galaxies")
    # Make 2D histogram for contours
    H, xedges, yedges = np.histogram2d(
        x, y,
        bins=[70, 70],
        range=[[6, 12], [-4, 4]]
    )
    # Bin centers
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xcenters, ycenters)
    H = H.T  # transpose for plotting
    # Only draw contours if there are enough points
    """if np.max(H) > 0:
        levels = np.linspace(np.max(H)*0.15, np.max(H)*0.9, 5)
        plt.contour(
            X, Y, H,
            levels=levels,
            colors=color,
            linewidths=1.8
        )
        legend_handles.append(
            Line2D([0], [0], color=color, lw=2, label=rf"${zmin}<z<{zmax}$")
        )"""
    # Median trend in stellar-mass bins
    mass_bins = np.arange(6.0, 12.1, 0.25)
    mass_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])
    med_mass = []
    med_sfr = []
    for lo, hi, cen in zip(mass_bins[:-1], mass_bins[1:], mass_centers):
        in_mass_bin = (x >= lo) & (x < hi)
        if np.sum(in_mass_bin) >= 30:
            med_mass.append(cen)
            med_sfr.append(np.median(y[in_mass_bin]))
    plt.plot(
        med_mass,
        med_sfr,
        color=color,
        linewidth=2.5,
        label=rf"${zmin}<z<{zmax}$"
    )

plt.xlabel(r"$\log_{10}(M_\star/M_\odot)$")
plt.ylabel(r"$\log_{10}(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})$")
plt.title("COSMOS2020 FARMER SFMS Evolution with Redshift")
#plt.legend(handles=legend_handles, title="Redshift bins")
plt.legend(title="Redshift bins")
plt.tight_layout()
plt.show()
