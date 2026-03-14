import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#part a
df = pd.read_csv("HW5_full_hktony1.csv")
mass = df["lgm_tot_p50"]
sfr = df["sfr_tot_p50"]
dn4000 = df["d4000_n"]
mask_valid = (mass > 0) & (sfr > -99)
mass = mass[mask_valid]
sfr = sfr[mask_valid]
dn4000 = dn4000[mask_valid]
sf_mask = dn4000 < 1.5
mass_sf = mass[sf_mask]
sfr_sf = sfr[sf_mask]
mass_line = np.linspace(6,12,100)
t = 13.7  # cosmic time at z ~ 0
slope = 0.84 - 0.026*t
intercept = -(6.51 - 0.11*t)
sfr_line = slope*mass_line + intercept
#part b
goals = pd.read_csv("goals_table3_system_mass_sfr.csv") 
goals["SFR"] = goals["SFR"].astype(str).str.replace("[", "").str.replace("]", "")
goals["SFR"] = goals["SFR"].astype(float)
mass_goals = np.log10(goals["Mass"])
sfr_goals = np.log10(goals["SFR"])
#part c
stages = pd.read_csv("goals_table3_with_stierwalt_stages.csv")
stages["SFR"] = stages["SFR"].astype(str).str.replace("[", "", regex=False)
stages["SFR"] = stages["SFR"].str.replace("]", "", regex=False)
stages["SFR"] = pd.to_numeric(stages["SFR"], errors="coerce")
stages["Mass"] = pd.to_numeric(stages["Mass"], errors="coerce")
stages = stages[(stages["Mass"] > 0) & (stages["SFR"] > 0)]
stages["logMass"] = np.log10(stages["Mass"])
stages["logSFR"] = np.log10(stages["SFR"])
stage_colors = {
    "N": "green",
    "a": "gold",
    "b": "orange",
    "c": "red",
    "d": "purple"
}
#part d
sfr_ms = slope * stages["logMass"] + intercept
stages["delta_sfr"] = stages["logSFR"] - sfr_ms

#part a
plt.figure(figsize=(8,6))
plt.scatter(mass, sfr, s=1, color="gray", alpha=0.2)
plt.scatter(mass_sf, sfr_sf, s=2, color="blue", alpha=0.5)
#part b
"""plt.scatter(mass_goals, sfr_goals,
            color="orange", marker="*", s=80,
            label="GOALS (Howell et al. 2010)")"""
#part c
for stage, color in stage_colors.items():
    sub = stages[stages["MergerStage_Stierwalt2013"] == stage]
    if len(sub) == 0:
        continue
    
    plt.scatter(sub["logMass"], sub["logSFR"],
                color=color, marker="*", s=100,
                edgecolor="black",
                label=f"GOALS stage {stage}")
plt.plot(mass_line, sfr_line, color='red', linewidth=3, label="Speagle et al. 2014")
plt.xlabel(r'$\log(M_*/M_\odot)$')
plt.ylabel(r'$\log(\mathrm{SFR}\,[M_\odot/yr])$')
plt.title("Star Forming Main Sequence with Dn4000 Cut")
plt.legend()
plt.show()
# part d
plt.figure(figsize=(8,6))

for stage, color in stage_colors.items():
    sub = stages[stages["MergerStage_Stierwalt2013"] == stage]
    if len(sub) == 0:
        continue

    plt.hist(sub["delta_sfr"],
             bins=15,
             alpha=0.5,
             color=color,
             label=f"GOALS stage {stage}")

plt.xlabel(r'$\Delta \log(\mathrm{SFR})$')
plt.ylabel("Number of galaxies")
plt.title("SFR Offset from the Main Sequence by Merger Stage")
plt.legend()
plt.show()
