from VSPEC.variable_star_model import FlareGenerator
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from pathlib import Path
from os import chdir


WORKING_DIRECTORY = Path(__file__).parent

chdir(WORKING_DIRECTORY)

Teff = 3300*u.K
Prot = 80*u.day
gen = FlareGenerator(Teff,Prot)
time = 10*u.hr

E = np.logspace(33,36,401)*u.erg
freq = (gen.powerlaw(E) * time).to(u.Unit('')).value
def get_flare(Es, freqs):
    f_previous = 1
    E_final = 0
    for e, f in zip(E,freq):
        if np.random.random() < f/f_previous:
            f_previous = f
            E_final = e
        else:
            break
    return E_final
N=100000
sim_E = []
for i in range(N):
    sim_E.append((get_flare(E,freq)/u.erg).value)
sim_E = np.array(sim_E)#*u.erg
# print(freq)
fig,ax =plt.subplots(1,1)
ax.plot(E,freq*N)
n, b, _ = ax.hist(sim_E,bins=E.value[::10],cumulative=-1)
bincenters = 0.5*(b[1:]+b[:-1])
ax.errorbar(bincenters,n,yerr = np.sqrt(n),alpha=0.1)
ax.set_xscale('log')
ax.set_xlabel('Log(E/erg)')
ax.set_ylabel(f'P(Flare with E$_f$ > E) in {time}')
fig.savefig('cumulative_flare_dist.png')


fig,ax = plt.subplots(1,1)
ax.hist(sim_E,bins=E.value[::20],density=True)
ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel('Log(E/erg)')
ax.set_ylabel(f'N')
ax.set_title(f'E>0 for {(sim_E>0).sum()/N*100:.2f} pct')
fig.savefig('flare_dist.png')

