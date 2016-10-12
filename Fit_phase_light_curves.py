import math
import sys
import numpy
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
import emcee
import pyfits
from matplotlib.ticker import MaxNLocator

filename = 'Prot=9.0,ActLev = 1%,Prad=69600.0,Porb=3.0'
name = filename + '.fits'

tabl = pyfits.open(name)
header = tabl[1].header
data = tabl[1].data

time = data.field(0)
phase = data.field(1)
flux = data.field(2)
error = data.field(3)

R_p = float(header['PLANRAD'])
a = float(header['SemimAx'])
e = float(header['Eccentr'])
t_p = float(header['Timeper'])
I = float(header['OrbitInc'])
w = float(header['Argperi'])
planet_period = float(header['OrbitPer'])
stellar_period_rotation = float(header['StelPer'])
extras = numpy.zeros(2)
extras[0] = 0
extras[1] = stellar_period_rotation

from Reflected_light import reflected_light

R_p_prior = scipy.stats.norm(loc = R_p, scale = 0.001)
a_prior = scipy.stats.norm(loc = a, scale = 0.001)
e_prior = scipy.stats.halfnorm(loc = e, scale = 0.001)  
t_p_prior = scipy.stats.norm(loc = t_p, scale = 0.001)
I_prior = scipy.stats.norm(loc = I, scale = 0.001)
w_prior = scipy.stats.norm(loc = w, scale = 0.001)
planet_period_prior = scipy.stats.norm(loc = planet_period, scale = 0.001)
albedo_prior = scipy.stats.uniform(loc = 0.0, scale = 1.)

def lnprior(parameters):
	planet_radius, orbital_period, inclination, semimajor_axis, eccentricity, time_periastron, longitude_periastron, albedo = parameters
	
	return R_p_prior.logpdf(planet_radius) + planet_period_prior.logpdf(orbital_period) + I_prior.logpdf(inclination) + a_prior.logpdf(semimajor_axis) + e_prior.logpdf(eccentricity) + t_p_prior.logpdf(time_periastron) + w_prior.logpdf(longitude_periastron) + albedo_prior.logpdf(albedo)
	

def lnlike(parameters, phase, flux, error):
	Fmod = numpy.zeros(len(phase))
 	Fmod = reflected_light(parameters, phase, Fmod, extras)
	inv_sigma2 = 1.0/(error**2)
	return -0.5*(numpy.sum((flux-Fmod)**2 * inv_sigma2 - numpy.log((2*math.pi)**(-1) * inv_sigma2)))

def lnprob(parameters, phase, flux, error):
	lp = lnprior(parameters)
	if not numpy.isfinite(lp):
		return -numpy.inf
	#print theta0, lp, lnlike(theta0, t, flux, fluxerr)
	return lp + lnlike(parameters, phase, flux, error)

ndim, nwalkers = 8, 50

#if 0.0 < albedo < 1.0 and 0 <= eccentricity <1 and planet_radius >= 0.0001 and t_p > 0 and a > 0:
pos = [[R_p_prior.rvs(), planet_period_prior.rvs(), I_prior.rvs(), a_prior.rvs(), e_prior.rvs(), t_p_prior.rvs(), w_prior.rvs(), albedo_prior.rvs()] for i in range(nwalkers)]
print(pos)

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(phase, flux, error))

state = numpy.random.get_state()
pos, prob, state = sampler.run_mcmc(pos, 100, rstate0 = state)

#print('pos = ', pos, 'prob = ', prob, 'state = ', state)
real = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*numpy.percentile(sampler.flatchain, [16, 50, 84], axis=0)))
parametersnew = [real[0][0], real[1][0], real[2][0], real[3][0], real[4][0], real[5][0], real[6][0], real[7][0]]
print(parametersnew)
chains = numpy.asarray(sampler.flatchain)

Fmod_new = numpy.zeros(len(phase))
Fmod_new = reflected_light(parametersnew, phase, Fmod_new, extras)
plt.rc('xtick', labelsize = 15)
plt.rc('ytick', labelsize = 15)
plt.figure (figsize = (15,7))
plt.xlabel('Time', fontsize = 30, color = 'black') 
plt.ylabel('Flux', fontsize = 30, color = 'black')
#plt.ylim(0.9997, 1.0001)
plt.plot(phase, flux, color = 'k', label = 'flux')
plt.plot(phase, Fmod_new, color = 'red',label='Fit')
plt.legend(title = '')
plt.ion()
plt.show(block=False)

import corner
#make the corner plot
fig = corner.corner(sampler.flatchain[:,:], labels=["$planet_radius$", "$orbital_period$", "$inclination$", "$semimajor_axis$", "$eccentricity$", "$time_periastron$", "$long_periastron$", "$albedo$"], truths = parametersnew)

fig.savefig("corner.png")

plt.clf()
fig, axes = plt.subplots(8, 1, sharex=True, figsize=(8, 9))
axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
axes[0].yaxis.set_major_locator(MaxNLocator(5))
axes[0].axhline(R_p, color="#888888", lw=2)
axes[0].set_ylabel("$planet_radius$")

axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
axes[1].yaxis.set_major_locator(MaxNLocator(5))
axes[1].axhline(planet_period, color="#888888", lw=2)
axes[1].set_ylabel("$orbital_period$")

axes[2].plot(sampler.chain[:, :, 2].T, color="k", alpha=0.4)
axes[2].yaxis.set_major_locator(MaxNLocator(5))
axes[2].axhline(I, color="#888888", lw=2)
axes[2].set_ylabel("$inclination$")

axes[3].plot(sampler.chain[:, :, 3].T, color="k", alpha=0.4)
axes[3].yaxis.set_major_locator(MaxNLocator(5))
axes[3].axhline(a, color="#888888", lw=2)
axes[3].set_ylabel("$semimajor_axis$")

axes[4].plot(sampler.chain[:, :, 4].T, color="k", alpha=0.4)
axes[4].yaxis.set_major_locator(MaxNLocator(5))
axes[4].axhline(e, color="#888888", lw=2)
axes[4].set_ylabel("$eccentricity$")

axes[5].plot(sampler.chain[:, :, 5].T, color="k", alpha=0.4)
axes[5].yaxis.set_major_locator(MaxNLocator(5))
axes[5].axhline(t_p, color="#888888", lw=2)
axes[5].set_ylabel("$time_periastron$")

axes[6].plot(sampler.chain[:, :, 6].T, color="k", alpha=0.4)
axes[6].yaxis.set_major_locator(MaxNLocator(5))
axes[6].axhline(w, color="#888888", lw=2)
axes[6].set_ylabel("$long_periastron$")

axes[7].plot(sampler.chain[:, :, 7].T, color="k", alpha=0.4)
axes[7].yaxis.set_major_locator(MaxNLocator(5))
axes[7].axhline(float(header['geomalb']), color="#888888", lw=2)
axes[7].set_ylabel("$albedo$")
axes[7].set_xlabel("Number_steps")

fig.tight_layout(h_pad=0.0)
fig.savefig("line-time.png")

