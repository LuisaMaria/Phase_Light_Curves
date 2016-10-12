import math
import sys
import numpy
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
import emcee
import pyfits
from matplotlib.ticker import MaxNLocator
import george
from george import kernels

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
s_prior = scipy.stats.halfcauchy(loc = 0, scale = 1)
t1_prior = scipy.stats.reciprocal(a = 0.00001, b = 1.)
t2_prior = scipy.stats.reciprocal(a = 1., b = 50.)
t3_prior = scipy.stats.uniform(loc = 1., scale = 40.)
t4_prior = scipy.stats.reciprocal(a = 0.1, b = 10.)

def lnprior(parameters):	
	planet_radius, orbital_period, inclination, semimajor_axis, eccentricity, time_periastron, longitude_periastron, albedo, s, t1, t2, t3, t4 = parameters
	return R_p_prior.logpdf(planet_radius) + \
               planet_period_prior.logpdf(orbital_period) + \
               I_prior.logpdf(inclination) + \
               a_prior.logpdf(semimajor_axis) + \
               e_prior.logpdf(eccentricity) + \
               t_p_prior.logpdf(time_periastron) + \
               w_prior.logpdf(longitude_periastron) + \
               albedo_prior.logpdf(albedo) + \
               s_prior.logpdf(s) + \
               t1_prior.logpdf(t1) + \
               t2_prior.logpdf(t2) + \
               t3_prior.logpdf(t3) + \
               t4_prior.logpdf(t4) 

def lnlike(parameters, time, flux, error):
	planet_radius, orbital_period, inclination, semimajor_axis, eccentricity, time_periastron, longitude_periastron, albedo, s, t1, t2, t3, t4 = parameters
	Fmod = numpy.zeros(len(time))
 	Fmod = reflected_light(parameters[:8], time, Fmod, extras)
	print(parameters[7], t3)
	p1 = t1**2
	p2 = t2**2
	p3 = t3
	p4 = 2/(t4)**2
	kernel = p1 * kernels.Product(kernels.ExpSquaredKernel(p2), kernels.ExpSine2Kernel(p4, p3))
	gp = george.GP(kernel, mean=1.0, solver=george.HODLRSolver)
	yerr = error + s
    	gp.compute(time, yerr)
	#print(gp.lnlikelihood(flux - Fmod))
    	return gp.lnlikelihood(flux - Fmod)
	
def lnprob(parameters, time, flux, error):
	lp = lnprior(parameters)
	if not numpy.isfinite(lp):
		return -numpy.inf
	#print(lp)
	return lp + lnlike(parameters, time, flux, error) if numpy.isfinite(lp) else -numpy.inf

ndim, nwalkers = 13, 50
#if 0.0 < albedo < 1.0 and 0 <= eccentricity <1 and planet_radius >= 0.0001 and t_p > 0 and a > 0:
pos = [[R_p_prior.rvs(), planet_period_prior.rvs(), I_prior.rvs(), a_prior.rvs(), e_prior.rvs(), t_p_prior.rvs(), w_prior.rvs(), albedo_prior.rvs(), s_prior.rvs(), t1_prior.rvs(), t2_prior.rvs(), t3_prior.rvs(), t4_prior.rvs()] for i in range(nwalkers)]

#posnew = pos[numpy.argmax(lnprob)]
#print(posnew)
#sampler.reset()
#steps = [1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 0.1, 0.001, 1, 1, 1, 0.1]
#posnew = [posnew + steps * numpy.random.randn(ndim) for i in xrange(nwalkers1)]
state = numpy.random.get_state()

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(time, flux, error))
posnew, prob, state = sampler.run_mcmc(pos, 10000, rstate0 = state)

#sys.exit(0)
#pos, prob, state = sampler.run_mcmc(posnew, 100, rstate0 = state)

#print('pos = ', pos, 'prob = ', prob, 'state = ', state)
real = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*numpy.percentile(sampler.flatchain, [16, 50, 84], axis=0)))
parametersnew = [real[0][0], real[1][0], real[2][0], real[3][0], real[4][0], real[5][0], real[6][0], real[7][0], real[8][0], real[9][0], real[10][0], real[11][0], real[12][0]]
print(parametersnew)
chains = numpy.asarray(sampler.flatchain)


Fmod_new = numpy.zeros(len(phase))
Fmod_new = reflected_light(parametersnew[:8], time, Fmod_new, extras) 
p1 = parametersnew[9]**2
p2 = parametersnew[10]**2
p3 = parametersnew[11]
p4 = 2/(parametersnew[12])**2
kernel = p1 * kernels.Product(kernels.ExpSquaredKernel(p2), kernels.ExpSine2Kernel(p4, p3)) 
gp = george.GP(kernel, mean=1.0, solver=george.HODLRSolver)
yerr = error + parametersnew[8]
gp.compute(time, yerr)
m = gp.sample_conditional(flux - Fmod_new, time) + Fmod_new

plt.rc('xtick', labelsize = 15)
plt.rc('ytick', labelsize = 15)
plt.figure (figsize = (15,7))
plt.xlabel('Time', fontsize = 30, color = 'black') 
plt.ylabel('Flux', fontsize = 30, color = 'black')
#plt.ylim(0.9997, 1.0001)
plt.plot(time, flux, color = 'k', label = 'flux')
plt.plot(time, m, color = 'red',label='Fit')
plt.legend(title = '')
plt.ion()
plt.show(block=True)


import corner
#make the corner plot
fig = corner.corner(sampler.flatchain[:,:], labels=["$planet_radius$", "$orbital period$", "$inclination$", "$semimajor axis$", "$eccentricity$", "$time periastron$", "$long periastron$", "$albedo$", "s", "$t_1$", "$t_2$", "$t_3$", "$t_4$"], truths = parametersnew)

fig.savefig(filename + "3corner.png")

plt.clf()
fig, axes = plt.subplots(13, 1, sharex=True, figsize=(8, 15))
axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
axes[0].yaxis.set_major_locator(MaxNLocator(5))
axes[0].axhline(R_p, color="#888888", lw=2)
axes[0].set_ylabel("$planet radius$")

axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
axes[1].yaxis.set_major_locator(MaxNLocator(5))
axes[1].axhline(planet_period, color="#888888", lw=2)
axes[1].set_ylabel("$orbital period$")

axes[2].plot(sampler.chain[:, :, 2].T, color="k", alpha=0.4)
axes[2].yaxis.set_major_locator(MaxNLocator(5))
axes[2].axhline(I, color="#888888", lw=2)
axes[2].set_ylabel("$inclination$")

axes[3].plot(sampler.chain[:, :, 3].T, color="k", alpha=0.4)
axes[3].yaxis.set_major_locator(MaxNLocator(5))
axes[3].axhline(a, color="#888888", lw=2)
axes[3].set_ylabel("$semimajor axis$")

axes[4].plot(sampler.chain[:, :, 4].T, color="k", alpha=0.4)
axes[4].yaxis.set_major_locator(MaxNLocator(5))
axes[4].axhline(e, color="#888888", lw=2)
axes[4].set_ylabel("$eccentricity$")

axes[5].plot(sampler.chain[:, :, 5].T, color="k", alpha=0.4)
axes[5].yaxis.set_major_locator(MaxNLocator(5))
axes[5].axhline(t_p, color="#888888", lw=2)
axes[5].set_ylabel("$time periastron$")

axes[6].plot(sampler.chain[:, :, 6].T, color="k", alpha=0.4)
axes[6].yaxis.set_major_locator(MaxNLocator(5))
axes[6].axhline(w, color="#888888", lw=2)
axes[6].set_ylabel("$long periastron$")

axes[7].plot(sampler.chain[:, :, 7].T, color="k", alpha=0.4)
axes[7].yaxis.set_major_locator(MaxNLocator(5))
axes[7].axhline(float(header['geomalb']), color="#888888", lw=2)
axes[7].set_ylabel("$albedo$")

axes[8].plot(sampler.chain[:, :, 8].T, color="k", alpha=0.4)
axes[8].yaxis.set_major_locator(MaxNLocator(5))
axes[8].axhline(parametersnew[8], color="#888888", lw=2)
axes[8].set_ylabel("$s$")

axes[9].plot(sampler.chain[:, :, 9].T, color="k", alpha=0.4)
axes[9].yaxis.set_major_locator(MaxNLocator(5))
axes[9].axhline(parametersnew[9], color="#888888", lw=2)
axes[9].set_ylabel("$t_1$")

axes[10].plot(sampler.chain[:, :, 10].T, color="k", alpha=0.4)
axes[10].yaxis.set_major_locator(MaxNLocator(5))
axes[10].axhline(parametersnew[10], color="#888888", lw=2)
axes[10].set_ylabel("$t_2$")

axes[11].plot(sampler.chain[:, :, 11].T, color="k", alpha=0.4)
axes[11].yaxis.set_major_locator(MaxNLocator(5))
axes[11].axhline(parametersnew[11], color="#888888", lw=2)
axes[11].set_ylabel("$t_3$")

axes[12].plot(sampler.chain[:, :, 12].T, color="k", alpha=0.4)
axes[12].yaxis.set_major_locator(MaxNLocator(5))
axes[12].axhline(parametersnew[12], color="#888888", lw=2)
axes[12].set_ylabel("$t_4$")
axes[12].set_xlabel("Number steps")


fig.tight_layout(h_pad=0.0)
fig.savefig(filename + "3line-time.png")

numpy.savetxt('/data/lserrano/Phase_Light_Curves/parameters3.txt', real)
