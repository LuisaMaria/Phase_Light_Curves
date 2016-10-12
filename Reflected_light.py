import math
import numpy
import matplotlib.pyplot as plt
import sys
import scipy.optimize

#In this program we are going to consider the same geometry of the SOAP-T tool: 
#the x axis is along the line of sight of the observer, the yz plane is the plane of the sky 

def reflected_light(parameters, t, F, extras):
	planet_radius = parameters[0] 
	planet_period = parameters[1]
	I = parameters[2]
	a = parameters[3]
	e = parameters[4]
	t_p = parameters[5] 
	w = parameters[6]
	A_g = parameters[7]
	T = extras[0]
	stellar_period_rotation = extras[1]
	
	#t = numpy.zeros(len(phase))
	#t = (phase + T) * stellar_period_rotation
	
	#Then we calculate the orbital phase associated to each value
	phi = numpy.zeros(len(t))
	phi = ((t - t_p)/planet_period) - numpy.trunc((t - t_p)/planet_period)
	i = 0
	while i < len(t):
		if phi[i] < 0:
			phi[i] = 1 + phi[i]
		i = i + 1

	#Mean anomaly
	M = numpy.zeros(len(phi))
	i = 0
	while i < len(phi):
		M[i] = 2 * math.pi * phi[i]
		i = i + 1
	#print(phi, M)

	#Eccentric anomaly
	#I need a function for the relation between the mean anomaly and the eccentric anomaly
	def eccentric_function(x, p):
		return (x - e * numpy.sin(x) - p)

	#Then a function for the derivative of the previous one
	def der_eccentric_function(x,p):
		return (1 - e * numpy.cos(x))

	#Now I use the Newton-Raphson tool for Python to get the eccentric anomaly. This numerical method is fine only for low eccentricities
	E = numpy.zeros(len(phi))
	E2 = numpy.zeros(len(phi))
	i = 0
	while i < len(phi):
		x0 = M[i] 
		p = M[i]
		E[i] = scipy.optimize.newton(eccentric_function, x0, args = (p,), fprime=der_eccentric_function, tol=1.0e-08, maxiter=200)
		i = i + 1
	
	#To conclude I calculate the true anomaly:
	theta = numpy.zeros(len(phi))
	theta = 2 * numpy.arctan(math.sqrt((1+e)/(1-e)) * numpy.tan(E/2.))

	#I now determine the phase angle
	alpha = numpy.zeros(len(phi))
	alpha = numpy.arccos(numpy.sin(theta + w) * math.sin(I))

	#The planet star distance is given by the quantity
	r = numpy.zeros(len(phi))
	r = (a * (1 - e * e))/(1 + e * numpy.cos(theta))

	#Then I estimate the planet/star reflected flux ratio using a Lambertian formalism
	F = numpy.zeros(len(phi))
	F = A_g * (planet_radius/r)**2 * (numpy.sin(alpha) + (math.pi - alpha) * numpy.cos(alpha)) / 2.
	
	return(F)
