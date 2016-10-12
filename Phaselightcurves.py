import ConfigParser
import os
import sys
import math
import numpy
import scipy
import shutil
import string
import matplotlib.pyplot as plt
import pyfits 
import subprocess

# create a file called phases.txt in /data/lserrano/SOAP_2_T+R/SOAP_2_T+R/
phases = numpy.linspace(0, 4, 1000)
numpy.savetxt('/data/lserrano/SOAP_2_T+R/SOAP_2_T+R/phases.txt', phases)

cwd = os.getcwd()
os.chdir('/data/lserrano/SOAP_2_T+R/SOAP_2_T+R/')

execfile('/data/lserrano/SOAP_2_T+R/SOAP_2_T+R/soap2_t+r.py')

os.chdir(cwd)


config = ConfigParser.ConfigParser()
config.read("/home/lserrano/data/SOAP_2_T+R/SOAP_2_T+R/config.cfg")

R_s = int(config.get('star','radius_sun'))

planet_radius = 0.1 * R_s  
phi1 = 0.00
phi2 = 1
d = float(config.get('output','ph_step'))
planet_period = float(config.get('planet','Pp'))      #orbital period
I = math.pi * float(config.get('planet','ip'))/180.   #orbital inclination  
a = float(config.get('planet','a')) * R_s             #semimajor axis
e = float(config.get('planet','e'))                   #eccentricity
t_p = float(config.get('planet','t0'))                #periastron time
w = math.pi * float(config.get('planet','w'))/180.    #periastron argument
A_g = 0.6                                             #geometric albedo
T = float(config.get('planet','t'))		      #difference between orbital and stellar phase
stellar_period_rotation = float(config.get('star','prot' ))
stellar_inclination = float(config.get('star','I'))

active_region1_long = float(config.get('active_regions','long1'))
active_region1_lat  = float(config.get('active_regions','lat1'))
active_region1_size = float(config.get('active_regions','size1'))

active_region2_long = float(config.get('active_regions','long2'))
active_region2_lat  = float(config.get('active_regions','lat2'))
active_region2_size = float(config.get('active_regions','size2'))
                                     
active_region3_long = float(config.get('active_regions','long3'))
active_region3_lat  = float(config.get('active_regions','lat3'))
active_region3_size = float(config.get('active_regions','size3'))
                                      
active_region4_long = float(config.get('active_regions','long4'))
active_region4_lat  = float(config.get('active_regions','lat4'))
active_region4_size = float(config.get('active_regions','size4'))

active_region5_long = float(config.get('active_regions','long5'))
active_region5_lat  = float(config.get('active_regions','lat5'))
active_region5_size = float(config.get('active_regions','size5'))

active_region6_long = float(config.get('active_regions','long6'))
active_region6_lat  = float(config.get('active_regions','lat6'))
active_region6_size = float(config.get('active_regions','size6'))

active_region7_long = float(config.get('active_regions','long7'))
active_region7_lat  = float(config.get('active_regions','lat7'))
active_region7_size = float(config.get('active_regions','size7'))

active_region8_long = float(config.get('active_regions','long8'))
active_region8_lat  = float(config.get('active_regions','lat8'))
active_region8_size = float(config.get('active_regions','size8'))

active_region9_long = float(config.get('active_regions','long9'))
active_region9_lat  = float(config.get('active_regions','lat9'))
active_region9_size = float(config.get('active_regions','size9'))

active_region10_long = float(config.get('active_regions','long10'))
active_region10_lat  = float(config.get('active_regions','lat10'))
active_region10_size = float(config.get('active_regions','size10'))                        

parameters = numpy.zeros(8)
parameters[0]  = planet_radius
parameters[1]  = planet_period
parameters[2]  = I
parameters[3]  = a
parameters[4]  = e
parameters[5]  = t_p
parameters[6]  = w
parameters[7] = A_g
extras = numpy.zeros(2)
extras[0] = T
extras[1] = stellar_period_rotation
 
CCF_folder_outputs = '/home/lserrano/data/SOAP_2_T+R/SOAP_2_T+R/outputs/CCF_PROT=%.2f_i=%.2f_lon=(%.1f,%.1f,%.1f,%.1f)_lat=(%.1f,%.1f,%.1f,%.1f)_size=(%.4f,%.4f,%.4f,%.4f)/' % (stellar_period_rotation,stellar_inclination,active_region1_long,active_region2_long,active_region3_long,active_region4_long,active_region1_lat,active_region2_lat,active_region3_lat,active_region4_lat,active_region1_size,active_region2_size,active_region3_size,active_region4_size)

filename = CCF_folder_outputs + "CCF_PROT=%.2f_i=%.2f_lon=(%.1f,%.1f,%.1f,%.1f)_lat=(%.1f,%.1f,%.1f,%.1f)_size=(%.4f,%.4f,%.4f,%.4f)" % (stellar_period_rotation,stellar_inclination,active_region1_long,active_region2_long,active_region3_long,active_region4_long,active_region1_lat,active_region2_lat,active_region3_lat,active_region4_lat,active_region1_size,active_region2_size,active_region3_size,active_region4_size)

def read_rdb(filename):
    
    f = open(filename, 'r')
    data = f.readlines()
    f.close()
    
    z=0
    while data[z][:2] == '# ' or data[z][:2] == ' #':
        z += 1

    key = string.split(data[z+0][:-1],'\t')
    output = {}
    for i in range(len(key)): output[key[i]] = []
    
    for line in data[z+2:]:
        qq = string.split(line[:-1],'\t')
        for i in range(len(key)):
            try: value = float(qq[i])
            except ValueError: value = qq[i]
            output[key[i]].append(value)

    return output, key

document = filename + '.rdb'
output, key = read_rdb(document)

phase = numpy.asarray(output[key[0]])
flux = numpy.asarray(output[key[1]])
time = numpy.zeros(len(phase))
time = (phase + T) * stellar_period_rotation

from Reflected_light import reflected_light
reflected_flux = numpy.zeros(len(time))
reflected_flux = reflected_light(parameters, time, reflected_flux, extras)
s = numpy.random.normal(0, 0.00011, len(phase))

total_flux = flux + reflected_flux + s
#total_flux = reflected_flux + s

plt.rc('xtick', labelsize = 15)
plt.rc('ytick', labelsize = 15)
plt.figure (figsize = (15,7))
plt.xlabel('Stellar phase', fontsize = 30, color = 'black') 
plt.ylabel('Flux ratio', fontsize = 30, color = 'black')
plt.plot(phase, total_flux, color = 'k', label = 'flux', linestyle = 'none', marker = '.')
#plt.plot(phase, flux, color = 'red', label = 'flux', linestyle = 'none', marker = '.')
#plt.plot(phase, reflected_flux, color = 'blue', label = 'flux', linestyle = 'none', marker = '.')
plt.legend(title = '')
plt.ion()
plt.show(block=True)

name = 'Prot=' + str(stellar_period_rotation) + ',ActLev = 1%,Prad=' + str(planet_radius) + ',Porb=' + str(planet_period)

filename = name + '.fits'
if os.path.exists(filename):
	os.remove(filename)
        
d = [0.]*4
d[0] = pyfits.Column(name = 'Time (days)', format='E', array = time)
d[1] = pyfits.Column(name = 'Stellar_phase', format='E', array = phase)
d[2] = pyfits.Column(name = 'Flux', format='E', array = total_flux)
d[3] = pyfits.Column(name = 'Instrumental noise', format='E', array = s)
print(d)
curve = pyfits.TableHDU.from_columns(d)
curve.header['OrbitPer'] = str(planet_period)
curve.header['OrbitInc'] = str(I)
curve.header['SemimAx'] = str(a)
curve.header['Eccentr'] = str(e)
curve.header['Timeper'] = str(t_p)
curve.header['Argperi'] = str(w)
curve.header['PlanRad'] = str(planet_radius)
curve.header['GeomAlb'] = str(A_g)
curve.header['StelPer'] = str(stellar_period_rotation) 
curve.header['Lon1spot'] = str(active_region1_long)
curve.header['Lat1spot'] = str(active_region1_lat)
curve.header['Siz1spot'] = str(active_region1_size)
curve.header['Lon2spot'] = str(active_region2_long)
curve.header['Lat2spot'] = str(active_region2_lat)
curve.header['Siz2spot'] = str(active_region2_size)
curve.header['Lon3spot'] = str(active_region3_long)
curve.header['Lat3spot'] = str(active_region3_lat)
curve.header['Siz3spot'] = str(active_region3_size)
curve.header['Lon4spot'] = str(active_region4_long)
curve.header['Lat4spot'] = str(active_region4_lat)
curve.header['Siz4spot'] = str(active_region4_size)
curve.header['Lon5spot'] = str(active_region5_long)
curve.header['Lat5spot'] = str(active_region5_lat)
curve.header['Siz5spot'] = str(active_region5_size)
curve.header['Lon6spot'] = str(active_region6_long)
curve.header['Lat6spot'] = str(active_region6_lat)
curve.header['Siz6spot'] = str(active_region6_size)
curve.header['Lon7spot'] = str(active_region7_long)
curve.header['Lat7spot'] = str(active_region7_lat)
curve.header['Siz7spot'] = str(active_region7_size)
curve.header['Lon8spot'] = str(active_region8_long)
curve.header['Lat8spot'] = str(active_region8_lat)
curve.header['Siz8spot'] = str(active_region8_size)
curve.header['Lon9spot'] = str(active_region9_long)
curve.header['Lat9spot'] = str(active_region9_lat)
curve.header['Siz9spot'] = str(active_region9_size)
curve.header['LonXspot'] = str(active_region10_long)
curve.header['LatXspot'] = str(active_region10_lat)
curve.header['SizXspot'] = str(active_region10_size)

curve.writeto(filename)

tabl = pyfits.open(filename)
header_primary = tabl[1].header
print(header_primary)
