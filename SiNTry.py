# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:51:56 2021

@author: KGB
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
#plt.style.use('default')
#plt.rcParams.update({'font.size': 18})

def Bfunc(x, Bstart, Bend, indStart, indEnd):
    numer = (Bend - Bstart)*(x - indStart)/(indEnd - indStart) + Bstart
    return numer

def BfuncR(x, Bstart, indStart, rate):
    numer = (x - indStart)*rate + Bstart
    return numer

def makeBs(index, BList, Bstart, Bend, indStart, indEnd):
    loc0 = np.where(index == indStart)[0][0]
    if(indEnd <= np.amax(index)):
        loc1 = np.where(index == (indEnd))[0][0]
    else:
        loc1 = np.where(index == np.amax(index))[0][0]
    indSec = index[loc0:loc1]
    BList[loc0:loc1] = Bfunc(indSec, Bstart, Bend, indStart, indEnd)
    return BList

def makethetas(index, TListx, thetastart, thetaend, indStart, indEnd):
    
    loc0 = np.where(index == indStart)[0][0]
    if(indEnd <= np.amax(index)):
        loc1 = np.where(index == (indEnd+1))[0][0]
    else:
        loc1 = np.where(index == np.amax(index))[0][0]
    indSec = index[loc0:loc1]
    TList[loc0:loc1] = Bfunc(indSec, thetastart, thetaend, indStart, indEnd)

    return TList

def cutCrash(index, BList, Bstart, indStart, rate):
    loc0 = np.where(index == indStart)[0][0]
    indSec = index[loc0:]
    BList[loc0:] = BfuncR(indSec, Bstart, indStart, rate)
    return BList

def maskOff(mask, index, BList, f0, dc):
    return index[mask], BList[mask], f0[mask], dc[mask]

def prefilter(index, f0, v, Q, dc):
    keep = (f0 > 1000)
    index = index[keep]
    f0 = f0[keep]
    v = v[keep]
    dc = dc[keep]
    Q = Q[keep]
    return index, f0, v, Q, dc

startcut = 0
endcut =-1

pt = '/home/sam/Documents/InPlane/6_15_21_inplane1.txt'
ptl = '/home/sam/Documents/InPlane/6_15_21_inplane1L.txt'

F = np.loadtxt(pt, delimiter="\t")
FL = np.loadtxt(ptl, delimiter="\t")
index = np.array(F[startcut:endcut, 3])
f0 = np.array(F[startcut:endcut, 2])
v = np.array(F[startcut:endcut, 4])
Q = np.array(np.abs(FL[startcut:endcut, 3]))
dc = np.array(F[startcut:endcut, 0])

zerof0 = np.mean([f0[0:20]])
print('Zero field f0 at beginning of file (Hz)')
print(zerof0)
print('Zero field f0 at end of file (Hz)')
zerofend = np.mean([f0[1556:-1]])
print(zerofend)
print('Difference between beginning and end zero field f0 (fend - fbeginning in mHz)')
print(1000*(zerofend-zerof0))

print('Difference between 0 and 360 degrees (mHz)')
print(1000*(np.mean(f0[90:112]) - np.mean(f0[1471:1493])))

index, f0, v, Q, dc =  prefilter(index, f0, v, Q, dc)

TList = -1000*np.ones(len(index))

TList = makethetas(index, TList, 0, 0, 90, 112)
TList = makethetas(index, TList, 10, 10, 128, 150)
TList = makethetas(index, TList, 20, 20, 160, 182)
TList = makethetas(index, TList, 30, 30, 190, 212)
TList = makethetas(index, TList, 40, 40, 219, 241)
TList = makethetas(index, TList, 50, 50, 246, 268)
TList = makethetas(index, TList, 60, 60, 276, 298)
TList = makethetas(index, TList, 70, 70, 306, 328)
TList = makethetas(index, TList, 80, 80, 339, 360)
TList = makethetas(index, TList, 90, 90, 371, 428)
TList = makethetas(index, TList, 100, 100, 455, 476)
TList = makethetas(index, TList, 110, 110, 488, 509)
TList = makethetas(index, TList, 120, 120, 516, 538)
TList = makethetas(index, TList, 130, 130, 545, 567)
TList = makethetas(index, TList, 140, 140, 583, 605)
TList = makethetas(index, TList, 150, 150, 616, 638)
TList = makethetas(index, TList, 160, 160, 645, 667)
TList = makethetas(index, TList, 170, 170, 675, 697)
TList = makethetas(index, TList, 180, 180, 711, 733)
TList = makethetas(index, TList, 190, 190, 756, 778)
TList = makethetas(index, TList, 200, 200, 799, 821)
TList = makethetas(index, TList, 210, 210, 830, 852)
TList = makethetas(index, TList, 220, 220, 859, 874)
TList = makethetas(index, TList, 230, 230, 1015, 1037)
TList = makethetas(index, TList, 240, 240, 1060, 1091)
TList = makethetas(index, TList, 250, 250, 1100, 1122)
TList = makethetas(index, TList, 260, 260, 1134, 1156)
TList = makethetas(index, TList, 270, 270, 1164, 1186)
TList = makethetas(index, TList, 280, 280, 1202, 1224)
TList = makethetas(index, TList, 290, 290, 1241, 1261)
TList = makethetas(index, TList, 300, 300, 1270, 1290)
TList = makethetas(index, TList, 310, 310, 1299, 1320)
TList = makethetas(index, TList, 320, 320, 1328, 1350)
TList = makethetas(index, TList, 330, 330, 1366, 1388)
TList = makethetas(index, TList, 340, 340, 1397, 1419)
TList = makethetas(index, TList, 350, 350, 1435, 1457)
TList = makethetas(index, TList, 360, 360, 1471, 1493)


def FF(x,a,b,c,d, e, f, g):
    th = x[0, :]
    rs = x[1, :]
    return a + b*rs + c*np.cos(np.pi*(th-f)/180)**2 + d*np.sin(2*np.pi*(th-f)/180) + e*np.sin(2*np.pi*(th-f)/180)**2 + g*rs**2

mask = (TList > -999)
index, TList, f0, dc = maskOff(mask, index, TList, f0, dc)

TList = TList - 16

data = np.vstack((TList, index))
popt, pcov = curve_fit(FF, data, f0, p0 = [np.mean(f0), 6E-6, .6, -4, 0, 0, 0])
print(popt)
ftry = FF(data, popt[0], 0, popt[2], popt[3], 0, popt[5], 0)
offy = FF(data, 0, popt[1], 0, 0, 0, 0, popt[6])
offy2 = FF(data, 0, 0, 0, 0, popt[4], popt[5], 0)
print(180*popt[2]/np.pi)

plt.scatter(TList, f0-offy-zerof0, s=10, linewidth = .5,  zorder=1, marker = '.', edgecolors = 'k', color = 'black')
plt.plot(TList, ftry + offy2-zerof0)
plt.grid()
plt.xlabel(r'$\phi$ ($^{\circ}$)')
plt.ylabel(r'$f_{0} - f_{0}$(B=0) (Hz)')
plt.show()


#plt.scatter(TList, f0-zerof0-offy, s=10, linewidth = .5,  zorder=1, marker = '.', edgecolors = 'k', color = 'black')
#plt.plot(TList, ftry-zerof0)
##plt.polar(TList, f0-zerof0-offy + np.amin(f0-zerof0-offy), 'b.')
##plt.polar(TList, ftry-zerof0, 'g.')
#plt.grid()
#plt.xlabel('Theta (degrees)')
#plt.ylabel('2 Theta Component (Hz)')
#plt.show()

#plt.scatter(TList, dc, s=10, linewidth = .5,  zorder=1, marker = '.', edgecolors = 'k', color = 'black')
##plt.polar(TList, f0-zerof0-offy + np.amin(f0-zerof0-offy), 'b.')
##plt.polar(TList, ftry-zerof0, 'g.')
#plt.grid()
#plt.xlabel('Theta (degrees)')
#plt.ylabel('DC Level (V)')
#plt.show()

plotter = f0-offy-zerof0
thLine = np.linspace(np.amin(plotter)-.05, np.amax(plotter))
plt.polar(TList*np.pi/180, plotter, 'b.')
plt.polar(TList*np.pi/180, [0]*len(TList), 'brown')
plt.polar(TList*np.pi/180, ftry + offy2-zerof0, 'g', zorder = 3)
plt.polar((283- 196)*np.ones(len(thLine))*np.pi/180, thLine, color = 'r')
plt.polar((283-180- 196)*np.ones(len(thLine))*np.pi/180, thLine, color = 'r')
plt.polar((53- 196)*np.ones(len(thLine))*np.pi/180, thLine, color = 'k')
plt.polar((53-180- 196)*np.ones(len(thLine))*np.pi/180, thLine, color = 'k')
plt.polar((143- 196)*np.ones(len(thLine))*np.pi/180, thLine, color = 'g')
plt.polar((143-180- 196)*np.ones(len(thLine))*np.pi/180, thLine, color = 'g')
plt.polar(-196*np.ones(len(thLine))*np.pi/180, thLine, color = 'b')
plt.polar((-196-180)*np.ones(len(thLine))*np.pi/180, thLine, color = 'b')
plt.ylim(np.amin(thLine), np.amax(thLine))
#plt.title('2 Theta Component vs Theta')
plt.show()

plotter = f0-zerof0-offy-offy2
thLine = np.linspace(np.amin(plotter)-.05, np.amax(plotter))
plt.polar(TList*np.pi/180, plotter, 'b.')
plt.polar(TList*np.pi/180, [0]*len(TList), 'brown')
plt.polar(TList*np.pi/180, ftry-zerof0, 'g', zorder = 3)
plt.polar((283- 196)*np.ones(len(thLine))*np.pi/180, thLine, color = 'r')
plt.polar((283-180- 196)*np.ones(len(thLine))*np.pi/180, thLine, color = 'r')
plt.polar((53- 196)*np.ones(len(thLine))*np.pi/180, thLine, color = 'k')
plt.polar((53-180- 196)*np.ones(len(thLine))*np.pi/180, thLine, color = 'k')
plt.polar((143- 196)*np.ones(len(thLine))*np.pi/180, thLine, color = 'g')
plt.polar((143-180- 196)*np.ones(len(thLine))*np.pi/180, thLine, color = 'g')
plt.polar(-196*np.ones(len(thLine))*np.pi/180, thLine, color = 'b')
plt.polar((-196-180)*np.ones(len(thLine))*np.pi/180, thLine, color = 'b')
plt.ylim(np.amin(thLine), np.amax(thLine))
#plt.title('2 Theta Component vs Theta')
plt.show()

plotter = dc - np.amin(dc)
thLine = np.linspace(np.amin(plotter), np.amax(plotter))
plt.polar(TList*np.pi/180, dc - np.amin(dc), 'r.', zorder = 3)
#plt.polar(TList, f0-zerof0-offy + np.amin(f0-zerof0-offy), 'b.')
#plt.polar(TList, ftry-zerof0, 'g.')
plt.polar((283- 196)*np.ones(len(thLine))*np.pi/180, thLine, color = 'r')
plt.polar((283-180- 196)*np.ones(len(thLine))*np.pi/180, thLine, color = 'r')
plt.polar((53- 196)*np.ones(len(thLine))*np.pi/180, thLine, color = 'k')
plt.polar((53-180- 196)*np.ones(len(thLine))*np.pi/180, thLine, color = 'k')
plt.polar((143- 196)*np.ones(len(thLine))*np.pi/180, thLine, color = 'g')
plt.polar((143-180- 196)*np.ones(len(thLine))*np.pi/180, thLine, color = 'g')
plt.polar(-196*np.ones(len(thLine))*np.pi/180, thLine, color = 'b')
plt.polar((-196-180)*np.ones(len(thLine))*np.pi/180, thLine, color = 'b')
plt.ylim(np.amin(thLine), np.amax(thLine))
plt.show()

#plt.scatter(TList, f0-ftry-offy2, s=10, linewidth = .5,  zorder=1, marker = '.', edgecolors = 'k', color = 'black')
#plt.plot(TList, offy-offy2)
#plt.grid()
#plt.xlabel('Theta (degrees)')
#plt.ylabel('4 Theta Component (Hz)')
#plt.show()

plotter = f0-ftry-offy
thLine = np.linspace(np.amin(plotter), np.amax(plotter))

plt.polar(TList*np.pi/180, plotter, 'b.')
plt.polar(TList*np.pi/180, offy2, 'g', zorder = 3)
#plt.title('4 Theta Component vs Theta')
plt.polar((283- 196)*np.ones(len(thLine))*np.pi/180, thLine, color = 'r')
plt.polar((283-180- 196)*np.ones(len(thLine))*np.pi/180, thLine, color = 'r')
plt.polar((53- 196)*np.ones(len(thLine))*np.pi/180, thLine, color = 'k')
plt.polar((53-180- 196)*np.ones(len(thLine))*np.pi/180, thLine, color = 'k')
plt.polar((143- 196)*np.ones(len(thLine))*np.pi/180, thLine, color = 'g')
plt.polar((143-180- 196)*np.ones(len(thLine))*np.pi/180, thLine, color = 'g')
plt.polar(-196*np.ones(len(thLine))*np.pi/180, thLine, color = 'b')
plt.polar((-196-180)*np.ones(len(thLine))*np.pi/180, thLine, color = 'b')
plt.ylim(np.amin(thLine), np.amax(thLine))
plt.show()