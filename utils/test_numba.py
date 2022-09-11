# -*- coding: utf-8 -*-
""" Using arrays as replacement for a struct. """
from numba import jit
import numpy as np

V_WIND = 8.0

V_wind           = 0 # (westwind, downwind direction to the east)
V_wind_gnd       = 1
Result           = 2

@jit(nopython = True)
def sub(a, b, result):
    """ Calculate the difference of two 3d vectors. """
    result[0] = a[0] - b[0]    
    result[1] = a[1] - b[1]
    result[2] = a[2] - b[2]  

@jit(nopython = True)
def getVec3():      
    vec3 = np.zeros((3, 3))
    vec3[V_wind, 0 ]        = 1.5 * V_WIND # (westwind, downwind direction to the east at 200m height)
    vec3[V_wind_gnd, 0]    = V_WIND       # (westwind, downwind direction to the east)    
    return vec3

@jit(nopython=True)
def test(vec3):
    sub(vec3[V_wind], vec3[V_wind_gnd], vec3[Result])

def init():
    vec3 = getVec3()
    print (vec3[V_wind]    )
    print (vec3[V_wind_gnd])
    test(vec3)
    print (vec3[Result])

init()