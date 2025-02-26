# -*- coding: utf-8 -*-
#!/usr/bin/python

from __future__ import division
import numpy as np  

N=2000
k=9
J=4

sigma=np.eye(k)

f = open('data.txt','w')

for j in range(0,J):

    mu=np.random.random(k)*k
    A = np.random.multivariate_normal(mu, sigma, N)
    
    for n in A:
        for s in n:
            f.write(str(s)+'\t') 
        f.write(str(j+1))
        f.write('\r\n') #\r\n为换行符  
        
f.close()
