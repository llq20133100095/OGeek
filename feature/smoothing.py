#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 14:32:06 2018

@author: llq
"""

import numpy
import random
import scipy.special as special
from scipy.special import digamma


class BayesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample(self, alpha, beta, num, imp_upperbound):
        sample = numpy.random.beta(alpha, beta, num)
        I = []
        C = []
        for clk_rt in sample:
            imp = random.random() * imp_upperbound
            imp = imp_upperbound
            clk = imp * clk_rt
            I.append(imp)
            C.append(clk)
        return I, C

    def update(self, imps, clks, iter_num, epsilon):
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(imps, clks, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0

        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clks[i]+alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i]-clks[i]+beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i]+alpha+beta) - special.digamma(alpha+beta))

        return alpha*(numerator_alpha/denominator), beta*(numerator_beta/denominator)


def test():
    bs = BayesianSmoothing(1, 1)
    I, C = bs.sample(500, 500, 1000, 10000)
    print('start')
    print I, C
    bs.update(I, C, 1000, 0.0000000001)
    print('end')
    print bs.alpha, bs.beta


#----------------------------
# get one iteration's alpha/beta.
#----------------------------
def iter_once(alpha, beta, ciList, I_name, C_name):
    sum0 = 0.0
    sum1 = 0.0
    sum2 = 0.0 
    sum3 = 0.0

    sum0=digamma(ciList[C_name]+alpha) - digamma(alpha)
    sum0=sum(sum0)
    sum1=digamma(ciList[I_name]+alpha+beta) - digamma(alpha+beta)
    sum1=sum(sum1)
    sum2=digamma(ciList[I_name]-ciList[C_name]+beta) - digamma(beta)
    sum2=sum(sum2)
    sum3=digamma(ciList[I_name]+alpha+beta) - digamma(alpha+beta)
    sum3=sum(sum3)
    
    alpha = alpha * (sum0 / sum1)
    beta = beta * (sum2 / sum3)

    return alpha,beta 

#------------------------------
# smooth the raw click/impression: ciList. 
#------------------------------
def smooth_ctr(iter_num, alpha, beta, ciList, I_name, C_name):
    i = 0
    while i < iter_num:
        if(i%100==0):
            print "iter:%d. alpha=%s,beta=%s" % (i,alpha, beta)    
        prev_alpha = alpha 
        prev_beta = beta 
     
        alpha, beta = iter_once(alpha, beta, ciList, I_name, C_name)

        ## early-stopping
        if abs(alpha - prev_alpha) < 1E-10 \
                and abs(beta - prev_beta) < 1E-10:
            break
    
        i+=1
        
    ctr=(ciList[C_name]+alpha)/(ciList[I_name]+alpha+beta)
    return ctr

if __name__ == '__main__':
    test()