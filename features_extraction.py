import math
import numpy as np
import scipy
from math import sqrt
from scipy.stats import *
from statistics import *
import pandas as pd
import pywt

def extract_features(sublist,t=True,f=False,wa=True):
  features11=[]
  if t:    #9 features
    mean = sum(sublist) / len(sublist)
    sum1= sum([(float(j) - mean)**2 for j in sublist])
    sum2= sum([float(j)**2 for j in sublist])
    IS = sum2/len(sublist)
    var = float(sum1) / (len(sublist) - 1) # Variance
    stdev = sqrt(var) # Standard deviation
    cv = stdev / mean # Coefficient of variation
    med = median(sublist) # median
    rang = max(sublist) - min(sublist) # range
    RMS = math.sqrt(sum2/len(sublist)) # Root Mean Square
    pd_series = pd.Series(sublist)
    counts = pd_series.value_counts()
    ent = entropy(counts)
    features11.extend([mean,IS,var,stdev,cv,med,rang,RMS,ent])
  if f: # 7 features
    DC_Feature = scipy.fft.dct(sublist)[0]
    sig_fft=list(scipy.fft.fft(sublist))
    Med_freq = 1/2*sum(sig_fft)
    max_freq = max(sig_fft)
    min_freq = min(sig_fft)
    mean_power = sum(sig_fft)/len(sig_fft)
    total_power = sum(sig_fft)
    energy_feature=sum(np.abs(sig_fft)**2)
    features11.extend([DC_Feature,Med_freq,max_freq,min_freq,mean_power,total_power,energy_feature])
  if wa:  #15 features
    cA1,cD11 = pywt.wavedec(sublist, wavelet='db2', level=1)
    cA2, cD22, cD12 = pywt.wavedec(sublist, wavelet='db2', level=2)
    cA3, cD33, cD23, cD13 = pywt.wavedec(sublist, wavelet='db2', level=3)
    cA4, cD44,cD34,cD24, cD14 = pywt.wavedec(sublist, wavelet='db2', level=4)
    cA5, cD55,cD45,cD35,cD25, cD15 = pywt.wavedec(sublist, wavelet='db2', level=5)
		#arr, coeff_slices, coeff_shapes = pywt.ravel_coeffs(coeffs)
#print(arr)
    Wave1=np.sum(np.square(cD11))
    Wave2=np.sum(np.square(cD22))
    Wave3=np.sum(np.square(cD12))
    Wave4=np.sum(np.square(cD33))
    Wave5=np.sum(np.square(cD23))
    Wave6=np.sum(np.square(cD13))
    Wave7=np.sum(np.square(cD44))
    Wave8=np.sum(np.square(cD34))
    Wave9=np.sum(np.square(cD24))
    Wave10=np.sum(np.square(cD14))
    Wave11=np.sum(np.square(cD55))
    Wave12=np.sum(np.square(cD45))
    Wave13=np.sum(np.square(cD35))
    Wave14=np.sum(np.square(cD25))
    Wave15=np.sum(np.square(cD15))
    features11.extend([Wave1,Wave2,Wave3,Wave4,Wave5,Wave6,Wave7,Wave8,Wave9,Wave10,Wave11,Wave12,Wave13,Wave14,Wave15])
  return [float(i) for i in features11]