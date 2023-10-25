import math
import pandas as pd
from statistics import mean, median
from scipy.stats import entropy
import pywt

def extract_wavelet_features(numbers, wavelet='db2', level=2):
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(numbers, wavelet, level=level)

    # Initialize dictionaries for detail and approximation features
    detail_features = {}
    approx_features = {}

    # Calculate statistical features for each detail coefficient
    for i in range(1, len(coeffs)):
        detail_features[f'detail_{i}_mean'] = sum(coeffs[i]) / len(coeffs[i])
        detail_features[f'detail_{i}_variance'] = sum(
            [(j - detail_features[f'detail_{i}_mean']) ** 2 for j in coeffs[i]]) / len(coeffs[i])

    # Calculate statistical features for the approximation coefficient (cA)
    cA = coeffs[0]
    approx_features['cA_mean'] = sum(cA) / len(cA)
    approx_features['cA_variance'] = sum([(j - approx_features['cA_mean']) ** 2 for j in cA]) / len(cA)

    # Combine detail and approximation features
    wavelet_features = {**approx_features, **detail_features}

    return wavelet_features


def compute_features(numbers):
    # Calculate the mean
    mean_value = mean(numbers)

    # Calculate the sums and other intermediate values
    sum1 = sum([(float(j) - mean_value)**2 for j in numbers])
    sum2 = sum([float(j)**2 for j in numbers])
    IS = sum2 / len(numbers)
    var = float(sum1) / (len(numbers) - 1)  # Variance
    stdev = math.sqrt(var)  # Standard deviation
    cv = stdev / mean_value  # Coefficient of variation
    med = median(numbers)  # Median
    rang = max(numbers) - min(numbers)  # Range
    RMS = math.sqrt(sum2 / len(numbers))  # Root Mean Square

    # Convert the list to a Pandas Series for value counts
    pd_series = pd.Series(numbers)
    counts = pd_series.value_counts()

    # Calculate entropy
    ent = entropy(counts)

    # Create a dictionary to store the computed features
    features = {
        'mean': mean_value,
        'sum1': sum1,
        'sum2': sum2,
        'IS': IS,
        'variance': var,
        'stdev': stdev,
        'cv': cv,
        'median': med,
        'range': rang,
        'RMS': RMS,
        'entropy': ent
    }

    return features

def extract_all_features(L):
    features_stat = compute_features(L).values()
    features_wav = extract_wavelet_features(L).values()
    all_features = list(features_stat) + list(features_wav)
    return all_features