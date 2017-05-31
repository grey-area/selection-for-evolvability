#!/usr/bin/env python

import best
import numpy as np
from pymc import Uniform, Normal, Exponential, NoncentralT, deterministic, Model, MCMC
import cPickle as pickle
import itertools
import sys

if len(sys.argv) < 2:
    sys.exit()
else:
    pid = int(sys.argv[1])

data_directory = '../processed_data/2_train_test_type_split/'
dest_directory = '../processed_data/4_classify_and_inference/'

def make_model(data):

    empirical_mean = np.mean(data)
    empirical_std  = np.std(data)

    mean = Normal('mean', empirical_mean, 1e-6 * 1/empirical_std**2)
    std  = Uniform('std', 0.001 * empirical_std, 1000 * empirical_std)
    nu_minus_one = Exponential('nu_minus_one', 1/29.0)

    @deterministic(plot=False)
    def nu(n=nu_minus_one):
        out = n+1
        return out

    @deterministic(plot=False)
    def lam(s=std):
        out = 1/s**2
        return out

    data_dist = NoncentralT('data', mean, lam, nu, value=data, observed=True)

    return Model({'data_dist':data_dist, 'mean':mean})

ffs = ['simple','mask','symmetry','lipson']
methods = ['kalman','particle']
r1s = ['f','e']

for index, (ff, method, r1) in enumerate(itertools.product(ffs, methods, r1s)):

    if index % 4 != pid:
        continue

    filename = '%s%s_%s_%s_EvsPTS_' % (data_directory, ff, method, r1)
    
    with open('%strain.dat' % filename) as f:
        f.readline()
        data = np.loadtxt(f, delimiter=',')[:, -2]
    with open('%stest.dat' % filename) as f:
        f.readline()
        data = np.hstack([data, np.loadtxt(f, delimiter=',')[:, -2]])
        
    model = make_model(data)
    M = MCMC(model)
    M.sample(iter=10000, burn=2000)
    stats = {}
    stats['empirical_median'] = np.median(data)
    for stat in ['mean','std']:
        samples = M.trace(stat)[:]
        stats['%s_summary_stats' % stat] = best.calculate_sample_statistics(samples)
        stats['%s_samples' % stat] = samples

        if stat == 'mean':
            stats['mean_probability_positive'] = (samples > 0).mean()

    filename = 'grouped_%s_%s_%s_EvsPTS.pkl' % (ff, method, r1)
    with open('%s%s' % (dest_directory, filename), 'wb') as f:
        pickle.dump(stats, f)
