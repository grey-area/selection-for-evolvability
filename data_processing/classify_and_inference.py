#!/usr/bin/env python

import best
import os
import numpy as np
from pymc import MCMC
import cPickle as pickle
import sys
import matplotlib.pyplot as plt # temp

if len(sys.argv) < 2:
    sys.exit()
else:
    pid = int(sys.argv[1])

stats = {}

data_directory = '../processed_data/2_train_test_type_split/'
tree_directory = '../processed_data/3_decision_trees/'
dest_directory = '../processed_data/4_' +  os.path.basename(__file__).split('.')[0] + '/'

def reject(data):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0
    return data[s < 5]

tree_files = filter(lambda x: x.endswith('.model'), sorted(os.listdir(tree_directory)))
for filename_i, filename in enumerate(tree_files):

    if filename_i % 4 != pid:
        continue
        
    treename = filename.split('.')[0]

    with open('%s%s' % (tree_directory, filename)) as f:
        clf = pickle.load(f)
        
    with open('%s%s_train.dat' % (data_directory, treename)) as f:
        f.readline()
        training_data = np.loadtxt(f, delimiter=',')
    with open('%s%s_test.dat' % (data_directory, treename)) as f:
        f.readline()
        testing_data = np.loadtxt(f, delimiter=',')

    training_signs = training_data[:, -1]
    testing_signs = testing_data[:, -1]
    proportion_positive = ((training_signs == 1).sum() + (testing_signs == 1).sum()) / float(training_signs.size + testing_signs.size)
    stats['proportion_positive'] = proportion_positive

    testing_Xs = testing_data[:, :-2]
    testing_values = testing_data[:, -2]
    
    predictions = clf.predict(testing_Xs)
    stats['true_positives'] = np.logical_and(predictions == 1, testing_signs == 1).sum()
    stats['false_negatives'] = np.logical_and(predictions == 0, testing_signs == 1).sum()
    stats['false_positives'] = np.logical_and(predictions == 1, testing_signs == 0).sum()
    stats['true_negatives'] = np.logical_and(predictions == 0, testing_signs == 0).sum()
    
    negative = testing_values[predictions == 0]
    positive = testing_values[predictions == 1]
    
    stats['negative_empirical_median'] = np.median(np.hstack([negative,[0.0]]))
    stats['positive_empirical_median'] = np.median(np.hstack([positive,[0.0]]))

    '''
    print filename
    print "Proportion positive: %f" % proportion_positive
    print "Empirical mean: %f" % testing_values.mean()
    print "Confusion matrix"
    print stats['true_positives'], stats['false_negatives']
    print stats['false_positives'], stats['true_negatives']
    print "Accuracy: %f" % (testing_signs == predictions).mean()
    print "Positive empirical mean: %f" % stats['positive_empirical_mean']
    print "Negative empirical mean: %f" % stats['negative_empirical_mean']
    print ""
    '''
    
    data = {'negative':negative,'positive':positive}

    model = best.make_model(data)
    M = MCMC(model)
    # temp, change to 25000
    M.sample(iter=15000, burn=5000) # todo 25000, 5000

    negative_means = M.trace('group1_mean')[:] # negatives
    positive_means = M.trace('group2_mean')[:] # positives

    negative_stats = best.calculate_sample_statistics(negative_means)
    positive_stats = best.calculate_sample_statistics(positive_means)

    for group, stats_dict in zip(['negative', 'positive'], [negative_stats, positive_stats]):
        for stat in ['mean','mode','hdi_min','hdi_max']:
            stats['%s_mean_%s' % (group, stat)] = stats_dict[stat]
    stats['probability_positive_is_positive'] = (positive_means > 0).mean()
    stats['probability_positive_greater_than_negative'] = ((positive_means - negative_means) > 0).mean()

    negative_stds = M.trace('group1_std')[:]
    positive_stds = M.trace('group2_std')[:]

    negative_std_stats = best.calculate_sample_statistics(negative_stds)
    positive_std_stats = best.calculate_sample_statistics(positive_stds)

    for group, stats_dict in zip(['negative', 'positive'], [negative_std_stats, positive_std_stats]):
        for stat in ['mean','mode','hdi_min','hdi_max']:
            stats['%s_std_%s' % (group, stat)] = stats_dict[stat]

    with open('%s%s.pkl' % (dest_directory, treename),'wb') as f:
        pickle.dump(stats, f)
