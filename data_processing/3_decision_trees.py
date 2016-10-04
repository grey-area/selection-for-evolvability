#!/usr/bin/env python

import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import cPickle as pickle

data_directory = '../processed_data/2_train_test_type_split/'
dest_directory = '../processed_data/' +  os.path.basename(__file__).split('.')[0] + '/'

to_remove = os.listdir(dest_directory)
for filename in to_remove:
    os.remove(dest_directory + filename)

accuracies_on_negative = [0.0]
accuracies_on_positive = [0.0]
    
data_files = os.listdir(data_directory)
for filename_i, filename in enumerate(data_files):
    if filename.endswith('test.dat'):
        continue
    
    treename = filename.split('_train')[0]
    
    with open(data_directory + filename) as f:
        feature_names = f.readline().split(',')[:-2]
        data = np.loadtxt(f, delimiter=',')
    Xs = data[:, :-2]
    ys = data[:, -1]
        
    clf = DecisionTreeClassifier(class_weight = "balanced", min_weight_fraction_leaf = 0.1)
    clf.fit(Xs, ys)
    export_graphviz(clf, '%s%s.dot' % (dest_directory, treename), feature_names = feature_names, class_names = ['negative', 'positive'], label = 'none', filled = False, impurity = False, node_ids = False, proportion = False)
    # To convert, dot -Tps tree.dot -o tree.ps
    
    with open('%s%s.model' % (dest_directory, treename), 'wb') as f:
        pickle.dump(clf, f)

