# python run_reg_sep.py -d dataset -hi no_hiddens -m no_pseudos -i no_iterations -b batch_size
# for example, for boston housing dataset, remember to check path to data, hard-coded in this file
# python run_reg_sep.py -d boston -hi 2 -m 50 -i 5000 -b 50
import sys
sys.path.append('../code/')
import os
import math
import numpy as np
import cPickle as pickle
import AEPDGP_net
from tools import *
import argparse
import time

parser = argparse.ArgumentParser(description='run regression experiment',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--dataset',
            action="store", dest="dataset",
            help="dataset name, eg. boston, power", default="boston")
parser.add_argument('-hi', '--hiddens', nargs='+', type=int,
            action="store", dest="n_hiddens",
            help="number of hidden dimensions, eg. 2 or 5 2", default=[])
parser.add_argument('-m', '--pseudos', type=int,
            action="store", dest="n_pseudos",
            help="number of pseudo points per layer, eg. 10", default=10)
parser.add_argument('-b', '--minibatch', type=int,
            action="store", dest="minibch_size",
            help="minibatch size, eg. 10", default=50)
parser.add_argument('-i', '--iterations', type=int,
            action="store", dest="n_iterations",
            help="number of stochastic updates, eg. 10", default=1000)
parser.add_argument('-s', '--seed', type=int,
            action="store", dest="random_seed",
            help="random seed, eg. 10", default=123)
parser.add_argument('-l', '--lrate', type=float,
            action="store", dest="lrate",
            help="adam learning rate", default=0.005)
parser.add_argument('-t', '--tied',
            action="store_true", dest="tied",
            help="tying inducing point (boolean)", default=False)

args = parser.parse_args()

name = args.dataset
n_hiddens = args.n_hiddens
n_hiddens_str = '_'.join(map(str, n_hiddens))
nolayers = len(n_hiddens) + 1
M = args.n_pseudos
n_pseudos = [M for _ in range(nolayers)]
no_iterations = args.n_iterations
no_points_per_mb = args.minibch_size
random_seed = args.random_seed
np.random.seed(random_seed)
lrate = args.lrate
tied = args.tied

fnames = {'boston': 'bostonHousing',
          'power': 'power-plant',
          'concrete': 'concrete',
          'energy': 'energy',
          'kin8nm': 'kin8nm',
          'naval': 'naval-propulsion-plant',
          'protein': 'protein-tertiary-structure',
          'wine_red': 'wine-quality-red',
          'yacht': 'yacht',
          'year': 'YearPredictionMSD'}

# We load the dataset
datapath = '../../datasets/' + fnames[name] + '/data/'
datafile = datapath + 'data.txt'
data = np.loadtxt(datafile)

# We obtain the features and the targets
xindexfile = datapath + 'index_features.txt'
yindexfile = datapath + 'index_target.txt'
xindices = np.loadtxt(xindexfile, dtype=np.int)
yindex = np.loadtxt(yindexfile, dtype=np.int)
X = data[:, xindices]
y = data[:, yindex]
y = y.reshape([y.shape[0], 1])

# We obtain the number of splits available
nosplits_file = datapath + 'n_splits.txt'
nosplits = np.loadtxt(nosplits_file, dtype=np.int8)

# prepare output files
outname1 = '/tmp/' + name + '_' + n_hiddens_str + '_' + str(M) + '.rmse'
if not os.path.exists(os.path.dirname(outname1)):
    os.makedirs(os.path.dirname(outname1))
outfile1 = open(outname1, 'w')
outname2 = '/tmp/' + name + '_' + n_hiddens_str + '_' + str(M) + '.nll'
outfile2 = open(outname2, 'w')
outname3 = '/tmp/' + name + '_' + n_hiddens_str + '_' + str(M) + '.time'
outfile3 = open(outname3, 'w')

for i in range(nosplits):
    print 'split', i

    train_ind_file = datapath + 'index_train_' + str(i) + '.txt'
    test_ind_file = datapath + 'index_test_' + str(i) + '.txt'
    index_train = np.loadtxt(train_ind_file, dtype=np.int)
    index_test = np.loadtxt(test_ind_file, dtype=np.int)
    X_train = X[index_train, :]
    y_train = y[index_train, :]
    X_test = X[index_test, :]
    y_test = y[index_test, :]

    # We construct the network
    net = AEPDGP_net.AEPDGP_net(X_train, y_train, n_hiddens, n_pseudos, lik='Gaussian', zu_tied=tied)
    # train
    t0 = time.time()
    no_epochs = np.ceil(no_iterations * no_points_per_mb * 1.0 / X_train.shape[0])
    test_nll, test_rms, logZ = net.train(X_test, y_test, no_epochs=no_epochs,
                                   no_points_per_mb=no_points_per_mb,
                                   lrate=lrate)
    t1 = time.time()
    outfile3.write('%.6f\n' % (t1-t0))
    outfile3.flush()
    os.fsync(outfile3.fileno())

    # We make predictions for the test set
    m, v = net.predict(X_test)
    # We compute the test RMSE
    rmse = np.sqrt(np.mean((y_test - m)**2))
    outfile1.write('%.6f\n' % rmse)
    outfile1.flush()
    os.fsync(outfile1.fileno())

    # We compute the test log-likelihood
    test_nll = np.mean(-0.5 * np.log(2 * math.pi * v) - 0.5 * (y_test - m)**2 / v)
    outfile2.write('%.6f\n' % test_nll)
    outfile2.flush()
    os.fsync(outfile2.fileno())

outfile1.close()
outfile2.close()
outfile3.close()
