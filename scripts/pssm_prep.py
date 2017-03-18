import io
import glob
import numpy as np
import math
import pickle
from sklearn.externals import joblib

def files_from_folder(dir):
    return glob.glob(dir + "/**")

# calculate log odds score
def sigmoid(x):
    s = 1/(1 + math.exp(-x))
    return s

def pssmWithID(pssm_directory):

    files = files_from_folder(pssm_directory)
    d = dict()

    for file in files:
        with io.open(file, 'r', encoding="windows-1252") as pssm_file:
            data = np.genfromtxt(file, usecols=range(2,22), dtype=None, skip_header=3, skip_footer=5)
            data2 = np.genfromtxt(file, usecols=range(22, 42), dtype=None, skip_header=3, skip_footer=5)

            id = file.replace(pssm_directory, '').replace('.pssm','').replace('.fasta', '').replace("'", "").replace('>', '').replace('/', '')
            d[id] = []

            for array in data2:
                lo = [sigmoid(i) for i in array]
                d[id].append(lo)

    with open('./../output/pssmID', 'wb') as f:

        pickle.dump(d, f)

pssm_directory = './../input/PSSM'

pssmWithID(pssm_directory)

