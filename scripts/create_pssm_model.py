import mp2
from sklearn import svm
from sklearn.externals import joblib
import pickle
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
import math
import pickle
from sklearn.metrics import f1_score
from sklearn.externals import joblib
import mp2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

def open_file(file):
    with open(file, 'r') as f:
        fi = f.read().splitlines()
    return fi

def split_file_content(file):
    ids = []
    seqs = []
    feats = []

    for i in range(0, int(len(file)), 3):
        ids.append(file[i].replace('>', ''))
        seqs.append(file[i + 1])
        feats.append(file[i + 2])

    print ('Features connected to id')
    d = dict(zip(ids, feats))

    print (d)
    return d

def connectPssmFeat(pssmDict, featDict):
    pssm_combined = dict()

    for k,v in pssmDict.items():

        if k in pssmDict.keys():
            if len(pssmDict[k]) == len(featDict[k]):

                pssm_combined[k] = (pssmDict[k], featDict[k])


    print ('Features and pssm connected to id')
    """print (pssmFeat)

    with open('./../oputput/pssmFeatId', 'wb') as f:
        pickle.dump(pssmFeat, f)"""

    seqs = []
    feats = []

    for v in pssm_combined.values():

        seqs.append(v[0])
        feats.append(v[1])

    return seqs, feats

def convert_sequence(sequences, ws):

    z = int((ws - 1) / 2)
    seqList = []

    for sequence in sequences:
        # pads the sequence with (ws-1)/2 zeros
        pssm = np.lib.pad(sequence, (z, z), 'constant', constant_values=(0, 0))

        slist = []
        for i in range(len(pssm)):

            if i + ws > len(pssm):
                break

            else:
                b = pssm[i:i + ws]
                c = np.concatenate(b, axis=0)
                slist.append(c)

        seqList.extend(slist)

    return seqList


def train_model(sequence, label):
    clf = svm.LinearSVC(C=1.2, class_weight='balanced')

    clf.fit(sequence, label)

    with open('./../output/svm_model_cf', 'wb') as f:
        pickle.dump(clf, f)



i = 19
k = 5

input = './../output/pssmID'

with open(input, 'rb') as f:
    pssm_dict = pickle.load(f)

f = open_file('./../input/alpha_beta_globular_sp_4state.txt')
label_dict = split_file_content(f)

connected = connectPssmFeat(pssm_dict, label_dict)

conv_seq = convert_sequence(connected[0], i)
conv_label = mp2.convert_features(connected[1])

train_model(conv_seq, conv_label)
