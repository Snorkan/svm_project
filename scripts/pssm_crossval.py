import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
import pickle
from sklearn.metrics import f1_score
import mp2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


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



def train_model(sequence, feature, k, ws):

    clf = svm.LinearSVC(class_weight='balanced')

    target_names = ['Membrane', 'Intracellular', 'Globular', 'Extracellular']

    scores = []

    f1scores = []

    count = 0
    for n in range(k):
        count += 1

        # k-fold cross-validation, one part for test
        x_test = sequence[n]
        y_test = feature[n]

        # k-1 parts for training
        x_train = []
        y_train = []
        for m in range(k):
            if m != n:
                x_train.extend(sequence[m])
                y_train.extend(feature[m])

        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        prediction = clf.predict(x_test)
        classification = classification_report(y_test, prediction, target_names=target_names)
        print ('Ws: %r, classification report: %r' % (ws, classification))
        # f1 score = average of precision and recall (sensitivity and specificity)
        f1 = f1_score(y_test, prediction, average='macro')

        cm = confusion_matrix(y_test, prediction)

        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('PSSM Confusion Matrix, ws 19')
        plt.colorbar()
        labels = ['Membrane', 'Intracellular', 'Globular', 'Extracellular']

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm2 = np.around(cm, decimals=5)

        t = cm2.max() / 2.
        for i, j in itertools.product(range(cm2.shape[0]), range(cm2.shape[1])):
            plt.text(j, i, cm2[i, j],
                     horizontalalignment="center",
                     color="black")

        ticks = np.arange(len(labels))
        plt.xticks(ticks, labels, rotation=45)
        plt.yticks(ticks, labels)

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('./../output/cm%r' % count, bbox_inches="tight")
        count += 1

        scores.append(score)
        #predictions.append(prediction)
        #classifications.append(classification)
        f1scores.append(f1)

    with open('./../output/pssm_scores', 'a') as f:

        f.write('ws: %r, svm, kernel = linear\n' % ws)
        f.write('Score mean: %f\n' % np.mean(scores))
        f.write('Score std: %r\n' % np.std(scores))
        f.write('f1 score mean: %f\n' % np.mean(scores))
        f.write('f1 score std: %r\n' % np.std(scores))



i = 19
k = 5

input = './../output/pssmID'
outputDir = './..output/pssm_scores'

with open(input, 'rb') as f:
    pssm_dict = pickle.load(f)

f = mp2.open_file('./../input/alpha_beta_globular_sp_4state.txt')
label_dict = split_file_content(f)

connected = connectPssmFeat(pssm_dict, label_dict)

conv_seq = convert_sequence(connected[0], i)
conv_label = mp2.convert_features(connected[1])

shuffled = mp2.randomized(conv_seq, conv_label)
sets = mp2.divide_in_sets(shuffled[0], shuffled[1], k)

train_model(sets[0], sets[1], k, i)








