import mp2
from sklearn import svm
from sklearn.externals import joblib
import pickle

def train_model(sequence, label):
    clf = svm.LinearSVC(C=1.2, class_weight='balanced')

    clf.fit(sequence, label)

    with open('./../output/svm_model_cf', 'wb') as f:
        pickle.dump(clf, f)


ws = 19
input = './../input/x3glob'

f = mp2.open_file(input)
splitFile = mp2.split_file_content(f)

conv_seq = mp2.convert_sequence(splitFile[0], ws)
conv_label = mp2.convert_features(splitFile[1])

train_model(conv_seq, conv_label)