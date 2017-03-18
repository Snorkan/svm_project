import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import mp2

def train_model(sequence, feature, k, ws):
    print ('train model')
    clf = svm.LinearSVC(class_weight='balanced')

    target_names = ['Membrane', 'Intracellular', 'Globular', 'Extracellular']

    scores = []
    f1scores = []
    predictions = []
    y_true = []

    count = 0
    for n in range(k):
        count += 1

        # k-fold cross-validation, one part for test
        x_test = sequence[n]
        y_test = feature[n]
        y_true.append(y_test)
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
        f1 = f1_score(y_test, prediction, average='macro')

        cm = confusion_matrix(y_test, prediction)

        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix, ws 19')
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
        plt.savefig('./../output/cm_%r' % count, bbox_inches="tight")
        count += 1

        print ('Model %r of %r, ws: %r\n' % (count, k, ws))
        print (classification)

        scores.append(score)
        f1scores.append(f1)
        predictions.append(predictions)

    with open('./../output/svm_ws_test', 'w') as f:
        f.write('predictions: /n%r' % predictions)
        f.write('y_true: /n%r' % y_true)
        f.write('ws: %r, svm, kernel = linear\n' % ws)
        f.write('Score mean: %f\n' % np.mean(scores))
        f.write('Standard deviation: %f\n' % np.std(scores))
        f.write('f1 score mean: %f\n' % np.mean(f1scores))
        f.write('f1 standard deviation: %f\n' % np.std(f1scores))


def run(input):

    ws = 19
    k = 5 # 5-fold cross-validation

    f = mp2.open_file(input)
    splitFile = mp2.split_file_content(f)

    conv_seq = mp2.convert_sequence(splitFile[0], ws)
    conv_label = mp2.convert_features(splitFile[1])

    shuffled = mp2.randomized(conv_seq, conv_label)

    sets = mp2.divide_in_sets(shuffled[0], shuffled[1], k)
    train_model(sets[0], sets[1], k, ws)


input = './../input/x3glob'

run(input)