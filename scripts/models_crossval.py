import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.externals import joblib

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

        print ('Model %r of %r, ws: %r\n' % (count, k, ws))
        print (classification)

        scores.append(score)
        f1scores.append(f1)

    print ('ws: %r' % ws)
    print ('Scores: %r\n' % scores)
    print ('Score mean: %f\n' % np.mean(scores))

    with open('./../output/svm_ws47', 'a') as f:
        f.write('ws: %r, svm, kernel = linear\n' % ws)
        f.write('Score mean: %f\n' % np.mean(scores))
        f.write('Standard deviation: %f\n' % np.std(scores))
        f.write('f1 score mean: %f\n' % np.mean(f1scores))
        f.write('f1 standard deviation: %f\n' % np.std(f1scores))

    with open('./../output/svm_model', 'w') as f:
        joblib.dump(clf, f)


def randomForest(sequence, feature, k, ws):
    #n = math.sqrt(4) # sqrt(n_samples) gives good default values for classification problems
    # If “auto”, then max_features=sqrt(n_features)

    clf = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0, class_weight='balanced')

    target_names = ['Membrane', 'Intracellular', 'Globular', 'Extracellular']

    scores = []
    f1scores = []

    count = 0

    for n in range(k):
        count += 1

        # k-fold cross-validation, one part for test
        x_test = sequence[n]
        y_test = feature[n]

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

        scores.append(score)
        f1scores.append(f1)
        print('Model %r of %r, ws: %r' % (count, k, ws))
        print (classification)

    print('ws: %r' % ws)
    print ('Scores: %r' % scores)
    print ('Score mean: %f' % np.mean(scores))

    with open('./../output/scores_rf', 'a') as f:
        f.write('ws: %r, random forest, n decision threes: 50\n' % ws)
        f.write('Score mean: %f\n' % np.mean(scores))
        f.write('Score standard deviation: %f\n' % np.std(scores))
        f.write('F1 mean: %r\n' % np.mean(f1scores))
        f.write('F1 standard deviation: %r\n' % np.std(f1scores))