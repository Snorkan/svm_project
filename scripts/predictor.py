from sklearn.externals import joblib

def open_file(file):
    with open(file, 'r') as f:
        fi = f.read().splitlines()
    return fi

def split_file_content(file):
    ids = []
    seqs = []

    for i in range(len(file)):
        if file[i][0] == '>':
            ids.append(file[i])
            seqs.append(file[i + 1])

    return ids, seqs

def decode_labels(labels):

    d = {'M': 1, 'I': 2, 'G': 3, 'O': 4}
    labels = [d[f] for feat in feats for f in feat]

    return labels

fasta = './../input/fastafile.fasta'
f = open_file(fasta)
splitFile = split_file_content(f)
sequence = splitFile[1]

f = open('./../output/svm_model', 'rb')
clf = joblib.load(f)

prediction = clf.predict(sequence)

f.close()





