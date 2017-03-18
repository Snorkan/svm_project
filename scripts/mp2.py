import numpy as np
import random

def open_file(file):
    with open(file, 'r') as f:
        fi = f.read().splitlines()
    return fi

def split_file_content(file):
    ids = []
    seqs = []
    feats = []

    for i in range(0, int(len(file)), 3):
        ids.append(file[i])
        seqs.append(file[i + 1])
        feats.append(file[i + 2])

    return seqs, feats

def convert_sequence(sequence, ws):
    z = int((ws-1)/2)

    all_aminoacids = list('GAVLIPFYWSTCMNQKRHDE')
    aminoacid_dict = {'X': np.zeros(20)}

    count = 0
    for i in all_aminoacids:
        a = np.zeros(20)
        a[count] = 1

        aminoacid_dict[i] = a
        count += 1

    windows = []

    for seq in sequence:
        sw = []
        a = [aminoacid_dict[i] for i in seq]
        a = np.lib.pad(a, (z, z), 'constant', constant_values=(0, 0))

        for i in range(len(a)):
            if i + ws > len(a):
                break
            else:
                b = (a[i:i + ws])
                c = np.concatenate(b, axis=0)
                sw.append(c)

        windows.extend(sw)

    print ('Sequence window l: %r' % len(windows))
    return windows


def convert_features(feats):

    d = {'M': 1, 'I': 2, 'G': 3, 'O': 4}
    labels = [d[f] for feat in feats for f in feat]

    return labels

def randomized(sequence, feature):
    new_seq = []
    new_feat = []

    rand = list(zip(sequence, feature))
    random.shuffle(rand)

    for i in range(len(rand)):
        new_seq.append(rand[i][0])
        new_feat.append(rand[i][1])

    return new_seq, new_feat

def divide_in_sets(s, l, k):

    slice = 0
    gap = int(len(s)/k)
    slist = []
    flist = []

    for i in range(k):
        sl = s[slice + gap * i: slice + gap * i + gap]
        slist.append(sl)
        fl = l[slice + gap * i: slice + gap * i + gap]
        flist.append(fl)

    return slist, flist