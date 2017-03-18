import mp2
import models

f = mp2.open_file('./../input/x3glob')
sf = mp2.split_file_content(f)

for i in range(19,21,2):
    outfile = './../output/ws%r' % i
    k = 5

    s = mp2.convert_sequence(sf[0], i)
    f = mp2.convert_features(sf[1])

    r = mp2.randomized(s, f)

    sets = mp2.divide_in_sets(r[0], r[1], k)

    h = models.randomForest(sets[0], sets[1], k, i)