import mp2

# The data set was reduced due to memory errors

f = mp2.open_file('./../input/alpha_beta_globular_sp_4state.txt')


def divByStructure(file):

    globular = open('globular_whole', 'w')
    transmembrane = open('transmembrane_whole', 'w')

    for i in range(0, int(len(file)), 3):

        if file[i + 2][0] == 'G':

            globular.write(file[i])
            globular.write('\n')
            globular.write(file[i+1])
            globular.write('\n')
            globular.write(file[i+2])
            globular.write('\n')

        else:
            transmembrane.write(file[i])
            transmembrane.write('\n')
            transmembrane.write(file[i + 1])
            transmembrane.write('\n')
            transmembrane.write(file[i + 2])
            transmembrane.write('\n')

    globular.close()
    transmembrane.close()

    print ('Files created: globular and transmembrane.')

divByStructure(f)

gRead = open_file('globular_whole')
tmRead = open_file('transmembrane_whole')


def splitFileContent(file):
    ids = []
    seqs = []
    feats = []

    for i in range(0, int(len(file)), 3):
        ids.append(file[i])
        seqs.append(file[i + 1])
        feats.append(file[i + 2])

    return seqs, feats

def xnSize(glob, tm, outfile):

    for i in range(0, int((len(tm)*2)), 3):
        outfile.write(glob[i])
        outfile.write('\n')
        outfile.write(glob[i+1])
        outfile.write('\n')
        outfile.write(glob[i+2])
        outfile.write('\n')

    for i in range(0, int(len(tm)), 3):
        outfile.write(tm[i])
        outfile.write('\n')
        outfile.write(tm[i+1])
        outfile.write('\n')
        outfile.write(tm[i+2])
        outfile.write('\n')


with open('./../input/x3glob', 'w') as f:
    xnSize(gRead, tmRead, f)





