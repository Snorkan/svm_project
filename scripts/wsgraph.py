import matplotlib.pyplot as plt

def open_file(file):
    with open(file, 'r') as f:
        fi = f.read().splitlines()
    return fi

# evaluate window size optimization
#sFile = open_file('./../output/scores_750')

sFile = open_file('c_optimization')

m = []
std = []
wss = []

for i in range(0, len(sFile), 5):
    ws = sFile[i].split(' ')
    c = ws[1].replace(',', '')
    wss.append(c)
    f1 = (sFile[i+3])
    f2 = f1.split(' ')
    f3 = (sFile[i+4])
    f4 = f3.split(' ')
    m.append(f2[3])
    std.append(f4[3])

print (m)
print (std)
print (wss)
c = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]

fig = plt.figure()
plt.plot(c, m)
fig.suptitle('C parameter performance')
plt.xlabel('C value')
plt.ylabel('F1 score')
fig.savefig('c.png', bbox_inches='tight')






