import csv
import matplotlib.pyplot as plt
import numpy as np


case = 'SuperResolution'
filename = 'Results/' + case + '/PSNR.csv'
iterations = 2000

def read_table(file):
    dta = []
    for line in csv.reader(file):
        dta.append(line)
    dta.pop(0)
    return np.asarray(dta)

def get_PSNR(dta):
    loss = []
    for l in dta[:]:
        loss.append(float(l[1]))
    return loss

PSNR = get_PSNR(read_table(open(filename, 'r')))

# training loss resnet 20, 56, 110
plt.subplots()
plt.plot(range(iterations), PSNR, label= 'PSNR')

plt.legend()
plt.ylim([0., 35])
plt.xlabel("Iteration")
plt.ylabel('PSNR')
plt.savefig('Results/' + case + '/PSNR.png', dpi=300, bbox_inches='tight')
plt.close()
