import csv
import matplotlib.pyplot as plt
import numpy as np


case = 'Denoising'
filename = 'Results/' + case + '/PSNR.csv'
iterations = 1800

def read_table(file):
    dta = []
    for line in csv.reader(file):
        dta.append(line)
    return np.asarray(dta)

def get_PSNR(dta):
    loss = []
    for l in dta[1:, 1]:
        loss.append(float(l))
    return loss

PSNR = get_PSNR(read_table(open(filename, 'r')))

# training loss resnet 20, 56, 110
plt.subplots()
plt.plot(range(iterations), PSNR, label= 'PSNR')

plt.legend()
plt.ylim([0., 30])
plt.xlabel("Iteration")
plt.ylabel('PSNR')
plt.savefig('Results/' + case + 'PSNR/.png", dpi=300, bbox_inches='tight')
plt.close()
