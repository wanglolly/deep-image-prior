import csv
import matplotlib.pyplot as plt
import numpy as np


filename = ['Results/LearningCurve_image.csv',
            'Results/LearningCurve_imageNoise.csv',
            'Results/LearningCurve_imageShuffle.csv',
            'Results/LearningCurve_Noise.csv']

def read_table(file):
    dta = []
    for line in csv.reader(file):
        dta.append(line)
    return np.asarray(dta)

def get_train_loss(dta):
    loss = []
    for l in dta[:, 1]:
        loss.append(float(l))
    return loss

imageloss = get_train_loss(read_table(open(trainFilename[0], 'r')))
imageNoiseloss = get_train_loss(read_table(open(trainFilename[1], 'r')))
imageShuffleloss = get_train_loss(read_table(open(trainFilename[2], 'r')))
noiseloss = get_train_loss(read_table(open(trainFilename[3], 'r')))

# training loss resnet 20, 56, 110
plt.subplots()
plt.plot(range(2400), imageloss, label= 'Image')
plt.plot(range(2400), imageNoiseloss, label= 'Image + noise')
plt.plot(range(2400), imageShuffleloss, label= 'Image shuffled')
plt.plot(range(2400), imageShuffleloss, label= 'U(0,1) noise')
plt.legend()
plt.ylim([0., 1.0])
plt.xlabel("Iteration")
plt.ylabel('MSE')
plt.savefig("LearningCurveLoss.png", dpi=300, bbox_inches='tight')
plt.close()
