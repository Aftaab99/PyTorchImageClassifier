import numpy as np
import torch
from torch.utils.data import DataLoader
from data_loader import data_loader
import os
from parameters import MODEL_NAME

path = os.path.join(os.path.dirname(__file__), 'Dataset/Test')
test_dataset = data_loader(path)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
classifier=torch.load('models/{}.pt'.format(MODEL_NAME))
correct=0

for _, data in enumerate(test_loader, 0):
	test_x, test_y = data
	pred=classifier.forward(test_x)
	y_hat=np.argmax(pred.data)
	if y_hat==test_y:
		correct+=1

print("Accuracy={}".format(correct/len(test_dataset)))
