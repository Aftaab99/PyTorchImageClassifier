from model import Model
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from data_loader import data_loader
import matplotlib.pyplot as plt
from parameters import MODEL_NAME, N_EPOCHS

root = os.path.dirname(__file__)
train_dataset = data_loader(os.path.join(root, 'Dataset/Train'))
print("Loaded data")
train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
net = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

loss_history = []


def train(epoch):
	for step, data in enumerate(train_loader, 0):
		train_x, train_y = data
		y_hat = net.forward(train_x)
		train_y = torch.Tensor(np.array(train_y))
		loss = criterion(y_hat, train_y.long())
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if step % 256 == 0:
			loss_history.append(loss.item())
			print("Epoch {}, loss {}".format(epoch, loss.item()))


for epoch in range(1, N_EPOCHS + 1):
	train(epoch)

torch.save(net, 'models/{}.pt'.format(MODEL_NAME))
print("Saved model...")

# Plotting loss vs number of epochs
plt.plot(np.array(range(1, N_EPOCHS + 1)), loss_history)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()
