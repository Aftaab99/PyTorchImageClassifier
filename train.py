from model import Model
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader import notMNIST
import matplotlib.pyplot as plt
from parameters import MODEL_NAME, N_EPOCHS, BATCH_SIZE

root = os.path.dirname(__file__)

# Instantiating the notMNIST dataset class we created
train_dataset = notMNIST(os.path.join(root, 'Dataset/Train'))
print("Loaded data")

# Creating a dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)

# Instantiating the model, loss function and optimizer
net = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

loss_history = []


def train(epoch):
	epoch_loss = 0
	n_batches = len(train_dataset) // BATCH_SIZE

	for step, data in enumerate(train_loader, 0):
		train_x, train_y = data
		y_hat = net.forward(train_x)
		train_y = torch.Tensor(np.array(train_y))

		# CrossEntropyLoss requires arg2 to be torch.LongTensor
		loss = criterion(y_hat, train_y.long())
		epoch_loss += loss.item()
		optimizer.zero_grad()

		# Backpropagation
		loss.backward()
		optimizer.step()
		# There are len(dataset)/BATCH_SIZE batches.
		# We print the epoch loss when we reach the last batch.
		if step % n_batches == 0 and step != 0:
			epoch_loss = epoch_loss / n_batches
			loss_history.append(epoch_loss)
			print("Epoch {}, loss {}".format(epoch, epoch_loss))
			epoch_loss = 0


for epoch in range(1, N_EPOCHS + 1):
	train(epoch)

# Saving the model
torch.save(net, 'models/{}.pt'.format(MODEL_NAME))
print("Saved model...")

# Plotting loss vs number of epochs
plt.plot(np.array(range(1, N_EPOCHS + 1)), loss_history)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()
