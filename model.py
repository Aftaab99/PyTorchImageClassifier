import torch.nn as nn
import torch
import torch.nn.functional as F


# All torch models have to inherit from the Module class
class Model(torch.nn.Module):

	def __init__(self):
		super(Model, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

		# Reshaping the tensor to BATCH_SIZE x 320. Torch infers this from other dimensions when one of the parameter is -1.
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = F.dropout(x)
		x = self.fc2(x)
		return F.softmax(x, dim=1)
