import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from scipy.misc import imread
from torchvision.transforms import transforms

"""
Loads the train/test set. 
Every image in the dataset is 28x28 pixels and the labels are numbered from 0-9
for A-J respectively.

Set root to point to the Train/Test folders.
"""

class data_loader(Dataset):
	def __init__(self, root):
		Images, Y = [], []
		folders = os.listdir(root)

		for folder in folders:
			folder_path = os.path.join(root, folder)
			for ims in os.listdir(folder_path):
				try:
					img_path = os.path.join(folder_path, ims)
					Images.append(np.array(imread(img_path)))
					Y.append(ord(folder) - 65)  # Folders are A-J so labels will be 0-9
				except:
					print("File {}/{} is broken".format(folder, ims))
		data = [(x, y) for x, y in zip(Images, Y)]
		self.data = data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		img = self.data[index][0]

		# 8 bit images. Scale between 0, 1
		img = img.reshape(1, 28, 28) / 255

		# Input for Conv2D should be Channels x Height x Width
		img_tensor = transforms.ToTensor()(img).view(1, 28, 28).float()
		label = self.data[index][1]
		return (img_tensor, label)
