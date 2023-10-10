from torch.utils.data import Dataset

class CustomTensorDataset(Dataset):

	def __init__(self, tensors, transforms=None):
		self.tensors = tensors
		self.transforms = transforms
		
        