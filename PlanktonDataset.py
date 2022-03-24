from torch.utils.data import Dataset

class PlanktonDataset(Dataset):
    def __init__(self, plankton_dict, transform, target_transform):
        self.labels           = plankton_dict['labels']
        self.images           = plankton_dict['images']
        self.image_names      = dict((v,k) for k,v in plankton_dict['names'].items())
        self.transform        = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            raise ValueError("Transformation Exists in Dataset: ", os.getcwd())
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def getimagename(self, label):
      return self.image_names[label]