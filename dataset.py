from chainer.dataset import DatasetMixin

import nibabel as nib
import numpy as np
from os import listdir, path


class NibDataset(DatasetMixin):
    def __init__(self, directory):
        self.directory = directory
        self.files = sorted(listdir(self.directory))
        self.len = len(self.files)
        self.descriptors = [nib.load(path.join(self.directory, file)) for file in self.files]

    def __len__(self):
        return self.len

    def get_example(self, i):
        return np.asarray(self.descriptors[i].get_data(), dtype=np.float32)
