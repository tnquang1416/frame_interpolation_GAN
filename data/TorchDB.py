import numpy as np
from os import listdir
from PIL import Image
from os.path import join, isdir
from torch.utils.data import Dataset
from torchvision import transforms


class DBreader_frame_interpolation(Dataset):
    """
    DBreader reads all triplet set of frames in a directory or from input tensor list
    """

    def __init__(self, db_dir=None, resize=None, tensor_list=None):
        if db_dir is not None:
            self._load_from_dir(db_dir, resize)
            self.mode = 1
        elif tensor_list is not None:
            self._load_from_tensor_list(tensor_list)
            self.mode = 2
            
    def _load_from_tensor_list(self, tensor_list):
        '''
        Load from numpy tensor list (no_triplets, triplets_index, c, w, h)
        :param tensor_list:
        '''
        self.triplet_list = tensor_list
        
    def _load_from_dir(self, db_dir, resize=None):
        '''
        DBreader reads all triplet set of frames in a directory.
        Each triplet set contains frame 0, 1, 2.
        Each image is named frame0.png, frame1.png, frame2.png.
        Frame 0, 2 are the input and frame 1 is the output.
        '''
        if resize is not None:
            self.transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

        self.triplet_list = np.array([(db_dir + '/' + f) for f in listdir(db_dir) if isdir(join(db_dir, f))])
        self.file_len = len(self.triplet_list)

    def __getitem__(self, index):
        if self.mode == 1:
            frame0 = self.transform(Image.open(self.triplet_list[index] + "/frame0.png"))
            frame1 = self.transform(Image.open(self.triplet_list[index] + "/frame1.png"))
            frame2 = self.transform(Image.open(self.triplet_list[index] + "/frame2.png"))
        elif self.mode == 2:
            # transformed already
            frame0 = self.triplet_list[index][0]
            frame1 = self.triplet_list[index][1]
            frame2 = self.triplet_list[index][2]

        return frame0, frame1, frame2

    def __len__(self):
        return self.file_len
