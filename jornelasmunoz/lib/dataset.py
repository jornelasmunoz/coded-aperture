# Load the data and then do MURA transformations
import os
import numpy as np
import torch
import torchvision
from torchvision.datasets.utils import download_and_extract_archive, extract_archive, verify_str_arg, check_integrity
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.mnist import read_image_file, read_label_file
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image
import MURA as mura

class MNIST_MURA(VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    mirrors = [
        'http://yann.lecun.com/exdb/mnist/',
        'https://ossci-datasets.s3.amazonaws.com/mnist/',
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, params, train=True, transform=None, target_transform=None, download=False):
        super(MNIST_MURA, self).__init__(root, transform=transform,
                                    target_transform=target_transform)

        self.train = train  # training set or test set
        self.params = params
        self._read_params(self.params)
        
        if self._check_legacy_exist():
            self.data, self.targets = self._load_legacy_data()
            
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        # Load data
        self.data, self.targets, self.digits = self._load_data()
    
    
    def _check_exists(self):
        return all(
            check_integrity(os.path.join(self.raw_folder, os.path.splitext(os.path.basename(url))[0]))
            for url, _ in self.resources
        )
    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False
    def _load_legacy_data(self):
        # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
        # directly.
        data_file = self.training_file if self.train else self.test_file
        return torch.load(os.path.join(self.processed_folder, data_file))
        
    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))
        
        # Compute MURA encoder and decoder (numpy)
        self.A = mura.create_binary_aperture_arr(self.image_size)
        self.G = mura.create_decoding_arr(self.A)
        
        # Resize data to prime number length and convolve with aperture
        data_resized = torchvision.transforms.functional.resize(data, [self.image_size,self.image_size], antialias=True)
        
        # normalize target data
        data_resized =  mura.normalize(data_resized.to(torch.float32))
        mura_data = torch.empty(data_resized.size())
        for idx, img in enumerate(data_resized):
            mura_data[idx] = mura.normalize(mura.FFT_convolve(img.squeeze(0), self.A,self.image_size))
            # mura.FFT_convolve(img.squeeze(0), self.A,self.image_size)
                        # torch.Tensor(mura.normalize(
                        #         mura.FFT_convolve(img.squeeze(0), self.A,self.image_size)),
                        #         dtype= torch.float)
                            
            
        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        digits = read_label_file(os.path.join(self.raw_folder, label_file))
        
        
        
        return mura_data, data_resized, digits

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, digit = self.data[index], self.targets[index], self.digits[index]
        #Change img to numpy and range to [0,255] -- do this only if image is normalized
        # img = np.uint8((img*255).numpy())
        
        #doing this so that it is consistent with all other datasets
        # to return a PIL Imagedata[torch.randperm(data.shape[0]),:,:]
        # img = Image.fromarray(img.numpy(), mode='L')
        # target = Image.fromarray(target.numpy(), mode='L')
        
        img = torch.Tensor(img.unsqueeze(0))
        target = torch.Tensor(target.unsqueeze(0))
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)            

        return dict({'img': img, 'target': target, 'digit': digit.item()})

    def __len__(self) -> int:
        return len(self.data)
        
    def _read_params(self, params):
        self.image_size = self.params['image_size']

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

class FashionMURA(MNIST_MURA):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``FashionMNIST/raw/train-images-idx3-ubyte``
            and  ``FashionMNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    mirrors = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"]

    resources = [
        ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
        ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
    ]
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    
