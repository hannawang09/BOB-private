import os
from os.path import expanduser
from os.path import join as ospj
import json
import pickle
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision as tv
from collections import defaultdict
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, list_dir, list_files
import math

from utils import make_dirs

from os.path import join
import random
from PIL import ImageFilter, ImageOps


NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)
CLIP_NORM_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_NORM_STD = (0.26862954, 0.26130258, 0.27577711)



def is_vector_label(x):
    if isinstance(x, np.ndarray):
        return x.size > 1
    elif isinstance(x, torch.Tensor):
        return x.size().numel() > 1
    elif isinstance(x, int):
        return False
    else:
        raise TypeError(f"Unknown type {type(x)}")

def get_transforms(model_type):
    if model_type == "clip":
        norm_mean = CLIP_NORM_MEAN
        norm_std = CLIP_NORM_STD
    elif model_type == "imagenet":
        norm_mean = NORM_MEAN
        norm_std = NORM_STD

    aux_transform = tv.transforms.Compose([
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomApply(
            [
                tv.transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                )
            ],
            p=0.8,
        ),
        tv.transforms.RandomGrayscale(p=0.2),
        GaussianBlur(0.2),
        Solarization(0.2),
    ])
    train_transform = tv.transforms.Compose([
        tv.transforms.Lambda(lambda x: x.convert("RGB")),
        tv.transforms.RandAugment(),
        tv.transforms.RandomResizedCrop(
            224, 
            scale=(0.25, 1.0), 
            interpolation=tv.transforms.InterpolationMode.BICUBIC,
            antialias=None,
        ),
        aux_transform,
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(norm_mean, norm_std)
    ])

    test_transform = tv.transforms.Compose([
        tv.transforms.Lambda(lambda x: x.convert("RGB")),
        tv.transforms.Resize(
            224, 
            interpolation=tv.transforms.functional.InterpolationMode.BICUBIC
        ),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(norm_mean, norm_std)
    ])

    return train_transform, test_transform


class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __repr__(self):
        return "{}(p={}, radius_min={}, radius_max={})".format(
            self.__class__.__name__, self.p, self.radius_min, self.radius_max
        )

    def __call__(self, img):
        if random.random() <= self.p:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __repr__(self):
        return "{}(p={})".format(self.__class__.__name__, self.p)

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Flower_Dataset(Dataset):
    flower_names = [
        'pink_primrose', 'hard-leaved_pocket_orchid', 'canterbury_bells', 'sweet_pea', 'english_marigold',
        'tiger_lily', 'moon_orchid', 'bird_of_paradise', 'monkshood', 'globe_thistle',
        'snapdragon', "colt's_foot", 'king_protea', 'spear_thistle', 'yellow_iris',
        'globe-flower', 'purple_coneflower', 'peruvian_lily', 'balloon_flower', 'giant_white_arum_lily',
        'fire_lily', 'pincushion_flower', 'fritillary', 'red_ginger', 'grape_hyacinth',
        'corn_poppy', 'prince_of_wales_feathers', 'stemless_gentian', 'artichoke', 'sweet_william',
        'carnation', 'garden_phlox', 'love_in_the_mist', 'mexican_aster', 'alpine_sea_holly',
        'ruby-lipped_cattleya', 'cape_flower', 'great_masterwort', 'siam_tulip', 'lenten_rose',
        'barbeton_daisy', 'daffodil', 'sword_lily', 'poinsettia', 'bolero_deep_blue',
        'wallflower', 'marigold', 'buttercup', 'oxeye_daisy', 'common_dandelion',
        'petunia', 'wild_pansy', 'primula', 'sunflower', 'pelargonium',
        'bishop_of_llandaff', 'gaura', 'geranium', 'orange_dahlia', 'pink-yellow_dahlia',
        'cautleya_spicata', 'japanese_anemone', 'black-eyed_susan', 'silverbush', 'californian_poppy',
        'osteospermum', 'spring_crocus', 'bearded_iris', 'windflower', 'tree_poppy',
        'gazania', 'azalea', 'water_lily', 'rose', 'thorn_apple',
        'morning_glory', 'passion_flower', 'lotus', 'toad_lily', 'anthurium',
        'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose',
        'tree_mallow', 'magnolia', 'cyclamen_', 'watercress', 'canna_lily',
        'hippeastrum_', 'bee_balm', 'ball_moss', 'foxglove', 'bougainvillea',
        'camellia', 'mallow', 'mexican_petunia', 'bromelia', 'blanket_flower',
        'trumpet_creeper', 'blackberry_lily'
    ]
    def __init__(self, root, transform=None):
        self.data = []
        for (i, name) in enumerate(self.flower_names):
            for file in os.listdir(os.path.join(root, name)):
                self.data.append((os.path.join(root, name, file), i))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

class DatasetSynthImage(Dataset):
    def __init__(
        self, 
        synth_train_data_dir, 
        transform, 
        n_img_per_cls,
        classnames,
        real_train_fewshot_data_dir='', 
        is_pooled_fewshot=False, 
        real_data_multiplier=None,
        **kwargs
    ):
        self.synth_train_data_dir = synth_train_data_dir
        self.transform = transform
        self.image_paths = []
        self.image_labels = []
        self.is_real = []
        self.n_img_per_cls = n_img_per_cls
        self.is_pooled_fewshot = is_pooled_fewshot

        value_counts = defaultdict(int)
        for label, class_name in enumerate(classnames):
            for fname in os.listdir(ospj(synth_train_data_dir, class_name)):
                if not fname.endswith(".png"):
                    continue
                self.image_paths.append(
                    ospj(synth_train_data_dir, class_name, fname))
                self.image_labels.append(label)
                self.is_real.append(False)

        if is_pooled_fewshot:
            for label, class_name in enumerate(classnames):
                real_img_paths = os.listdir(
                    ospj(real_train_fewshot_data_dir, class_name))
                real_subset = [
                    ospj(
                        real_train_fewshot_data_dir, 
                        class_name, 
                        real_img_paths[i]
                    ) for i in range(len(real_img_paths))
                ]
                n_shot = len(real_subset)
                if real_data_multiplier is not None:
                    reps = real_data_multiplier
                else:   
                    reps = round(n_img_per_cls // n_shot)
                for i in range(reps):
                    self.image_paths.extend(real_subset)
                    self.image_labels.extend([label] * n_shot)
                    self.is_real.extend([True] * n_shot)
                
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_label = self.image_labels[idx]
        is_real = self.is_real[idx]
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = self.transform(image)

        if self.is_pooled_fewshot:
            return image, image_label, is_real
        else:
            return image, image_label

    def __len__(self):
        return len(self.image_paths)


def validation_data_loader(
    real_train_data_dir,
    dataset, 
    batch_size,
    model_type,
):

    _, test_transform = get_transforms(model_type)
    if dataset == 'cub':
        test_dataset = Cub2011(
            root=real_train_data_dir,
            train=True,
            transform=test_transform,
            download=False,
        )
        indices = pickle.load(open("../dataset/validation_splits/cub.pkl", "rb"))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
        assert len(test_dataset) == 2000
    elif dataset == 'imbalanced_cub':
        test_dataset = Cub2011(
            root=real_train_data_dir,
            train=False,
            transform=test_transform,
            download=False,
        )
        indices = pickle.load(open("../dataset/validation_splits/cub_test_validsplit.pkl", "rb"))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
        assert len(test_dataset) == 1000
    elif dataset == 'imbalanced_flower':
        test_dataset = Flower_Dataset(real_train_data_dir, transform=test_transform)
        indices = pickle.load(open("../dataset/validation_splits/flower_test_validsplit.pkl", "rb"))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
        assert len(test_dataset) == 102*5
    elif dataset == 'pet':
        test_dataset = tv.datasets.OxfordIIITPet(
            root=real_train_data_dir,
            split='trainval',
            target_types='category',
            download=False,
            transform=test_transform,
        )
        indices = pickle.load(open("../dataset/validation_splits/pet.pkl", "rb"))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
        assert len(test_dataset) == 592
    elif dataset == 'aircraft':
        test_dataset = tv.datasets.FGVCAircraft(
            root=real_train_data_dir,
            split='trainval',
            annotation_level='variant',
            transform=test_transform,
            download=True,
        )
        indices = pickle.load(open("../dataset/validation_splits/fgvc_aircraft.pkl", "rb"))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
        assert len(test_dataset) == 1600
    elif dataset == 'car':
        test_dataset = tv.datasets.StanfordCars(
            root=real_train_data_dir,
            split='train',
            transform=test_transform,
            download=False,
        )
        indices = pickle.load(open("../dataset/validation_splits/car.pkl", "rb"))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
        assert len(test_dataset) == 1960
    else:
        raise ValueError("Please specify a valid dataset.")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, 
        sampler=None,
        shuffle=False,
        prefetch_factor=4, pin_memory=False,
        num_workers=16) 

    return test_loader


def get_data_loader(
    real_test_data_dir,
    dataset, 
    batch_size,
    model_type,
):

    _, test_transform = get_transforms(model_type)
    if dataset == 'cub':
        test_dataset = Cub2011(
            root=real_test_data_dir,
            train=False,
            transform=test_transform,
            download=False,
        )
    elif dataset == 'imbalanced_cub':
        test_dataset = Cub2011(
            root=real_test_data_dir,
            train=False,
            transform=test_transform,
            download=False,
        )
        indices = pickle.load(open("../dataset/validation_splits/cub_test_testsplit.pkl", "rb"))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
    elif dataset == 'imbalanced_flowers':
        test_dataset = Flower_Dataset(real_test_data_dir, transform=test_transform)
        indices = pickle.load(open("../dataset/validation_splits/flower_test_testsplit.pkl", "rb"))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
    elif dataset == 'pet':
        test_dataset = tv.datasets.OxfordIIITPet(
            root=real_test_data_dir,
            split='test',
            target_types='category',
            download=False,
            transform=test_transform,
        )
    elif dataset == 'aircraft':
        test_dataset = tv.datasets.FGVCAircraft(
            root=real_test_data_dir,
            split='test',
            annotation_level='variant',
            transform=test_transform,
            download=True,
        )
    elif dataset == 'car':
        test_dataset = tv.datasets.StanfordCars(
            root=real_test_data_dir,
            split='test',
            transform=test_transform,
            download=False,
        )
    else:
        raise ValueError("Please specify a valid dataset.")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=16, pin_memory=False)

    return test_loader


def get_synth_train_data_loader(
    synth_train_data_dir,
    batch_size, 
    is_rand_aug,
    n_img_per_cls,
    real_train_fewshot_data_dir,
    is_pooled_fewshot,
    model_type,
    classnames,
    real_data_multiplier,
):
    train_transform, test_transform = get_transforms(model_type)

    train_dataset = DatasetSynthImage(
        synth_train_data_dir=synth_train_data_dir, 
        transform=train_transform if is_rand_aug else test_transform,
        n_img_per_cls=n_img_per_cls,
        real_train_fewshot_data_dir=real_train_fewshot_data_dir,
        is_pooled_fewshot=is_pooled_fewshot,
        classnames=classnames,
        real_data_multiplier=real_data_multiplier,
    ) 
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, 
        sampler=None,
        shuffle=is_rand_aug,
        num_workers=16, pin_memory=False,
    )
    return train_loader



