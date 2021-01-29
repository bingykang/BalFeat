import os
import io
import torch
import random
import numpy as np
import torchvision.datasets as datasets

from PIL import Image

def load_fnames(txtfile, with_label=False):
    keys = []
    with open(txtfile) as f:
        for line in f:
            key = line.split()[0]
            if key.startswith('train/'):
                key = key.replace('train/', '/')
            if with_label:
                keys.append((key, int(line.split()[1])))
            else:
                keys.append(key)
    return keys

class Dataset(datasets.VisionDataset):
    def __init__(self, root, subset=None, clsset=None,
                 transform=None, target_transform=None):
        super(Dataset, self).__init__(root, transform=transform,
                                      target_transform=target_transform)
        print(self.root)
        self.loader = datasets.folder.default_loader
        self.extensions = datasets.folder.IMG_EXTENSIONS
        samples = load_fnames(subset, with_label=True) if subset else None
        self.classes, self.class_to_idx, self.samples = self.setup(root, samples)

        # Take care of cls set
        if clsset:
            print('=> generating cls subset: ', clsset)
            # Get subset of classes 
            with open(clsset) as f:
                cs = [l.split()[0].strip('/') for l in f]
            
            # reset classes
            new_classes = sorted(cs)
            new_class_to_idx = {c: i for i, c in enumerate(new_classes)}

            tmp = []
            for s in self.samples:
                old_class = self.classes[s[1]]
                if old_class in new_classes:
                    tmp.append((s[0], new_class_to_idx[old_class]))
            self.samples = tmp
            self.classes = new_classes
            self.class_to_idx = new_class_to_idx
            print('=> Dataset Size: ', len(self.samples))

        self.targets = [s[1] for s in self.samples]

    def setup(self, root, samples):
        raise NotImplementedError

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(os.path.join(self.root, path.lstrip('/')))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
    
    def __len__(self):
        return len(self.samples)
    
    @property
    def num_classes(self):
        return len(self.classes)

    def get_percls_idxs(self):
        percls_idxs = [[] for _ in range(self.num_classes)]
        for i, s in enumerate(self.samples):
            percls_idxs[s[-1]].append(i)
        return percls_idxs

    def get_cls_cnts(self):
        if hasattr(self, 'cls_cnts'):
            return self.cls_cnts
        self.cls_cnts = np.zeros(self.num_classes)
        for s in self.samples:
            self.cls_cnts[s[-1]] += 1
        return self.cls_cnts


class ImageFolder(Dataset):
    def setup(self, root, samples):
        classes, class_to_idx = self._find_classes(root)
        if not samples: 
            samples = datasets.folder.make_dataset(root, 
                class_to_idx, self.extensions, None)
            samples = [(s[0].replace(root, ''), s[1]) for s in samples]
        return classes, class_to_idx, samples

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class Inaturalist(Dataset):
    def __init__(self, root, subset=None, clsset=None,
                 transform=None, target_transform=None):
        assert subset is not None, 'subset must be provided'
        assert clsset is None, 'clsset must be None'
        super(Inaturalist, self).__init__(root, subset, None,
                                          transform, target_transform)
    
    def setup(self, root, samples):
        assert samples is not None 
        classes = list(set([str(s[-1]) for s in samples]))
        classes.sort(key=lambda x: int(x))
        class_to_idx = {c: int(c) for c in classes}
        return classes, class_to_idx, samples


if __name__ == "__main__":
    import torchvision.transforms as transforms
    root = '/opt/tiger/bykang/inat18'
    subset = '/opt/tiger/bykang/mocoinat/data/iNaturalist18_train.txt'

    normalize = transforms.Normalize(mean=[0.466, 0.471, 0.380],
                                     std=[0.195, 0.194, 0.192])
    tt = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    dataset = Inaturalist(root, subset, transform=tt)
    import pdb; pdb.set_trace()
