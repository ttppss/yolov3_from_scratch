import sys
# sys.path.append('/home/zinan/PycharmProjects/yolov3_from_scratch')
sys.path.append('/home/zinan/pycharmproject/yolov3_from_scratch')

import torch
import torchvision
from torch import Tensor
from torch.utils import data
from torch.utils.data import DataLoader

import os
from PIL import Image
from typing import List, Dict, Tuple, Any, Union, Optional
import xml.etree.ElementTree as ET
import collections
import numpy as np
from visdom import Visdom

from yacs.config import CfgNode
from torchvision import transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.utils import visualize_data_with_bbox, collate_fn
from config import config as cfg

VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor']


def build_vocdetection_dataset_v2(dataset_cfg: CfgNode):
    data_transforms = {
        'train': A.Compose([
            A.Resize(dataset_cfg.IMAGE_SIZE[0], dataset_cfg.IMAGE_SIZE[1]),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.RandomBrightnessContrast(),
            A.Normalize(),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
        'test': A.Compose([
            A.Resize(dataset_cfg.IMAGE_SIZE[0], dataset_cfg.IMAGE_SIZE[1]),
            A.Normalize(),
            ToTensorV2(),
        ]),
    }

    train_dataset = VOCDetection(dataset_cfg.DATA_ROOT, 'train', transforms=data_transforms['train'])
    val_dataset = VOCDetection(dataset_cfg.DATA_ROOT, 'val', transforms=data_transforms['test'])
    return train_dataset, val_dataset


data_transforms = {
        'train': A.Compose([
            A.LongestMaxSize(max_size=512, interpolation=1),
            A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, value=(0,0,0)),
            A.Resize(416, 416),
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(),
            A.Normalize(),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
        'test': A.Compose([
            A.LongestMaxSize(max_size=512, interpolation=1),
            A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, value=(0,0,0)),
            A.Resize(416, 416),
            A.Normalize(),
            ToTensorV2(),
        ]),
    }


class VOCDetection(data.Dataset):
    def __init__(self, root, image_set, use_difficult=False, transforms=None):
        super(VOCDetection, self).__init__()
        self.root = root
        self.image_set = image_set
        self.use_difficult = use_difficult
        self.transforms = transforms

        valid_sets = ['train', 'trainval', 'val']
        if self.image_set not in valid_sets:
            msg = '{} is not a valid dataset'.format(self.image_set)
            raise ValueError(msg)

        image_dir = os.path.join(self.root, 'JPEGImages')
        anno_dir = os.path.join(self.root, 'Annotations')

        if not os.path.isdir(self.root):
            raise RuntimeError('Root folder not found or corrupted, please download it first.')

        splits_dir = os.path.join(self.root, 'ImageSets/Main')
        split_file = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_file), 'r') as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + '.jpg') for x in file_names]
        self.annos = [os.path.join(anno_dir, x + '.xml') for x in file_names]
        assert (len(self.images) == len(self.annos))

        if self.transforms is None:
            normalize = T.Normalize(mean = [0.485, 0.456, 0.406],
                                     std = [0.229, 0.224, 0.225])
            if self.image_set != 'train':
                self.transforms = T.Compose([
                    # note: modified the resolution here for SSD300
                    T.Resize((416, 416)),
                    T.ToTensor(),
                    T.Normalize(mean = [0.485, 0.456, 0.406],
                                     std = [0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize((416, 416)),
                    T.ToTensor(),
                    T.Normalize(mean = [0.485, 0.456, 0.406],
                                     std = [0.229, 0.224, 0.225])
                ])

        else:
            self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        img_width, img_height = img.size
        tree = ET.parse(self.annos[index])
        file_name = tree.find('filename').text
        file_full_path = os.path.join(self.root, 'JPEGImages', file_name)
        objs = tree.findall('object')
        if not self.use_difficult:
            non_difficult_objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]
        else:
            non_difficult_objs = [obj for obj in objs]

        num_objs = len(non_difficult_objs)

        bboxes = np.zeros((num_objs, 4))
        gt_classes = np.zeros(num_objs)
        difficult = [int(obj.find('difficult').text) for obj in non_difficult_objs]

        for idx, obj in enumerate(non_difficult_objs):
            # TODO: may need to clip the value and make sure it is larger than 0.
            bbox = obj.find('bndbox')
            x0 = float(bbox.find('xmin').text)
            x1 = float(bbox.find('xmax').text)
            y0 = float(bbox.find('ymin').text)
            y1 = float(bbox.find('ymax').text)
            cls = int(VOC_CLASSES.index(obj.find('name').text.lower().strip()))

            # w_image_ratio = (x1 - x0) / img_width
            # h_image_ratio = (y1 - y0) / img_height
            # x_image_ratio = x0 / img_width + w_image_ratio / 2
            # y_image_ratio = y0 / img_height + h_image_ratio / 2

            bboxes[idx, :] = [x0, y0, x1, y1]
            gt_classes[idx] = cls

        img = np.asarray(img)
        transformed = self.transforms(image=img, bboxes=bboxes, class_labels=gt_classes)
        img = transformed['image']

        bboxes = transformed['bboxes']
        transformed_bboxes_original = []
        bboxes_in_ratio = []
        for bbox in bboxes:
            transformed_bboxes_original.append([b for b in bbox])
            box = [(bbox[0] + bbox[2]) / 2 / 416, (bbox[1] + bbox[3]) / 2 / 416,
                   (bbox[2] - bbox[0]) / 416, (bbox[3] - bbox[1]) / 416]
            bboxes_in_ratio.append(box)

        bboxes_in_ratio = torch.as_tensor(bboxes_in_ratio)  # [x1, y1, x2, y2], not normalized.
        transformed_bboxes_original = torch.as_tensor(transformed_bboxes_original)  # [cx, cy, w, h], normalized

        # get the ground truth tensor for each feature size.
        labels = {}
        for feature_size, anchors in cfg.ANCHORS_GROUP.items():
            labels[feature_size] = torch.zeros((feature_size, feature_size, 3, cfg.CLASS_NUM + 5))

            for bbox, cls in zip(bboxes_in_ratio, gt_classes):
                # get the iou between the bbox and all the anchors.
                ious = []
                for anchor in anchors:
                    iou = bbox_iou(bbox, anchor)
                    ious.append(iou)
                ious = torch.as_tensor(ious)

                # get the anchor index with the max iou.
                max_iou, max_iou_index = ious.max(0)

                # get the feature map index for the bbox.
                feature_map_index_x = int(bbox[0] * feature_size)
                feature_map_index_y = int(bbox[1] * feature_size)

                # get the label for the feature map.
                label = labels[feature_size][feature_map_index_y, feature_map_index_x, max_iou_index, :]
                label[0:4] = bbox
                label[4] = 1
                label[5 + int(cls)] = 1


        return img, bboxes_in_ratio, gt_classes, difficult, file_full_path, transformed_bboxes_original

    def __len__(self) -> int:
        return len(self.images)


if __name__ == '__main__':
    # alienware-desktop
    # root = '/home/zinan/dataset/demo/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'
    # alienware-home
    root = '/home/zinan/dataset/voc/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
    image_set = 'train'
    dataset = VOCDetection(root=root, image_set=image_set, transforms=data_transforms[image_set])
    print(dataset[0])
    print(cfg.ANCHORS_GROUP)
    
    # viz = Visdom()
    # out = visualize_data_with_bbox(loader, VOC_CLASSES)
    # viz.images(out, win='data_viewer', opts={'title': 'data viewer'})