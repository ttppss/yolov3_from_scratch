import torch
import torchvision
from torchvision.utils import draw_bounding_boxes


def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
    This describes how to combine these tensors of different sizes. We use lists.
    Note: this need not be defined in this Class, can be standalone.
    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
    """

    images = list()
    bboxes = list()
    labels = list()
    full_path = list()
    original_bboxes = list()

    for b in batch:
        images.append(b[0])
        bboxes.append(b[1])
        labels.append(b[2])
        full_path.append(b[-2])
        original_bboxes.append(b[-1])

    images = torch.stack(images, dim=0)

    return images, bboxes, labels, full_path, original_bboxes


def visualize_data_with_bbox(data_loader, classes):
    inputs = next(iter(data_loader))
    for i in range(inputs[0].shape[0]):
        labels_num = inputs[2][i]
        # draw_bounding_boxes only accepts string, so convert it from int to class name.
        labels = [classes[int(j)] for j in labels_num]
        inp = torch.transpose(inputs[0][i], 0, 2)
        mean = torch.FloatTensor([0.485, 0.456, 0.406])
        std = torch.FloatTensor([0.229, 0.224, 0.225])
        inp = (std * 255) * inp + mean * 255
        inp = torch.transpose(inp, 0, 2)
        
        # TODO: Boxes need to be in (xmin, ymin, xmax, ymax) format. Use torchvision.ops.box_convert to convert them
        inp = draw_bounding_boxes(inp.byte(), inputs[-1][i], labels, width=3)
        inputs[0][i] = inp
    out = torchvision.utils.make_grid(inputs[0], nrow=5)
    
    return out


def iou(bbox1, bbox2):
    """ get the IOU between two boxes"""
    # bbox1 and bbox2 are in the format of [x1, y1, x2, y2]
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    if x2 > x1 and y2 > y1:
        inter_area = (x2 - x1) * (y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        iou = inter_area / (bbox1_area + bbox2_area - inter_area)
    else:
        iou = 0
    return iou