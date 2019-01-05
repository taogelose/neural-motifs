"""
Visualization script. I used this to create the figures in the paper.

WARNING: I haven't tested this in a while. It's possible that some later features I added break things here, but hopefully there should be easy fixes. I'm uploading this in the off chance it might help someone. If you get it to work, let me know (and also send a PR with bugs/etc)
"""

from dataloaders.visual_genome import VGDataLoader, VG
from lib.rel_model import RelModel
import numpy as np
import torch

from config import ModelConfig
from lib.pytorch_misc import optimistic_restore
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE
from lib.fpn.box_utils import bbox_overlaps
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import os
from functools import reduce

conf = ModelConfig()
train, val, test = VG.splits(num_val_im=conf.val_size)
if conf.test:
    val = test

train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)


############################################ HELPER FUNCTIONS ###################################

def get_cmap(N):
    import matplotlib.cm as cmx
    import matplotlib.colors as colors
    """Returns a function that maps each index in 0, 1, ... N-1 to a distinct RGB color."""
    color_norm = colors.Normalize(vmin=0, vmax=N - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')

    def map_index_to_rgb_color(index):
        pad = 40
        return np.round(np.array(scalar_map.to_rgba(index)) * (255 - pad) + pad)

    return map_index_to_rgb_color


cmap = get_cmap(len(train.ind_to_classes) + 1)


def load_unscaled(fn):
    """ Loads and scales images so that it's 1024 max-dimension"""
    image_unpadded = Image.open(fn).convert('RGB')
    im_scale = 1024.0 / max(image_unpadded.size)

    image = image_unpadded.resize((int(im_scale * image_unpadded.size[0]), int(im_scale * image_unpadded.size[1])),
                                  resample=Image.BICUBIC)
    return image


font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', 32)


def draw_box(draw, boxx, cls_ind, text_str):
    box = tuple([float(b) for b in boxx])
    if '-GT' in text_str:
        color = (255, 128, 0, 255)
    else:
        color = (0, 128, 0, 255)

    color = tuple([int(x) for x in cmap(cls_ind)])

    # draw the fucking box
    draw.line([(box[0], box[1]), (box[2], box[1])], fill=color, width=8)
    draw.line([(box[2], box[1]), (box[2], box[3])], fill=color, width=8)
    draw.line([(box[2], box[3]), (box[0], box[3])], fill=color, width=8)
    draw.line([(box[0], box[3]), (box[0], box[1])], fill=color, width=8)

    # draw.rectangle(box, outline=color)
    w, h = draw.textsize(text_str, font=font)

    x1text = box[0]
    y1text = max(box[1] - h, 0)
    x2text = min(x1text + w, draw.im.size[0])
    y2text = y1text + h
    print("drawing {}x{} rectangle at {:.1f} {:.1f} {:.1f} {:.1f}".format(
        h, w, x1text, y1text, x2text, y2text))

    draw.rectangle((x1text, y1text, x2text, y2text), fill=color)
    draw.text((x1text, y1text), text_str, fill='black', font=font)
    return draw


def mid_of_2point(point1, point2):
    """read two tuple, return a tuple"""
    midx = (point1[0] + point2[0]) / 2
    midy = (point1[1] + point2[1]) / 2
    return (midx, midy)


def draw_relation(draw, box1, box2, cls_ind, text_str):
    box1 = tuple([float(b) for b in box1])
    box2 = tuple([float(b) for b in box2])
    if '-GT' in text_str:
        color = (255, 128, 0, 255)
    else:
        color = (0, 128, 0, 255)

    color = tuple([int(x) for x in cmap(cls_ind)])

    # draw the fucking relation
    mid_point1 = mid_of_2point((box1[0], box1[1]), (box1[2], box1[3]))
    mid_point2 = mid_of_2point((box2[0], box2[1]), (box2[2], box2[3]))
    draw.line([mid_point1, mid_point2], fill=color, width=8)

    text_point = mid_of_2point(mid_point1, mid_point2)
    draw.text(text_point, text_str, fill='black', font=font)
    return draw


def val_epoch():
    for val_b, batch in enumerate(tqdm(val_loader)):
        val_batch(conf.num_gpus * val_b, batch)


cnt = 0


def val_batch(batch_num, b, thrs=(20, 50, 100)):
    # if conf.num_gpus == 1:
    #     det_res = [det_res]
    assert conf.num_gpus == 1
    gt_entry = {
        'gt_classes': val.gt_classes[batch_num].copy(),
        'gt_relations': val.relationships[batch_num].copy(),
        'gt_boxes': val.gt_boxes[batch_num].copy(),
    }

    theimg = load_unscaled(val.filenames[batch_num])
    theimg2 = theimg.copy()
    draw2 = ImageDraw.Draw(theimg2)

    # Draw the relation
    has_relation = {}
    for rels_id in range(len(gt_entry['gt_relations'])):
        box1_id = gt_entry['gt_relations'][rels_id][0]
        box2_id = gt_entry['gt_relations'][rels_id][1]
        has_relation[box1_id] = has_relation.get(box1_id, 0) + 1
        has_relation[box2_id] = has_relation.get(box2_id, 0) + 1

        box1 = gt_entry['gt_boxes'][box1_id]
        box2 = gt_entry['gt_boxes'][box2_id]
        draw2 = draw_relation(draw2, box1, box2,
                              cls_ind=gt_entry['gt_relations'][rels_id][2],
                              text_str=val.ind_to_predicates[gt_entry['gt_relations'][rels_id][2]])
    # Draw the bbox
    for obj_id in range(len(gt_entry['gt_classes'])):
        if has_relation.get(obj_id, 0):
            draw2 = draw_box(draw2, gt_entry['gt_boxes'][obj_id],
                             cls_ind=gt_entry['gt_classes'][obj_id],
                             text_str=val.ind_to_classes[gt_entry['gt_classes'][obj_id]])

    global cnt
    cnt += 1
    pathname = 'qualitative'
    if not os.path.exists(pathname):
        os.mkdir(pathname)
    theimg.save(os.path.join(pathname, 'img' + str(cnt) + '.jpg'), quality=100, subsampling=0)
    theimg2.save(os.path.join(pathname, 'imgbox' + str(cnt) + '.jpg'), quality=100, subsampling=0)


mAp = val_epoch()
