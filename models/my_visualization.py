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
import os, tempfile
from imageio import imread
from functools import reduce

conf = ModelConfig()
train, val, test = VG.splits(num_val_im=conf.val_size)
if conf.test:
    val = test

train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)

WIDTH = 1024
HEIGHT = 768

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


cmap = get_cmap(len(train.ind_to_classes) + len(train.ind_to_predicates) + 1)


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
    draw.line([(box[0], box[1]), (box[2], box[1])], fill=color, width=1)
    draw.line([(box[2], box[1]), (box[2], box[3])], fill=color, width=1)
    draw.line([(box[2], box[3]), (box[0], box[3])], fill=color, width=1)
    draw.line([(box[0], box[3]), (box[0], box[1])], fill=color, width=1)

    # draw.rectangle(box, outline=color)
    w, h = draw.textsize(text_str, font=font)

    x1text = box[0]
    y1text = max(box[1] - h, 0)
    x2text = min(x1text + w, draw.im.size[0])
    y2text = y1text + h
    # print("drawing {}x{} rectangle at {:.1f} {:.1f} {:.1f} {:.1f}".format(
    #     h, w, x1text, y1text, x2text, y2text))

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
    draw.line([mid_point1, mid_point2], fill=color, width=2)

    text_point = mid_of_2point(mid_point1, mid_point2)
    draw.text(text_point, text_str, fill=color, font=font)
    return draw


def val_epoch():
    for val_b, batch in enumerate(tqdm(val_loader)):
        if val_b > 1000:
            break
        val_batch(conf.num_gpus * val_b, batch)


cnt = 0
val_vocab = {
    'object_idx_to_name': val.ind_to_classes,
    'object_name_to_idx': {name: ind for ind, name in enumerate(val.ind_to_classes)},
    'pred_idx_to_name': val.ind_to_predicates,
    'pred_name_to_idx': {name: ind for ind, name in enumerate(val.ind_to_predicates)},
}


def draw_scene_graph(objs, triples, has_relation=None, vocab=None, **kwargs):
    """
    Use GraphViz to draw a scene graph. If vocab is not passed then we assume
    that objs and triples are python lists containing strings for object and
    relationship names.

    Using this requires that GraphViz is installed. On Ubuntu 16.04 this is easy:
    sudo apt-get install graphviz
    """
    output_filename = kwargs.pop('output_filename', 'graph.png')
    orientation = kwargs.pop('orientation', 'V')
    edge_width = kwargs.pop('edge_width', 6)
    arrow_size = kwargs.pop('arrow_size', 1.5)
    binary_edge_weight = kwargs.pop('binary_edge_weight', 1.2)
    ignore_dummies = kwargs.pop('ignore_dummies', True)

    if orientation not in ['V', 'H']:
        raise ValueError('Invalid orientation "%s"' % orientation)
    rankdir = {'H': 'LR', 'V': 'TD'}[orientation]

    if vocab is not None:
        # Decode object and relationship names
        assert torch.is_tensor(objs)
        assert torch.is_tensor(triples)
        objs_list, triples_list = [], []
        for i in range(objs.size(0)):
            objs_list.append(vocab['object_idx_to_name'][objs[i]])
        for i in range(triples.size(0)):
            s = triples[i, 0]
            # p = vocab['pred_name_to_idx'][triples[i, 1]]
            p = vocab['pred_idx_to_name'][triples[i, 1]]
            o = triples[i, 2]
            triples_list.append([s, p, o])
        objs, triples = objs_list, triples_list

    # General setup, and style for object nodes
    lines = [
        'digraph{',
        'graph [size="5,3",ratio="compress",dpi="300",bgcolor="transparent"]',
        'rankdir=%s' % rankdir,
        'nodesep="0.5"',
        'ranksep="0.5"',
        'node [shape="box",style="rounded,filled",fontsize="48",color="none"]',
        'node [fillcolor="lightpink1"]',
    ]
    # Output nodes for objects
    for i, obj in enumerate(objs):
        if (has_relation is not None and not has_relation.get(i, 0)) or (ignore_dummies and obj == '__image__'):
            continue
        lines.append('%d [label="%s"]' % (i, obj))

    # Output relationships
    next_node_id = len(objs)
    lines.append('node [fillcolor="lightblue1"]')
    for s, p, o in triples:
        if ignore_dummies and p == '__in_image__':
            continue
        lines += [
            '%d [label="%s"]' % (next_node_id, p),
            '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
                s, next_node_id, edge_width, arrow_size, binary_edge_weight),
            '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
                next_node_id, o, edge_width, arrow_size, binary_edge_weight)
        ]
        next_node_id += 1
    lines.append('}')

    # Now it gets slightly hacky. Write the graphviz spec to a temporary
    # text file
    ff, dot_filename = tempfile.mkstemp()
    with open(dot_filename, 'w') as f:
        for line in lines:
            f.write('%s\n' % line)
    os.close(ff)

    # Shell out to invoke graphviz; this will save the resulting image to disk,
    # so we read it, delete it, then return it.
    output_format = os.path.splitext(output_filename)[1][1:]
    os.system('dot -T%s %s > %s' % (output_format, dot_filename, output_filename))
    os.remove(dot_filename)
    # img = Image.open(output_filename)
    # os.remove(output_filename)
    # img = img.resize((WIDTH, HEIGHT))
    # img.save(output_filename)
    # return img


def val_batch(batch_num, b, thrs=(20, 50, 100)):
    # if conf.num_gpus == 1:
    #     det_res = [det_res]
    assert conf.num_gpus == 1

    # 5元组：第一个物体在本张图的id，第二个物体在本张图的id，第一个物体的类别，第二个物体的类别，俩个物体的关系类别
    # gt_classes: 本图的第一个物体对应的类别id， 第二个物体。。。  {本图第i个物体： 类别id}
    # gt_relations: [物体(本图id), 物体(本图id), 关系id]
    gt_entry = {
        'gt_classes': val.gt_classes[batch_num].copy(),
        'gt_relations': np.unique(val.relationships[batch_num].copy(), axis=0),
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

        # use diff color with bbox
        draw2 = draw_relation(draw2, box1, box2,
                              cls_ind=gt_entry['gt_relations'][rels_id][2] + len(train.ind_to_classes),
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
    theimg.save(os.path.join(pathname, 'img/img_' + str(cnt) + '.png'), quality=100, subsampling=0)
    theimg2.save(os.path.join(pathname, 'imgbox/imgbox_' + str(cnt) + '.png'), quality=100, subsampling=0)

    # Generate the png graph
    objs = torch.LongTensor(gt_entry['gt_classes'])
    triples = gt_entry['gt_relations']
    triples[:, [1, 2]] = triples[:, [2, 1]]          # 交换列的写法
    triples = torch.LongTensor(triples)
    draw_scene_graph(objs, triples, has_relation, vocab=val_vocab, orientation='V',
                     output_filename='qualitative/graph/graph_' + str(cnt) + '.png')


mAp = val_epoch()
