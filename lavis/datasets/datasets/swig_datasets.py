from __future__ import print_function, division
import sys
import os
from tkinter import image_names
import torch
import numpy as np
import random
import csv
import skimage.io
import skimage.transform
import skimage.color
import skimage
import json
import cv2

from PIL import Image
from pathlib import Path

from lavis.datasets.datasets.base_dataset import BaseDataset

class SWIGDataset(BaseDataset):
    """SWIG dataset."""

    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, info):
        super().__init__(vis_processor, text_processor, vis_root, [])
        """
        Parameters:
            - ann_file : CSV file with annotations
            - class_list : CSV file with class list
        """
        
        split_name = {
            'train': 'train',
            'val': 'dev', 
            'test': 'test'
            }
        swig_path = info.get('swig_path')
        
        self.split = info.get('split')

        with open(f'{swig_path}/SWiG_jsons/imsitu_space.json') as f:
            all = json.load(f)
        self.verb_info = all['verbs']

        self.class_list = Path(swig_path) / "SWiG_jsons" / "train_classes.csv"
        with open(self.class_list, 'r') as file:
            self.classes, self.idx_to_class = self.load_classes(csv.reader(file, delimiter=','))

        self.verb_path = Path(swig_path) / "SWiG_jsons" / "verb_indices.txt"
        with open(self.verb_path, 'r') as f:
            self.verb_to_idx, self.idx_to_verb = self.load_verb(f)

        self.role_path = Path(swig_path) / "SWiG_jsons" / "role_indices.txt"
        with open(self.role_path, 'r') as f:
            self.role_to_idx, self.idx_to_role = self.load_role(f)

        self.annot_data = json.load(open(ann_paths[0]))

        self.image_data = self._read_annotations(self.annot_data, self.verb_info, self.classes)
        self.image_names = list(self.image_data.keys())

        self.image_to_image_idx = {}
        i = 0
        for image_name in self.image_names:
            self.image_to_image_idx[image_name] = i
            i += 1

        # verb_role
        self.verb_role = {verb: value['order'] for verb, value in self.verb_info.items()}

        # for each verb, the indices of roles in the frame.
        self.vidx_ridx = [[self.role_to_idx[role] for role in self.verb_role[verb]] for verb in self.idx_to_verb]

    def load_classes(self, csv_reader):
        result = {}
        idx_to_result = []
        for line, row in enumerate(csv_reader):
            line += 1
            class_name, class_id = row
            class_id = int(class_id)
            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
            idx_to_result.append(class_name.split('_')[0])

        return result, idx_to_result


    def load_verb(self, file):
        verb_to_idx = {}
        idx_to_verb = []

        k = 0
        for line in file:
            verb = line.split('\n')[0]
            idx_to_verb.append(verb)
            verb_to_idx[verb] = k
            k += 1
        return verb_to_idx, idx_to_verb


    def load_role(self, file):
        role_to_idx = {}
        idx_to_role = []

        k = 0
        for line in file:
            role = line.split('\n')[0]
            idx_to_role.append(role)
            role_to_idx[role] = k
            k += 1
        return role_to_idx, idx_to_role


    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, idx):
        img = self.load_image(idx)
        img = self.vis_processor(img)

        # annot = self.load_annotations(idx)
        verb = self.image_names[idx].split('/')[-1]
        verb = verb.split('_')[0]
        text_input = self.text_processor(verb)

        verb_idx = self.verb_to_idx[verb]
        verb_role = self.verb_info[verb]['order']
        verb_role_idx = [self.role_to_idx[role] for role in verb_role]
        sample = {
            'image': img, 
            'text_input': verb, 
            'image_id': idx,
            "instance_id": self.image_names[idx],
            'label': verb_idx, 
        }
    
        return sample

    def load_image(self, image_index):
        im = Image.open(os.path.join(self.vis_root, self.image_names[image_index]))
        im = im.convert('RGB')

        return im
    
    def _read_annotations(self, json, verb_orders, classes):
        result = {}

        for image in json:
            total_anns = 0
            verb = json[image]['verb']
            order = verb_orders[verb]['order']
            img_file = image
            result[img_file] = []
            for role in order:
                total_anns += 1
                [x1, y1, x2, y2] = json[image]['bb'][role]
                class1 = json[image]['frames'][0][role]
                class2 = json[image]['frames'][1][role]
                class3 = json[image]['frames'][2][role]
                if class1 == '':
                    class1 = 'blank'
                if class2 == '':
                    class2 = 'blank'
                if class3 == '':
                    class3 = 'blank'
                if class1 not in classes:
                    class1 = 'oov'
                if class2 not in classes:
                    class2 = 'oov'
                if class3 not in classes:
                    class3 = 'oov'
                result[img_file].append(
                    {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class1': class1, 'class2': class2, 'class3': class3})

            while total_anns < 6:
                total_anns += 1
                [x1, y1, x2, y2] = [-1, -1, -1, -1]
                class1 = 'Pad'
                class2 = 'Pad'
                class3 = 'Pad'
                result[img_file].append(
                    {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class1': class1, 'class2': class2, 'class3': class3})

        return result


    def name_to_label(self, name):
        return self.classes[name]

    def num_nouns(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)
    
    def collater(self, samples):
        data = samples
        image_list, text_input, instance_id_list, image_id_list = [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            text_input.append(sample["text_input"])
            instance_id_list.append(sample["instance_id"])
            image_id_list.append(sample["label"])
        
        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": text_input,
            "instance_id": instance_id_list,
            "image_id": torch.tensor(image_id_list, dtype=torch.int),
        }

class SWIGEvalDataset(SWIGDataset):
    """SWIG verb eval dataset."""

    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, info):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, info)
    
        self.text = [self.text_processor(t) for t in list(self.verb_to_idx.keys())]  # get verb list
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        self.annot_data = json.load(open(ann_paths[0]))
        for img_id, ann in self.annot_data.items():
            self.image.append(img_id) 
            txt_id = self.verb_to_idx[ann['verb']]
            img_id = len(self.img2txt)
            self.img2txt[img_id] = [txt_id]
    
    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        img = self.vis_processor(img)

        # annot = self.load_annotations(idx)
        verb = self.image_names[idx].split('/')[-1]
        verb = verb.split('_')[0]

        verb_idx = self.verb_to_idx[verb]
        verb_role = self.verb_info[verb]['order']
        sample = {
            'image': img, 
            'text_input': verb, 
            'image_id': idx,
            "instance_id": self.image_names[idx],
            'label': verb_idx, 

        }
    
        return sample

    def collater(self, samples):
        image_list, instance_id_list, image_id_list = [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            instance_id_list.append(sample["instance_id"])
            image_id_list.append(sample["image_id"])
        
        return {
            "image": torch.stack(image_list, dim=0),
            "instance_id": instance_id_list,
            "image_id": torch.tensor(image_id_list, dtype=torch.int),
        }