"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import numpy as np
import pandas as pd
import io
import random
import torch
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.sherlock_datasets import crop_widescreens
from lavis.datasets.vc_utils import  crop_image, draw_PersonX, draw_PersonY, draw_PersonZ, read_jsonl

def get_region_references(references, tokens):
    """
    get references in the order of it appears in the tokens
    Args:
        references (Dict): reference metadata info
        tokens (List): word tokens that also contains list IDs.

    Returns:
        List[Dict]: List of references sorted based on the order they appear in tokens.
    """
    parsed_reference = []
    region_references = []
    for w in tokens:
        if isinstance(w, list):
            for id in w:
                if id in references and id not in parsed_reference:
                    parsed_reference.append(id)
                    region_references.append(references[id])
    return region_references

def add_tags(tokens, refinfo, reorder_ids=True):
    # assumes text is tokenized with ids 
    text_tags = []
    seen_ids = []
    for w in tokens:
        if isinstance(w, list):
            w_tag = []
            for id in w:
                if id in refinfo:
                    if reorder_ids:
                        if id not in seen_ids:
                            seen_ids.append(id)
                        w_id = seen_ids.index(id)
                    else:
                        w_id = id
                    w_tag.append("[{}]".format(w_id))
            w = ','.join(w_tag)
        text_tags.append(w)
    return text_tags

def add_sherlock_tags(tokens, refinfo, reorder_ids=True):
    """ add_tags but deals with only one region"""
    text_tags = []
    for w in tokens:
        if isinstance(w, list): # asserts only 1 region in sherlock
            if id in refinfo:
                if reorder_ids:
                    w = "[0]"
                else:
                    w = "[{}]".format(id)
        text_tags.append(w) 
    return text_tags

"""
    {
        'index': 1737,
        'qa': 'Question: What is the engine of the plane suggesting about the scene? Answer: The engine of the plane suggests that this is an active and exciting scene, possibly an air show or aerobatics event. ',
        'question': 'Question: What is the engine of the plane suggesting about the scene?',
        'answer': 'Answer: The engine of the plane suggests that this is an active and exciting scene, possibly an air show or aerobatics event. ',
        'source': 'chatgpt_region1_ref0_1',
        'region': '0',
        'label': ['accept', 'reject', 'accept'],
        'image': 'VG_100K_2/2378381.jpg',
        'references': [
            {'name': '0',
            'boxes': [[114.4206848145, 171.4768066406, 136.3255767822, 193.6623840332]],
            'segms': None,
            'obj_guess': 'propeller',
            'tag': 'region'},
            {'name': '1',
            'boxes': [[109.6865615845, 179.2686004639, 129.4137573242, 202.239730835]],
            'segms': None,
            'obj_guess': 'motor',
            'tag': 'region'},
            {'name': '2',
            'boxes': [[114.892616272, 168.8507995605, 138.860748291, 185.9940795898]],
            'segms': None,
            'obj_guess': 'tank_(storage_vessel)',
            'tag': 'region'},
            {'name': '3',
            'boxes': [[152.9057159424, 176.6907348633, 162.3196868896, 185.0597229004]],
            'segms': None,
            'obj_guess': 'machine_gun',
            'tag': 'region'}
        ]
    }
"""
class UnifiedVQABinaryDataset(BaseDataset):
    """ UnifiedVQA Binary Label Classification Dataset
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, info):        
        super().__init__(vis_processor, text_processor, vis_root, [])
        
        self.annotation = []
        for ann_path in ann_paths:      
            data = read_jsonl(ann_path)
            self.annotation.extend(data)
        self.region_mode = info.get('region_mode', 'boxes')
        self.draw_others = info.get('draw_others', False)
        self.use_multi_class = info.get('multi_class', False) # multi_class: class determined by number of accepts
        self.use_generative_label = info.get('use_generative', False)   # label as string; for blip-2 / t5 style model
        self.text_only = info.get('text_only', False)

        print(self.__getitem__(0))

    def load_region_image(self, datum):
        image_path = os.path.join(self.vis_root, datum['image'])
        image = Image.open(image_path).convert('RGB')
        
        # draw region to image
    
        if datum['region'] is not None and datum['region'] != 'None':
            references = datum['references']
            references = {d['name']: d for d in references}
            regions = datum['region']
            if not isinstance(regions, list):
                regions = [regions]
            for ref_idx, ref in enumerate(regions[:3]):
                ref = str(ref)
                region = references[ref]['boxes'][0]
                if self.region_mode == 'crop':
                    image = crop_image(image, region)
                else:
                    if ref_idx == 0:
                        image = draw_PersonX(image, region, mode='boxes')
                    elif ref_idx == 1:
                        image = draw_PersonY(image, region, mode='boxes')
                    elif ref_idx == 2:
                        image = draw_PersonZ(image, region, mode='boxes')
        image = self.vis_processor(image)

        if self.text_only:
            image = torch.zeros_like(image)

        return image
    
    def get_label(self, datum):
        if self.use_multi_class:
            label = datum['label'].count('accept')

        # binary
        else:
            if 'reject_label' in datum:
                label = 1 - datum['reject_label']
            else:
                label = int(datum['label'].count('accept') < 2) 
        
        if self.use_generative_label:
            label = f"I give rating of {label}"
        
        return label

    def __getitem__(self, index):
        datum = self.annotation[index]
        image = self.load_region_image(datum)
        
        # fill ID tags and convert tokens to string
        # tag_fn = add_sherlock_tags if is_sherlock else add_tags
        text = self.text_processor(datum['qa'])
        
        label = self.get_label(datum)
        
        return {
                "image": image, 
                "text_input": text, 
                "label": label,
                'instance_id': datum['index'],
                "image_id": index
            }
    
    def collater(self, samples):
        image_list, text_list, label_list, image_id_list = [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            text_list.append(sample["text_input"])
            label_list.append(sample["label"])
            image_id_list.append(sample["image_id"])

        if self.use_generative_label:
            return {
                "image": torch.stack(image_list, dim=0),
                "text_input": text_list,
                "text_output": label_list,
                "image_id": torch.tensor(image_id_list, dtype=torch.int),
            }
        else:
            return {
                "image": torch.stack(image_list, dim=0),
                "text_input": text_list,
                "label": torch.LongTensor(label_list),
                "image_id": torch.tensor(image_id_list, dtype=torch.int),
            }

class UnifiedVQABinaryEvalDataset(UnifiedVQABinaryDataset):
    """ UnifiedVQA Binary Label Classification Dataset
    """
  
    def __getitem__(self, index):
        datum = self.annotation[index]

        image = self.load_region_image(datum)
        
        # fill ID tags and convert tokens to string
        # tag_fn = add_sherlock_tags if is_sherlock else add_tags
        text = self.text_processor(datum['qa'])
        
        # multi-class
        if 'label' in datum:
            label = self.get_label(datum)
            
            return {
                "image": image, 
                "text_input": text, 
                "label": label,
                "image_id": index,
                'instance_id': datum['index'],
            }

        else:            
            return {
                    "image": image, 
                    "text_input": text, 
                    "image_id": index,
                    'instance_id': datum['index'],
                }
    
    def collater(self, samples):
        image_list, text_list, label_list, instance_id_list, image_id_list = [], [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            text_list.append(sample["text_input"])
            if "label" in sample:
                label_list.append(sample["label"])
            instance_id_list.append(sample["instance_id"])
            image_id_list.append(sample["image_id"])

        output = {
            "image": torch.stack(image_list, dim=0),
            "text_input": text_list,
            "instance_id": instance_id_list,
            "image_id": torch.tensor(image_id_list, dtype=torch.int),
        }
        
        if len(label_list) > 0:
            if self.use_generative_label:
                output.update({"text_output": label_list})
            else:
                output.update({"label": torch.LongTensor(label_list)})
        
        return output
            
