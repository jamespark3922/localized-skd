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


def tokens2str(tokens):
    return ' '.join([str(w) for w in  tokens])

def get_references(datum, is_train=True):
    references = {}
    if isinstance(datum['references'], dict):
        references = datum['references'] 
    elif isinstance(datum['references_in_input'], dict):
        references = datum['references_in_input']
    references = {int(k): v for k,v in references.items()} # change key to int...
    return references

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

class UnifiedVQARankDataset(BaseDataset):
    """ UnifiedVQA Training Dataset with Ranking Objective
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, info):        
        super().__init__(vis_processor, text_processor, vis_root, [])
        
        self.annotation = []
        for ann_path in ann_paths:
            data = read_jsonl(ann_path)
            self.annotation.extend(data)
        self.region_mode = info.get('region_mode', 'boxes')
        self.draw_others = info.get('draw_others', False)
 
    def __getitem__(self, index):
        datum = self.annotation[index]
        
        # load image
        is_sherlock = datum['source'] == 'sherlock'
        if 'VG_100K' in datum['image']:
            image_path = os.path.join(self.vis_root, datum['image']) 
        else:
            image_path = os.path.join(self.vis_root, 'vcr1images', datum['image'])
        image = Image.open(image_path).convert('RGB')
        
        # draw region to image
        if 'metadata' in datum:
            meta = json.load(open(os.path.join(self.vis_root, 'vcr1images', datum['metadata'])))
            references = {}
            for r in range(len(meta['boxes'])):
                references[r] = {'boxes': [meta['boxes'][r]], 'segms': [meta['segms'][r]], 'obj_guess': meta['names'][r]}
        else:
            references = get_references(datum, is_train=True)
        full_text = [c for choice in datum['choices'] for c in choice]
        region_references = get_region_references(references, full_text)
        region_references = region_references[:3] # keep only 3 references for now
        for ref_idx, ref in enumerate(region_references):
            mode = 'boxes' if is_sherlock else self.region_mode
            region = ref[mode][0]
            if self.region_mode == 'crop':
                image = crop_image(image, region)
            else:
                if ref_idx == 0:
                    image = draw_PersonX(image, region, mode=mode)
                elif ref_idx == 1:
                    image = draw_PersonY(image, region, mode=mode)
                elif ref_idx == 2:
                    image = draw_PersonZ(image, region, mode=mode)
        image = self.vis_processor(image)
        
        # fill ID tags and convert tokens to string
        tag_fn = add_sherlock_tags if is_sherlock else add_tags
        texts = datum['choices']        
        texts = [self.text_processor(tokens2str(tag_fn(text, references))) for text in texts]
                    
        data = {"image": image, "text_input": texts, "image_id": index}
        
        # output
        if "scores" in datum:  # rating scores
            data.update({"scores": datum['scores']})
        if "label" in datum:  # mc label
            data.update({"label": datum["label"]})

        return data
    
    def collater(self, samples):
        image_list, text_list, score_list, label_list, image_id_list = [], [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            text_list.append(sample["text_input"])
            image_id_list.append(sample["image_id"])
            if "scores" in sample:
                score_list.append(sample["scores"])
            if "label" in sample:
                label_list.append(sample["label"])
        
        data = {
            "image": torch.stack(image_list, dim=0),
            "text_input": text_list,
            "image_id": torch.tensor(image_id_list, dtype=torch.int),
        }
        
        if len(score_list) > 0:
            data.update({"scores": torch.FloatTensor(score_list)})
        if len(label_list) > 0:
            data.update({"label": torch.tensor(label_list, dtype=torch.long)})
      
        return data
            

class UnifiedVQARankGenerativeDataset(BaseDataset):
    """ UnifiedVQA Rank Task with Generative Objective (e.g. for T5 models.)
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, info):        
        super().__init__(vis_processor, text_processor, vis_root, [])
        
        self.annotation = []
        for ann_path in ann_paths:      
            data = read_jsonl(ann_path)
            self.annotation.extend(data)
        self.region_mode = info.get('region_mode', 'boxes')
        self.draw_others = info.get('draw_others', False)
  
    def __getitem__(self, index):
        datum = self.annotation[index]
        
        # load image
        is_sherlock = datum['source'] == 'sherlock'
        if 'VG_100K' in datum['image']:
            image_path = os.path.join(self.vis_root, datum['image']) 
        else:
            image_path = os.path.join(self.vis_root, 'vcr1images', datum['image'])
        image = Image.open(image_path).convert('RGB')
        
        # draw region to image
        if 'metadata' in datum:
            meta = json.load(open(os.path.join(self.vis_root, 'vcr1images', datum['metadata'])))
            references = {}
            for r in range(len(meta['boxes'])):
                references[r] = {'boxes': [meta['boxes'][r]], 'segms': [meta['segms'][r]], 'obj_guess': meta['names'][r]}
        else:
            references = get_references(datum, is_train=True)
        full_text = [c for choice in datum['choices'] for c in choice]
        region_references = get_region_references(references, full_text)
        region_references = region_references[:3] # keep only 3 references for now
        for ref_idx, ref in enumerate(region_references):
            mode = 'boxes' if is_sherlock else self.region_mode
            region = ref[mode][0]
            if self.region_mode == 'crop':
                image = crop_image(image, region)
            else:
                if ref_idx == 0:
                    image = draw_PersonX(image, region, mode=mode)
                elif ref_idx == 1:
                    image = draw_PersonY(image, region, mode=mode)
                elif ref_idx == 2:
                    image = draw_PersonZ(image, region, mode=mode)
        image = self.vis_processor(image)
        
        # fill ID tags and convert tokens to string
        tag_fn = add_sherlock_tags if is_sherlock else add_tags
        texts = datum['choices']        
        texts = [self.text_processor(tokens2str(tag_fn(text, references))) for text in texts]
        scores = datum['scores']
        
        return {
                "image": image, 
                "text_input": texts, 
                "scores": scores,
                "image_id": index}
    
    def collater(self, samples):
        image_list, text_list, score_list, image_id_list = [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            text_list.append(sample["text_input"])
            score_list.append(sample["scores"])
            image_id_list.append(sample["image_id"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": text_list,
            "scores": torch.FloatTensor(score_list),
            "image_id": torch.tensor(image_id_list, dtype=torch.int),
        }