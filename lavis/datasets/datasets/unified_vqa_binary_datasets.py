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
from typing import List, Dict
import io
import random
import torch
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.vc_utils import draw_region, add_tags, tokens2str, read_jsonl

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
        
        self.use_multi_class = info.get('multi_class', False) # multi_class: class determined by number of accepts
        self.use_generative_label = info.get('use_generative', False)   # label as string; for blip-2 / t5 style model
        self.text_only = info.get('text_only', False)
        self.is_train = True


        verbalizations = pd.read_json(info['region_mapping'], lines=True).to_dict(orient='records')
        self.regions = {d['image']: d['region_locations'] for d in verbalizations}

        print(self.__getitem__(0))

    def get_regions(self, datum):

        if random.random() < 0.6 or not self.is_train:
            regions = datum['region']
        else:
            regions = random.sample(datum['region_all'], len(datum['region_all']))
        regions_info = [self.regions[datum['image']][ref] | {'name': ref} for ref in regions]   # List of bounding boxes and ids.
        return regions_info    
    
    def load_region_image(self, datum, regions):
        # if 'VG_100K' in datum['image'] or 'vcr1images' in datum['image']:
        image_path = os.path.join(self.vis_root, datum['image']) 
        # else:
        #     image_path = os.path.join(self.vis_root, 'vcr1images', datum['image'])
        image = Image.open(image_path).convert('RGB')
        for ref_idx, region in enumerate(regions[:5]):
            box = region['boxes'][0]
            image = draw_region(image, box, ref_idx, mode='boxes')

        return image
    
    def get_label(self, datum):
        if 'label' not in datum and 'reject_label' not in datum:
            return None
        
        if self.use_multi_class:
            label = datum['label'].count('accept')

        else:
            if 'reject_label' in datum:
                label = 1 - datum['reject_label']
            else:
                label = int(datum['label'].count('accept') < 2) 
        
        if self.use_generative_label:
            label = f"I give rating of {label}"
        
        return label
    
    def parse_region_tokens(self, tokens: List[str], regions: List[Dict]) -> str:
        """ 
        Replace region IDs in tokens in the order of `regions` with 0-index and parse into string.

        :param: tokens (List[str]): List of tokens to replace the region IDs
        :param: regions (List[Dict]): list of regions containing bounding box coordinates and the IDs.
        """

        region_dct = {}
        for idx, region in enumerate(regions):
            region_dct[int(region['name'])] = region | {'index': idx} 

        process_text = tokens2str(add_tags(tokens, region_dct, reorder_ids=True))

        return process_text

    def __getitem__(self, index):
        datum = self.annotation[index]

        regions = self.get_regions(datum)
        image = self.load_region_image(datum, regions)     
        
        image = self.vis_processor(image)   

        text = datum['question'] + ['Answer:'] + datum['answer'] + ['Rationale:'] + datum['rationale']
        text = self.parse_region_tokens(text, regions)
        text = self.text_processor(text)

        label = self.get_label(datum)
        qa_label = datum['qa_score']
        r_label = datum['r_score']
        
        return {
                "image": image, 
                "text_input": text,
                "label": label,
                "qa_label": qa_label,
                "r_label": r_label,
                'instance_id': datum['index'],
                "image_id": index
            }
    
    def collater(self, samples):
        image_list, text_list, label_list, qa_label_list, r_label_list,  image_id_list = [], [], [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            text_list.append(sample["text_input"])
            label_list.append(sample["label"])
            qa_label_list.append(sample["qa_label"])
            r_label_list.append(sample["r_label"])
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
                "qa_label": torch.FloatTensor(qa_label_list),
                "r_label": torch.FloatTensor(r_label_list),
                "image_id": torch.tensor(image_id_list, dtype=torch.int),
            }

class UnifiedVQABinaryEvalDataset(UnifiedVQABinaryDataset):
    """ UnifiedVQA Binary Label Classification Dataset
    """
    
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, info):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, info)
        self.is_train = False
  
    def __getitem__(self, index):
        datum = self.annotation[index]

        regions = self.get_regions(datum)
        image = self.load_region_image(datum, regions)        
        image = self.vis_processor(image)

        text = datum['question'] + ['Answer:'] + datum['answer'] + ['Rationale:'] + datum['rationale']
        text = self.parse_region_tokens(text, regions)
        text = self.text_processor(text)

        # multi-class
        label = self.get_label(datum)
        if label is None:
            return {
                    "image": image, 
                    "text_input": text, 
                    "image_id": index,
                    'instance_id': datum['index'],
                }
    
        else:
            label = self.get_label(datum)
            
            return {
                "image": image, 
                "text_input": text, 
                "label": label,
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
            
