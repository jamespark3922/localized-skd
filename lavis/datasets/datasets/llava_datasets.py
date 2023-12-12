"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
from tqdm import tqdm 
import numpy as np
import pandas as pd
import io
from typing import List, Dict
import random
import torch
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.sherlock_datasets import crop_widescreens

"""
    {
        'qa': 'Question: What is the engine of the plane suggesting about the scene? Answer: The engine of the plane suggests that this is an active and exciting scene, possibly an air show or aerobatics event. ',
        'qar': {
            'question': 'What is the engine of the plane suggesting about the scene?',
            'answer': 'The engine of the plane suggests that this is an active and exciting scene, possibly an air show or aerobatics event. ',
            'rationale': 'rationale',
        }
        'index': 'chatgpt_region1_ref0_1',
        'region': '0',
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
class LLAVADataset(BaseDataset):
    """
    LLAVA Dataset in conversation
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, info):        
        super().__init__(vis_processor, text_processor, vis_root, [])
        
        self.annotation = []
        for ann_path in ann_paths:
            data = []
            
            with open(ann_path) as f:      
                data = json.load(f)
                for d in data:
                    assert len(d['conversations']) % 2 == 0
                    for i in range(0, len(d['conversations']), 2):
                        text_input = d['conversations'][i]['value'].replace('\n','').replace('<image>','')
                        text_output = d['conversations'][i+1]['value'].replace('\n','').replace('<image>','') 
                        annot = {
                            "image": d["image"],
                            "text_input": text_input,
                            "text_output": text_output
                        }
                        self.annotation.append(annot)

        # breakpoint()
        self.is_train = True
    
    def __getitem__(self, index):
        datum = self.annotation[index]

        image = Image.open(os.path.join(self.vis_root, 'COCO_train2014_' + datum['image'])).convert('RGB')
        image = self.vis_processor(image)   

        text_input = self.text_processor(datum['text_input'])
        text_output = self.text_processor(datum['text_output'], add_prompt=False)
        full_text_input = f"{text_input} {text_output}" 

        return {
                "image": image, 
                "text_input": text_input,
                "text_output": text_output,
                "full_text_input": full_text_input, 
                'instance_id': str(index),
                "image_id": index
        }
    
    def collater(self, samples):
        image_list, text_input, full_text_input, text_output, instance_id_list, image_id_list = [], [], [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            text_input.append(sample["text_input"])
            full_text_input.append(sample["full_text_input"])
            text_output.append(sample["text_output"])
            instance_id_list.append(sample["instance_id"])
            image_id_list.append(sample["image_id"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": text_input,
            "full_text_input": full_text_input,
            "text_output": text_output,
            "instance_id": instance_id_list,
            "image_id": torch.tensor(image_id_list, dtype=torch.int),
        }

class LLAVAContrastiveDataset(LLAVADataset):

    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, info):        
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, info)
    
    def __getitem__(self, index):
        datum = self.annotation[index]

        image = Image.open(os.path.join(self.vis_root, 'COCO_train2014_' + datum['image'])).convert('RGB')
        image = self.vis_processor(image)   
        text_input = self.text_processor(f"{datum['text_input']} Answer: {datum['text_output']}" )

        return {
                "image": image, 
                "text_input": text_input,
                'instance_id': str(index),
                "image_id": index
        }
    
    def collater(self, samples):
        image_list, text_input, full_text_input, text_output, instance_id_list, image_id_list = [], [], [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            text_input.append(sample["text_input"])
            instance_id_list.append(sample["instance_id"])
            image_id_list.append(sample["image_id"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": text_input,
            "instance_id": instance_id_list,
            "image_id": torch.tensor(image_id_list, dtype=torch.int),
        }