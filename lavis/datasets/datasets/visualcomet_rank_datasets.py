"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import numpy as np

import io
import torch
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.csv_datasets import CSVDataset
from lavis.datasets.datasets.sherlock_datasets import crop_widescreens
from lavis.datasets.vc_utils import  crop_image, draw_PersonX, draw_PersonY, draw_PersonZ

prompt_dict = {
                'before': 'PersonX needed to ', 
                'intent': 'PersonX wanted to ',
                'after': 'PersonX will most likely ',
                'ambience': 'PersonX seems to be ',
                'emotion': 'PersonX feels ',
                'identity': 'PersonX looks like ', 
                'action': 'PersonX is doing ',
            }
class VisualCometRankDataset(CSVDataset):
    """ VisualCOMET Training Dataset with Human Ranking Scores
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, info):
        """
        arrow_root (string): directory containing *.arrow files
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.region_mode = info.get('region_mode', 'bbox')
        self.draw_others = info.get('draw_others', False)
        self.event_only = info.get('event_only', False)                
 
    def __getitem__(self, index):
        
        datum = self.annotation[index]
        image_path = os.path.join(self.vis_root, datum['img_fn'])
        image = Image.open(image_path).convert('RGB')
        
        subject = datum['subject']
        
        meta_path = os.path.join(self.vis_root, datum['metadata_fn'])
        meta = json.load(open(meta_path))
        if self.region_mode == 'polygon':
            regions = meta['segms']
        else:
            regions = np.array(meta['boxes'])[:,:4].tolist()

        # draw person regions
        region = regions[subject]
        if self.region_mode == 'crop':
            image = crop_image(image, region)
        else:
            image = draw_PersonX(image, region, mode=self.region_mode)
        if self.draw_others:
            if datum.get('PersonY', None):
                pid = int(datum['PersonY'])
                region = regions[pid]
                image = draw_PersonY(image, region, mode=self.region_mode)
            if datum.get('PersonZ', None):
                pid = int(datum['PersonZ'])
                region = regions[pid]
                image = draw_PersonZ(image, region, mode=self.region_mode)
        image = self.vis_processor(image)

        # get text endings
        texts = [prompt_dict[datum['inf_type']] + datum['ending%d' % i]  for i in range(5)]
        texts = [self.text_processor(text) for text in texts]
        scores = [datum['answer%d' % (i+1)]-1 for i in range(5)] # subtract 1 to ensure 0-index

        return {"image": image, "text_input": texts, "scores": scores, "image_id": index}

    def collater(self, samples):
        image_list, text_list, score_list, image_id_list = [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            text_list.append(sample["text_input"])
            score_list.append(sample["scores"]),
            image_id_list.append(sample["image_id"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": text_list,
            "scores": torch.tensor(score_list),
            "image_id": torch.tensor(image_id_list, dtype=torch.int),
        }

class VisualCometRankEvalDataset(CSVDataset):
    """ VisualCOMET Eval Dataset with Human Ranking Scores
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, info):
        """
        arrow_root (string): directory containing *.arrow files
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.region_mode = info.get('region_mode', 'bbox')
        self.draw_others = info.get('draw_others', False)
        self.event_only = info.get('event_only', False)
                

    def __getitem__(self, index):
        
        datum = self.annotation[index]
        image_path = os.path.join(self.vis_root, datum['img_fn'])
        image = Image.open(image_path).convert('RGB')
        
        subject = datum['subject']
        
        meta_path = os.path.join(self.vis_root, datum['metadata_fn'])
        meta = json.load(open(meta_path))
        if self.region_mode == 'polygon':
            regions = meta['segms']
        else:
            regions = np.array(meta['boxes'])[:,:4].tolist()

        # draw person regions
        region = regions[subject]
        if self.region_mode == 'crop':
            image = crop_image(image, region)
        else:
            image = draw_PersonX(image, region, mode=self.region_mode)
        if self.draw_others:
            if datum.get('PersonY', None):
                pid = int(datum['PersonY'])
                region = regions[pid]
                image = draw_PersonY(image, region, mode=self.region_mode)
            if datum.get('PersonZ', None):
                pid = int(datum['PersonZ'])
                region = regions[pid]
                image = draw_PersonZ(image, region, mode=self.region_mode)
        image = self.vis_processor(image)

        # get text endings
        texts = [prompt_dict[datum['inf_type']] + datum['ending%d' % i]  for i in range(5)]
        texts = [self.text_processor(text) for text in texts]

        return {"image": image, "text_input": texts, "image_id": index}

    def collater(self, samples):
        image_list, text_list, image_id_list = [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            text_list.append(sample["text_input"])
            image_id_list.append(sample["image_id"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": text_list,
            "image_id": torch.tensor(image_id_list, dtype=torch.int),
        }
