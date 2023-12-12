"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import random
from typing import Dict, List
from tqdm import tqdm
from pathlib import Path
from functools import partial

import torch
import io
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.vc_utils import read_jsonl, draw_region, tokens2str

idx2letter = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
}

prompt = 'Choose the best letter option answering the question.'

def add_tags(tokens, refinfo, use_descriptions=False, use_object_tags=False, reorder_ids=True):
    """
    Replace ID tag (type: List) with the corresponding tags in string.
    - use_descriptions: Use region description to fill in the ID tags.
    - use_tags: Use guessed object to fill in the tags. Usually used for VCR train/evaluation (disabled if use_descriptions is True.)
    - reorder_ids: Assign IDs based on the order they appear. example: [2] hugs [0] -> [0] hugs [1]. (disabled if use_descriptions is True.)
    """
    # assumes text is tokenized with ids 
    text_tags = []
    seen_ids = []
    for w in tokens:
        if isinstance(w, list):
            w_tag = []
            for id in w:
                if id in refinfo:
                        
                    # ids in consectuive order
                    if reorder_ids:
                        if 'index' in refinfo[id]:
                            w_id = refinfo[id]['index']
                        else:
                            if id not in seen_ids:
                                seen_ids.append(id)
                            w_id = seen_ids.index(id)
                    else:
                        w_id = id
                    if use_object_tags:
                        w_tag.append("{}{}".format(refinfo[id]["obj_guess"], w_id))
                    else:
                        w_tag.append("[{}]".format(w_id))
            w = ' , '.join(w_tag)
        text_tags.append(w)
    return text_tags

class VCRGenerativeDataset(BaseDataset):
    '''
        Generative Dataset additionally supporting multiple choice evaluation.
        This is used to train BLIP2 language model.
        Following Datasets are Supported:
        - VCR-X
            - Train: Generative
            - Eval: Generative
        - VisualCOMET
            - Train: Generative
            - Eval:
                - Generative
                - Multiple Choice
    '''
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, info):  
        super().__init__(vis_processor, text_processor, vis_root, [])
        
        self.annotation = []
        for ann_path in ann_paths:
            with open(ann_path) as f:
                for line in tqdm(f):
                    data = json.loads(line.strip())
                    self.annotation.append(data)
            print('loaded', ann_path)
        
        self.draw_others = info.get('draw_others', False)
        self.is_mc = info.get('is_mc', False)

        # breakpoint()
        print(self.__getitem__(0))
    
    def get_regions(self, datum):

        regions_info = [{'name': ref } | val for ref, val  in datum['references'].items()]
        return regions_info

    def load_region_image(self, datum, regions):
        
        image_path = os.path.join(self.vis_root, datum['image']) 
        image = Image.open(image_path).convert('RGB')
        for region in regions:
            box = region['boxes'][0]
            image = draw_region(image, box, int(region['name']), mode='boxes')

        return image

    def parse_region_tokens(self, tokens: List[str], regions: List[Dict]) -> str:
        """ 
        Replace region IDs in tokens in the order of `regions` with 0-index and parse into string.

        :param: tokens (List[str]): List of tokens to replace the region IDs
        :param: regions (List[Dict]): list of regions containing bounding box coordinates and the IDs.
        """

        region_dct = {}
        for idx, region in enumerate(regions):
            region_dct[int(region['name'])] = region | {'index': idx} 

        process_text = tokens2str(add_tags(tokens, region_dct, reorder_ids=False, use_object_tags=True))

        return process_text

    def __getitem__(self, index):
        """
        Args:
            index (int): index of the sample.
            datum should always include the following:
            - question [List[str]]: question text in list form
            - references [Dict[str]]: {ref: 'boxes': [x1,y1,x2,y2,score], 'obj_guess': str } 
            - choices [List[Dict]]: [{'text': str, 'score': float}]
            - label [Optional[int]]: optional label for multiple choice
            - instance_id [str]: id to refer to
        Returns:
            - image [PIL.Image]: image
            - text_input [str]: Question text
            - text_output [List[str]]: Multiple choice text
            - label [Optional[int]]: optional label for multiple choice task
            - scores [List[float]]: score for text output
            - image_id [int]: dataloader image id
            - instance_id [str]: instance id
        """
            
        datum = self.annotation[index]       
        
        # get list of regions in order of processing for image and text
        regions = self.get_regions(datum)

        # load image with drawn region references
        image = self.load_region_image(datum, regions)
        image = self.vis_processor(image)

        # text processor with regions
        answer_prefix = datum.get('answer_prefix', 'Answer:')
        text_input = self.parse_region_tokens(datum['text_input'], regions)
        answer_choices = [self.parse_region_tokens(t['text_output'], regions) for t in datum['choices']]
        answer_choices = ' '.join([f'{idx2letter[idx]}) {t}' for idx, t in enumerate(answer_choices)])

        text_input = f"{prompt} Question: {text_input} {answer_prefix} {answer_choices}"
        text_output = idx2letter[datum['label']]

        label = datum['label']
 
        return {
            "image": image, 
            "text_input": text_input, 
            "text_output": text_output, 
            "label": label, 
            "image_id": index, 
            "instance_id": datum['instance_id'],
        }
    
    def collater(self, samples):
        image_list, input_list, output_list, label_list, image_ids, instance_ids = [], [], [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            input_list.append(sample["text_input"])
            output_list.append(sample["text_output"])
            label_list.append(sample["label"])
            image_ids.append(sample["image_id"])
            instance_ids.append(sample['instance_id'])            

        to_return = {
            "image": torch.stack(image_list, dim=0),
            "text_input": input_list,
            "text_output": output_list,
            "image_id": torch.tensor(image_ids, dtype=torch.int),
            "instance_id": instance_ids,
        }

        if label_list[0] is not None:
            to_return.update({
                "label": torch.LongTensor(label_list)
            })

        return to_return

class VCRGenerativeEvalDataset(VCRGenerativeDataset):

    def __getitem__(self, index):

        datum = self.annotation[index]       
        
        # get list of regions in order of processing for image and text
        regions = self.get_regions(datum)

        # load image with drawn region references
        image = self.load_region_image(datum, regions)
        image = self.vis_processor(image)

         
        # text processor with regions
        answer_prefix = datum.get('answer_prefix', 'Answer:')
        text_input = self.parse_region_tokens(datum['text_input'], regions)
        answer_choices = [self.parse_region_tokens(t['text_output'], regions) for t in datum['choices']]
        answer_choices = ' '.join([f'{idx2letter[idx]}) {t}' for idx, t in enumerate(answer_choices)])

        text_input = f"{prompt} Question: {text_input} {answer_prefix} {answer_choices}"
        text_output = [idx2letter[idx] for idx in range(4)]
        label = datum.get('label', None)

        return {
            "image": image, 
            "text_input": text_input, 
            "text_output": text_output, 
            "label": label, 
            "image_id": index, 
            "instance_id": datum['instance_id'],
        }