"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import random
from typing import Dict, List, Optional
from tqdm import tqdm
from pathlib import Path
from functools import partial

import torch
import io
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.vc_utils import read_jsonl, draw_region, tokens2str, add_tags, add_sherlock_tags

def crop_widescreens(image):
    width, height = image.size
    if width >= height:
        im1 = {'height': height, 'width': height, 'left': 0, 'top': 0}
        im2 = {'height': height, 'width': height, 'left': width-height, 'top': 0}
    else:
        im1 = {'height': width, 'width': width, 'left': 0, 'top': 0}
        im2 = {'height': width, 'width': width, 'left': 0, 'top': height-width}
    regions = [image.crop((bbox['left'], bbox['top'], bbox['left'] + bbox['width'], bbox['top'] + bbox['height'])) for bbox in [im1, im2]]
    return regions

class UnifiedGenerativeDataset(BaseDataset):
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
        
        self.reorder_regions = info.get('reorder_regions', True)
        self.use_object_tags = info.get('use_object_tags', False)
        self.draw_others = info.get('draw_others', False)
        self.is_mc = info.get('is_mc', False)

        self.is_train = True
        # breakpoint()

        print(self.__getitem__(0))
    
    def get_regions(self, datum):

        regions_info = [{'name': ref } | val for ref, val  in datum['references'].items()]

        # region augmentation
        if self.is_train:
            regions_info = random.sample(regions_info, len(regions_info))

        return regions_info

    def load_region_image(self, datum, regions):
        # if 'VG_100K' in datum['image'] or 'vcr1images' in datum['image']:
        image_path = os.path.join(self.vis_root, datum['image']) 
        # else:
        #     image_path = os.path.join(self.vis_root, 'vcr1images', datum['image'])
        image = Image.open(image_path).convert('RGB')
        for ref_idx, region in enumerate(regions):
            box = region['boxes'][0]
            image = draw_region(image, box, ref_idx, mode='boxes')

        return image

    def parse_region_tokens(self, tokens: List[str], regions: List[Dict], no_ids: Optional[bool] = False) -> str:
        """ 
        Replace region IDs in tokens in the order of `regions` with 0-index and parse into string.

        :param: tokens (List[str]): List of tokens to replace the region IDs
        :param: regions (List[Dict]): list of regions containing bounding box coordinates and the IDs.
        """

        region_dct = {}
        for idx, region in enumerate(regions):
            region_dct[int(region['name'])] = region | {'index': idx} 

        process_text = tokens2str(add_tags(tokens, 
                                           region_dct, 
                                           reorder_ids=True, 
                                           use_object_tags=self.use_object_tags, 
                                           no_ids=no_ids))

        return process_text

    def __getitem__(self, index, save_image=False):
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

        # debugging
        if save_image:
            image.save('region.jpg')
        image = self.vis_processor(image)

        # text processor with regions
        answer_prefix = datum.get('answer_prefix', 'Answer:')
        text_input = self.parse_region_tokens(datum['text_input'], regions)
        text_input = self.text_processor(f"{text_input} {answer_prefix}")

        text_output = self.parse_region_tokens(datum['text_output'], regions)
        text_output  = self.text_processor(text_output, add_prompt=False)

        scores = None
        label = datum.get('label', None)
 
        return {
            "image": image, 
            "text_input": text_input, 
            "text_output": text_output, 
            "regions": regions,
            "label": label, 
            "scores": scores, 
            "image_id": index, 
            "instance_id": datum['instance_id'],
        }
    
    def collater(self, samples):
        image_list, input_list, output_list, region_list, label_list, score_list, image_ids, instance_ids = [], [], [], [], [] , [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            input_list.append(sample["text_input"])
            output_list.append(sample["text_output"])
            region_list.append(sample["regions"])
            label_list.append(sample["label"])
            score_list.append(sample["scores"])
            image_ids.append(sample["image_id"])
            instance_ids.append(sample['instance_id'])            

        to_return = {
            "image": torch.stack(image_list, dim=0),
            "text_input": input_list,
            "text_output": output_list,
            "regions": region_list,
            "image_id": torch.tensor(image_ids, dtype=torch.int),
            "instance_id": instance_ids,
        }
        
        if None not in score_list:
            to_return.update({
                "scores": score_list
            })

        if None not in label_list:
            to_return.update({
                "label": torch.LongTensor(label_list)
            })

        return to_return

class UnifiedGenerativeEvalDataset(UnifiedGenerativeDataset):

    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, info):  
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, info)
        self.is_train = False

    def __getitem__(self, index):

        datum = self.annotation[index]       
        
        # get list of regions in order of processing for image and text
        regions = self.get_regions(datum)

        # load image with drawn region references
        image = self.load_region_image(datum, regions)
        image = self.vis_processor(image)

         
        answer_prefix = datum.get('answer_prefix', 'Answer:')
        text_input = self.text_processor(f"{self.parse_region_tokens(datum['text_input'], regions)} {answer_prefix}")
        # text_input = self.text_processor('What can you tell me about highlighted region?')
        # Perplexity based Multiple Choice Evaluation
        if self.is_mc:
            text_output = [self.text_processor(self.parse_region_tokens(t['text_output'], regions), add_prompt=False) for t in datum['choices']]
            scores = [t['score'] for t in datum['choices']]
        # Generative
        else:
            if all([isinstance(d, list) for d in datum['text_output']]):  # multiple outputs
                text_output = [self.text_processor(self.parse_region_tokens(t, regions), add_prompt=False) for t in datum['text_output']]
            else: # single output
                text_output  = self.text_processor(self.parse_region_tokens(datum['text_output'], regions), add_prompt=False)
            scores = None
        label = datum.get('label', None)

        return {
            "image": image, 
            "text_input": text_input, 
            "text_output": text_output, 
            "regions": regions,
            "label": label, 
            "scores": scores, 
            "image_id": index, 
            "instance_id": datum['instance_id'],
        }
        

class UnifiedContrastiveDataset(UnifiedGenerativeDataset):
    """
    Unified Training Dataset for Contrastive Learning.
    This is used to train BLIP2 Qformer model.

    Set `text_input` as concatenated text of `text_input` and `text_output`.
    
    Following Datasets are Supported:
        - VCR
            - Train: Multiple Choice
            - Eval: Multiple Choice
        - Sherlock
            - Train: Contrastive
            - Eval: Multiple Choice
        - VisualCOMET
            - Train: Contrastive
            - Eval: Multiple Choice

    """

    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, info):

        self.no_regions = info.get('no_regions', False)
        self.no_question = info.get('no_question', False)
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, info)
        print(self.__getitem__(0))
        # breakpoint()

    def __getitem__(self, index):
        """
            datum should always include the following:
            - question [List[str]]: question text in list form
            - references [Dict[str]]: {ref: 'boxes': [x1,y1,x2,y2,score], 'obj_guess': str } 
            - choices [List[Dict]]: [{'text': str, 'score': float}]
            - label [Optional[int]]: optional label for multiple choice
            - instance_id [str]: id to refer to

            Returns:
            - image [PIL.Image]: image
           !- text_input [List[str]]: Multiple choice text input concatenated with `text_output`
            - label [Optional[int]]: optional label for multiple choice task
            - scores [List[float]]: score for text output
            - image_id [int]: dataloader image id
            - instance_id [str]: instance id
        """
        
        datum = self.annotation[index]       
        
        regions = self.get_regions(datum)

        # load image with drawn region references
        if self.no_regions:
            image = self.load_region_image(datum, [])
        else:    
            image = self.load_region_image(datum, regions)
        image = self.vis_processor(image)

        # text processor with regions
        text_input = self.parse_region_tokens(datum['text_input'], regions, no_ids=self.no_regions)        
        # text_input = 'What can you tell me about highlighted region?'
        answer_prefix = datum.get('answer_prefix', 'Answer:')

        if self.no_question:
            text_input = ''
            answer_prefix = ''

        # Multiple Choice
        if self.is_mc:
            text_output = [self.parse_region_tokens(t['text_output'], regions, no_ids=self.no_regions) for t in datum['choices']]
            text_input = [self.text_processor(f"{text_input} {answer_prefix} {t}") for t in text_output]
            scores = [t['score'] for t in datum['choices']]
        
        # Contrastive Learning
        else:
            text_output = self.parse_region_tokens(datum['text_output'], regions, no_ids=self.no_regions)
            text_input = self.text_processor(f"{text_input} {answer_prefix} {text_output}")
            scores = None
        label = datum.get('label', None)

        return {
            "image": image, 
            "text_input": text_input, 
            "regions": regions,
            "label": label, 
            "scores": scores, 
            "image_id": index, 
            "instance_id": datum['instance_id'],
         }
    
    def collater(self, samples):

        image_list, input_list, region_list, label_list, score_list, image_ids, instance_ids = [], [], [], [] , [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            input_list.append(sample["text_input"])
            region_list.append(sample["regions"])
            label_list.append(sample["label"])
            score_list.append(sample["scores"])
            image_ids.append(sample["image_id"])
            instance_ids.append(sample['instance_id'])            

        to_return = {
            "image": torch.stack(image_list, dim=0),
            "text_input": input_list,
            "regions": region_list,
            "image_id": torch.tensor(image_ids, dtype=torch.int),
            "instance_id": instance_ids,
        }
        
        if score_list[0] is not None:
            to_return.update({
                "scores": score_list
            })

        if label_list[0] is not None:
            to_return.update({
                "label": torch.LongTensor(label_list)
            })
        

        return to_return