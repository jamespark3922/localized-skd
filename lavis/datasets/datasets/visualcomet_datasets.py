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
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.sherlock_datasets import crop_widescreens
from lavis.datasets.vc_utils import read_jsonl, crop_image, draw_PersonX, draw_PersonY, draw_PersonZ

def split_qa(qa):
    question = qa.split('?')[0]
    question = question+'?'
    answer = '?'.join(qa.split('?')[1:])
    
    return question, answer

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

def add_tags(tokens, refinfo, use_descriptions=False, reorder_ids=True):
    # assumes text is tokenized with ids 
    text_tags = []
    seen_ids = []
    for w in tokens:
        if isinstance(w, list):
            w_tag = []
            for id in w:
                if id in refinfo:
                    
                    # explicit description
                    if use_descriptions:
                        w_tag.append(refinfo[id]['description'])
                        
                    # id tags
                    else:
                        # ids in consectuive order
                        if reorder_ids:
                            if id not in seen_ids:
                                seen_ids.append(id)
                            w_id = seen_ids.index(id)
                        else:
                            w_id = id
                        w_tag.append("[{}]".format(w_id))
            w = ' , '.join(w_tag)
        text_tags.append(w)
    return text_tags

def add_sherlock_tags(tokens, refinfo, use_descriptions=False, reorder_ids=True):
    """ add_tags but deals with only one region"""
    text_tags = []
    for w in tokens:
        if isinstance(w, list):
            for id in refinfo: # only 1 region should be present in sherlock
                # explicit description
                if use_descriptions:
                    w = refinfo[id]['description']
                # id tags
                else:
                    if reorder_ids:
                        w = "[0]"
                    else:
                        w = "[{}]".format(id)
        text_tags.append(w) 
    return text_tags

class VisualCometDataset(BaseDataset):
    """ VisualComet Training Dataset with Generation Objective
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, info):        
        super().__init__(vis_processor, text_processor, vis_root, [])
        random.seed(42)

        self.region_mode = info.get('region_mode', 'boxes')
        
        self.annotation = []
        for ann_path in ann_paths:
            with open(ann_path) as f:
                for line in tqdm(f):
                    data = json.loads(line.strip())

                    # remove segms to reduce size
                    # if len(data['target_options']) > 0:
                    #     if self.region_mode == 'boxes':
                    #         for k,v in data['references'].items():
                    #             if 'segms' in v:
                    #                 v.pop('segms')
                        
                    #     # just randomly choose target option for training
                    #     qa_idx = random.choice(range(len(data['target_options'])))
                    #     data['target_options'] = [data.pop('target_options')[qa_idx]]
                    #     data['questions'] = [data.pop('questions')[qa_idx]]
                    #     data['answers'] = [data.pop('answers')[qa_idx]]
                    self.annotation.append(data)
            print('loaded', ann_path)
        
        self.draw_others = info.get('draw_others', False)
  
    def __getitem__(self, index):
        datum = self.annotation[index]
        
         # get text input-output    
        qa = datum['qa']
        question = datum['question']
        answer = datum['answer']
        inference = datum['inference']
        references = get_references(datum, is_train=True)
        
        # load image
        is_sherlock = datum['source'] == 'sherlock'
        if 'VG_100K' in datum['image'] or 'vcr1images' in datum['image']:
            image_path = os.path.join(self.vis_root, datum['image']) 
        else:
            image_path = os.path.join(self.vis_root, 'vcr1images', datum['image'])
        image = Image.open(image_path).convert('RGB')
        
        # draw up to three regions in image
        draw_region = random.random() < 0.8 # or not is_sherlock
        if draw_region:  # randomly draw region for training
            region_references = get_region_references(references, qa)
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
        # else: 
        #     # if not drawn, replace id with region descriptions
        #     if not is_sherlock:
                
        image = self.vis_processor(image)
        
        # fill ID tags and convert tokens to string
        # question, answer = split_qa(qa)
        use_descriptions = not draw_region   # use region descriptions instead of ids if region is not drawn.
            
        tag_fn = add_tags
        question = tokens2str(tag_fn(question, references))
        answer = tokens2str(tag_fn(answer, references, use_descriptions))
        if inference is not None:
            inference = tokens2str(tag_fn(inference, references, use_descriptions))
        
        answer_question = random.random() < 0.25  # directly answer question instead of generating questions
        if answer_question:
            text_input = self.text_processor(question)
            if inference is not None:
                text_output = f"Answer: {answer} Inference: {inference}"
            else:
                text_output = f"Answer: {answer}"
        else:  # generate question as well
            text_input = self.text_processor.prompt
            if inference is not None:
                text_output = f"{question} Answer: {answer} Inference: {inference}"
            else:
                text_output = f"{question} Answer: {answer}"
        
        text_output = self.text_processor(text_output, add_prompt=False)
            
            
        return {"image": image, 
                "text_input": text_input,
                "text_output": text_output,
                "image_id": index,
                "instance_id": datum['index']
                }
    
    def collater(self, samples):
        image_list, input_list, output_list, instance_id_list, image_id_list = [], [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            input_list.append(sample["text_input"])
            output_list.append(sample["text_output"])
            instance_id_list.append(sample["instance_id"])
            image_id_list.append(sample["image_id"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": input_list,
            "text_output": output_list,
            "instance_id": instance_id_list,
            "image_id": torch.tensor(image_id_list, dtype=torch.int),
        }

class VisualCometDataset(BaseDataset):
    """ UnifiedVQA Eval Dataset with Question Input
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, info):        
        super().__init__(vis_processor, text_processor, vis_root, [])
        random.seed(42)

        self.region_mode = info.get('region_mode', 'boxes')
        
        self.annotation = []
        for ann_path in ann_paths:
            with open(ann_path) as f:
                for line in tqdm(f):
                    data = json.loads(line.strip())
                    self.annotation.append(data)
            print('loaded', ann_path)
        self.draw_others = info.get('draw_others', False)
  
    def __getitem__(self, index):
        datum = self.annotation[index]
        
         # get text input-output    
        question = datum['question']
        answers = datum['answers']
        conn = datum['connector']
        references = get_references(datum, is_train=True)
        
        # load image
        if 'VG_100K' in datum['image'] or 'vcr1images' in datum['image']:
            image_path = os.path.join(self.vis_root, datum['image']) 
        else:
            image_path = os.path.join(self.vis_root, 'vcr1images', datum['image'])
        image = Image.open(image_path).convert('RGB')
        
        # draw up to three regions in image
        region_references = get_region_references(references, question)
        region_references = region_references[:3] # keep only 3 references for now
        for ref_idx, ref in enumerate(region_references):
            region = ref[self.region_mode][0]
            if self.region_mode == 'crop':
                image = crop_image(image, region)
            else:
                if ref_idx == 0:
                    image = draw_PersonX(image, region, mode=self.region_mode)
                elif ref_idx == 1:
                    image = draw_PersonY(image, region, mode=self.region_mode)
                elif ref_idx == 2:
                    image = draw_PersonZ(image, region, mode=self.region_mode)
                
        image = self.vis_processor(image)
        
        # fill ID tags and convert tokens to string
        # question, answer = split_qa(qa)
        tag_fn = add_tags
        question = tokens2str(tag_fn(question, references))
        answers = [tokens2str(tag_fn(answer, references)) for answer in answers]
        
        text_input = f'{question} Answer: {conn} '
        text_input = self.text_processor(text_input)
        text_output = [self.text_processor(answer, add_prompt=False) for answer in answers]
            
            
        return {"image": image, 
                "text_input": text_input,
                "text_output": text_output,
                "image_id": index,
                "instance_id": datum['index']
                }
    
    def collater(self, samples):
        image_list, input_list, output_list, instance_id_list, image_id_list = [], [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            input_list.append(sample["text_input"])
            output_list.append(sample["text_output"])
            instance_id_list.append(sample["instance_id"])
            image_id_list.append(sample["image_id"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": input_list,
            "text_output": output_list,
            "instance_id": instance_id_list,
            "image_id": torch.tensor(image_id_list, dtype=torch.int),
        }
    
    @staticmethod
    def get_demo_input(image, bboxes, vis_processor, text_processor, question=None):
        
        for ref_idx, ref in enumerate(bboxes):
            mode = 'boxes'
            if ref_idx == 0:
                image = draw_PersonX(image, ref, mode=mode)
            elif ref_idx == 1:
                image = draw_PersonY(image, ref, mode=mode)
            elif ref_idx == 2:
                image = draw_PersonZ(image, ref, mode=mode)
        
        
        if question is None:
            references = [i for i in range(len(bboxes))]
            if len(references) == 1:
                question = "What is [0] doing?"
            else:
                question = f"What are {references} doing?"
        
        proc_image = vis_processor(image)
        question = text_processor(question)
        data = {"image": proc_image, "question": question, "region_references": bboxes,}

        return image, data