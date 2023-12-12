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
from lavis.datasets.vc_utils import *

class ExplanationDataset(BaseDataset):
    """ Visual Explanation Train Dataset
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, info):        
        super().__init__(vis_processor, text_processor, vis_root, [])
        
        self.region_mode = info.get('region_mode', 'boxes')  # for VCR
        
        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path)))
            print('loaded', ann_path)
        
        self.draw_others = info.get('draw_others', False)
  
    def __getitem__(self, index):
        datum = self.annotation[index]

        # load images
        image_path = os.path.join(self.vis_root, 'vcr1images', datum['img_id'])
        image = Image.open(image_path).convert('RGB')
        
        # fill ID tags and convert tokens to string
        qa = datum['question'] + datum['answer']
        # draw region to image
        references = get_references(datum)
        region_references = get_region_references(references, qa)
        region_references = region_references[:3] # keep only 3 references for now
        for ref_idx, ref in enumerate(region_references):
            mode = self.region_mode
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
        
        
        question = tokens2str(add_tags(datum['question'], references))
        answer = tokens2str(add_tags(datum['answer'], references))
        explanation = tokens2str(add_tags(datum['explanation'][0], references))
        
        question = self.text_processor(question)
        output = f"Answer: {answer} Rationale: {explanation}"
        text_output = self.text_processor(output, add_prompt=False) # no need for question prefix in answer.

        return {"image": image, 
                "text_input": question,
                "text_output": text_output, 
                "question": question, "answer": text_output,
                "image_id": index}
    
    def collater(self, samples):
        image_list, text_input_list, text_output_list, question_list, answer_list, image_id_list = [], [], [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            text_input_list.append(sample["text_input"])
            text_output_list.append(sample["text_output"])
            question_list.append(sample["question"])
            answer_list.append(sample["answer"])
            image_id_list.append(sample["image_id"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": text_input_list,
            "text_output": text_output_list,
            "question": question_list,
            "answer": answer_list,
             "image_id": torch.tensor(image_id_list, dtype=torch.int),
        }

class ExplanationVCREvalDataset(BaseDataset):
    """ Explanation Eval Dataset for Answer Multiple Choice Task
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, info):        
        super().__init__(vis_processor, text_processor, vis_root, [])
        
        self.annotation = []
        for ann_path in ann_paths:
            with open(ann_path) as f:
                for line in f:
                    data = json.loads(line.strip())
                    if len(data['target_options']) > 0:
                        self.annotation.append(data)
        self.region_mode = info.get('region_mode', 'boxes')
        self.draw_others = info.get('draw_others', False)
 
    def __getitem__(self, index):
        datum = self.annotation[index]
        
        qa_idx = 0
        question = datum['questions'][qa_idx]
        references = get_references(datum, is_train=False)
        is_sherlock = datum['source'] == 'sherlock'
        
        # load image
        if 'VG_100K' in datum['image']:
            image_path = os.path.join(self.vis_root, datum['image']) 
        else:
            image_path = os.path.join(self.vis_root, 'vcr1images', datum['image'])
        image = Image.open(image_path).convert('RGB')
        
        # draw region to image
        region_references = get_region_references(references, question)
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
        
        
        tag_fn = add_tags
        question = tokens2str(tag_fn(question, references))
                    
        image = self.vis_processor(image)
        data = {"image": image, "question": question, "region_references": region_references, "image_id": index}
        
        # get text endings
        if 'answers' in datum:
            qa = datum['target_options'][qa_idx]
            answer = datum['answers'][qa_idx]
            inference = datum['inferences']
            
            qa = tokens2str(tag_fn(qa, references))
            answer = tokens2str(tag_fn(answer, references))
            if inference is not None:
                inference = inference[qa_idx]
                inference = tokens2str(tag_fn(inference, references))
                answer = answer + ' ' + inference
                
            text_input = self.text_processor(qa)
            answer = self.text_processor(answer, add_prompt=False) # no need for question prefix in answer.
            data.update({"text_input": text_input,"answer": answer})

        return data
    
    def collater(self, samples):
        image_list, text_list, question_list, answer_list, image_id_list = [], [], [], [], []

        if "answer" in samples[0]:
            for sample in samples:
                image_list.append(sample["image"])
                text_list.append(sample["text_input"])
                question_list.append(sample["question"])
                answer_list.append(sample["answer"])
                image_id_list.append(sample["image_id"])

            return {
                "image": torch.stack(image_list, dim=0),
                "text_input": text_list,
                "question": question_list,
                "answer": answer_list,
                "image_id": torch.tensor(image_id_list, dtype=torch.int),
            }
        else:
            for sample in samples:
                image_list.append(sample["image"])
                text_list.append(sample["text_input"])
                question_list.append(sample["question"])
                image_id_list.append(sample["image_id"])

            return {
                "image": torch.stack(image_list, dim=0),
                "text_input": text_list,
                "question": question_list,
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
            
