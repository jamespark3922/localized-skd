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
import h5py

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.vc_utils import *

# for evaluation
class Flickr30KDataset(BaseDataset):
    """ Visual Explanation Train Dataset
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, info):        
        super().__init__(vis_processor, text_processor, vis_root, [])
        
        self.region_mode = info.get('region_mode', 'boxes')  # for VCR
        
        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path)))
            print('loaded', ann_path)
        
        self.regions = h5py.File(info.get('regions'))
        self.draw_others = info.get('draw_others', False)
  
    def __getitem__(self, index):
        datum = self.annotation[index]

        text_input = 'What is happening in [0]? [0] is a {}'
        input_images = []
        for image in images:
            self.vis_processor(image)
        # load images
        image_path = os.path.join(self.vis_root, 'flickr30k', datum['img_id'])
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
            
