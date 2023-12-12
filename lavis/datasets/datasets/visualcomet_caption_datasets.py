"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json

from pathlib import Path

import io
import torch
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.arrow_dataset import ArrowDataset, ArrowEvalDataset
from lavis.datasets.datasets.sherlock_datasets import crop_widescreens

class VisualCometCaptionDataset(ArrowDataset):
    """ VisualCOMEt caption Dataset
    
        Only supports event generation.
    """
    def __init__(self, vis_processor, text_processor, arrow_root, ann_root, image_key, info, split):
        """
        arrow_root (string): directory containing *.arrow files
        """
        arrow_files = [os.path.join(arrow_root, f"visualcomet_XY_train.arrow")]
        super().__init__(vis_processor, text_processor, arrow_files, image_key=image_key, text_column_name='inference_text')
        self.use_widescreen = False # info.get('use_widescreen', False)

        # Get image-text mapping
        event_prompt = "event: "
        events = [event_prompt + event for event in self.table['event_text'].to_pandas().tolist()]

        self.all_texts = []
        self.index_mapper = dict() # {batch index: (image_index, sent_index)}
        j = 0
        for i in range(len(events)):
            for _j in range(len(events[i])):
                text = events[i][_j]
                assert isinstance(text, str)
                self.index_mapper[j] = (i, _j)
                self.all_texts.append(text)
                j += 1
        self.use_inference = info.get('use_inference', False)
        if self.use_inference:
            inference_prompt = "inference: "
            inferences = [inference_prompt + inf for inf in self.table['inference_text'].to_pandas().tolist()]
            assert len(events) == len(inferences)

            for i in range(len(inferences)):
                n = len(events[i])
                for _k in range(len(inferences[i])):
                    text = inferences[i][_k]
                    assert isinstance(text, str)
                    self.index_mapper[j] = (i, n+_k)
                    self.all_texts.append(text)
                    j += 1

        print('Total number of Training Image-Text Pairs:', len(self.index_mapper))    

    def __getitem__(self, index):

        img_index, caption_index = self.index_mapper[index]
        
        image_bytes = io.BytesIO(self.table[self.image_key][img_index].as_py())
        image_bytes.seek(0)
        image = Image.open(image_bytes).convert('RGB')
        
        if self.use_widescreen:
            images = crop_widescreens(image) # [image1, image2]
            image = torch.stack([self.vis_processor(r) for r in images], 0)
        else:   
            image = self.vis_processor(image)

        text = self.all_texts[index]
        caption = self.text_processor(text)
        
        return {"image": image, "text_input": caption, "image_id": img_index}

class VisualCometCaptionEvalDataset(ArrowDataset):
    def __init__(self, vis_processor, text_processor, arrow_root, ann_root, image_key, info, split):
        """
        arrow_root (string): directory containing *.arrow files
        split (string): val or test
        """
        
        arrow_files = [os.path.join(arrow_root, f"visualcomet_XY_{split}.arrow")]
        super().__init__(vis_processor, text_processor, arrow_files, image_key=image_key, text_column_name='inference_text')
        self.use_widescreen = info.get('use_widescreen', False)
        
    def __getitem__(self, index):

        img_index, caption_index = self.index_mapper[index]
        
        image_bytes = io.BytesIO(self.table[self.image_key][img_index].as_py())
        image_bytes.seek(0)
        image = Image.open(image_bytes).convert('RGB')   

        if self.use_widescreen:
            images = crop_widescreens(image) # [image1, image2]
            image = torch.stack([self.vis_processor(r) for r in images], 0)
        else:   
            image = self.vis_processor(image)

        text = self.get_text(img_index, caption_index)
        caption = self.text_processor(text)

        return {"image": image, "text_input": caption, "image_id": img_index}
