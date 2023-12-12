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

class VisualCometMCDataset(ArrowDataset):
    """ Sherlock Training Dataset
    
        We support mult-task clue+inference learning
    """
    def __init__(self, vis_processor, text_processor, arrow_root, ann_root, image_key, info, split):
        """
        arrow_root (string): directory containing *.arrow files
        """
        arrow_files = [os.path.join(arrow_root, f"visualcomet_XY_train.arrow")]
        super().__init__(vis_processor, text_processor, arrow_files, image_key=image_key, text_column_name='inference_text')
        self.use_widescreen = info.get('use_widescreen', True)
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

        text = self.get_text(img_index, caption_index)
        caption = self.text_processor(text)

        # multiple answer choices
        choices = [self.table["inference_ending%d" % i][index][caption_index].as_py() for i in range(4)]
        choices = [self.text_processor(c) for c in choices]
        label = self.table["label"][index].as_py()

        return {"image": image, "text_input": caption, "choices": choices, "label": label, "image_id": img_index}

    def collater(self, samples):
        image_list, caption_list, choices_list, label_list, image_ids = [], [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            caption_list.append(sample["text_input"])
            choices_list.append(sample["choices"])
            label_list.append(sample["label"])
            image_ids.append(sample["image_id"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": caption_list,
            "choices": choices_list,
            "label": torch.tensor(label_list, dtype=torch.long),
            "image_id": torch.tensor(image_ids, dtype=torch.int),
        }

class VisualCometMCEvalDataset(ArrowDataset):
    def __init__(self, vis_processor, text_processor, arrow_root, ann_root, image_key, info, split):
        """
        arrow_root (string): directory containing *.arrow files
        split (string): val or test
        """
        
        arrow_files = [os.path.join(arrow_root, f"visualcomet_XY_{split}.arrow")]
        super().__init__(vis_processor, text_processor, arrow_files, image_key=image_key, text_column_name='inference_text')
        self.use_widescreen = info.get('use_widescreen', True)
        
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

        choices = [self.table["inference_ending%d" % i][index][caption_index].as_py() for i in range(4)]
        choices = [self.text_processor(c) for c in choices]
        label = self.table["label"][index].as_py()

        return {"image": image, "text_input": choices, "label": label, "image_id": img_index}
    
    def collater(self, samples):
        image_list, caption_list, label_list, image_ids = [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            caption_list.append(sample["text_input"])
            label_list.append(sample["label"])
            image_ids.append(sample["image_id"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": caption_list,
            "label": torch.tensor(label_list, dtype=torch.long),
            "image_id": torch.tensor(image_ids, dtype=torch.int),
        }
