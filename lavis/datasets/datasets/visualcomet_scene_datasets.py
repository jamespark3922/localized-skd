"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import numpy as np
import os
import json

from pathlib import Path

import io
import torch
from PIL import Image, ImageFile, ImageDraw

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.arrow_dataset import ArrowDataset, ArrowEvalDataset
from lavis.datasets.datasets.sherlock_datasets import crop_widescreens

class VisualCometSceneDataset(ArrowDataset):
    """ VisualCOMET Scene Dataset.
        Scene Descriptor for Overall Image in VCR
    """
    def __init__(self, vis_processor, text_processor, arrow_root, ann_root, image_key, info, split):
        """
        arrow_root (string): directory containing *.arrow files
        """
        arrow_files = [os.path.join(arrow_root, f"visualcomet_{split}_person_details_XYZ.arrow")]
        super().__init__(vis_processor, text_processor, arrow_files, image_key=image_key, text_column_name='event_XYZ')

        self.ann_root = Path(ann_root)
        self.all_texts = []
        self.all_mappings = []
        self.index_mapper = dict() # {batch index: (image_index, text_type)}

        # events
        event_prompt = "scene"
        events = [event_prompt + event for event in self.table['event_XYZ'].to_pandas().tolist()]
        event_mappings = self.table['event_XYZ_mapping'].to_pandas().tolist()
        j = 0
        for i in range(len(events)):
            text = events[i]
            mapping = event_mappings[i]
            assert isinstance(text, str)
            assert isinstance(mapping, dict)
            self.index_mapper[j] = (i, 'event')
            self.all_texts.append(text)
            self.all_mappings.append(mapping)
            j += 1

        print('Total number of Training Image-Text Pairs:', len(self.index_mapper))
        assert len(self.index_mapper) == len(self.all_texts) == len(self.all_mappings)

    def __getitem__(self, index):

        img_index, text_type = self.index_mapper[index]
        
        text = self.all_texts[index]
        mapping = self.all_mappings[index]
        caption = self.text_processor(text)

        image_bytes = io.BytesIO(self.table[self.image_key][img_index].as_py())
        image_bytes.seek(0)
        image = Image.open(image_bytes).convert('RGB')
        
        # draw person region as bbox or polygon
        if self.region_mode == 'polygon':
            meta = self.table['metadata_fn'][img_index].as_py()
            meta = json.load(open(self.ann_root / meta))
            regions = meta['segms']
        else:
            regions = self.table['bboxes'][img_index].as_py()

        if self.use_widescreen:
            images = crop_widescreens(image) # [image1, image2]
            image = torch.stack([self.vis_processor(r) for r in images], 0)
        else:   
            image = self.vis_processor(image)

        return {"image": image, "text_type": text_type, "text_input": caption, "image_id": img_index}

class VisualCometCaptionEvalDataset(ArrowDataset):
    def __init__(self, vis_processor, text_processor, arrow_root, ann_root, image_key, info, split):
        """
        arrow_root (string): directory containing *.arrow files
        split (string): val or test
        """
        
        arrow_files = [os.path.join(arrow_root, f"visualcomet_{split}_person_details_XYZ.arrow")]
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
