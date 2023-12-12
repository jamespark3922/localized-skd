"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json

from pathlib import Path

import torch
import io
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.arrow_dataset import ArrowDataset, ArrowEvalDataset

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

class SherlockCaptionDataset(ArrowDataset):
    """ Sherlock Training Dataset
    
        We support mult-task clue+inference learning
    """
    def __init__(self, vis_processor, text_processor, arrow_root, ann_root, image_key, info, split):
        """
        arrow_root (string): directory containing *.arrow files
        image_key: 
                - with region: 'region_image', 
                - no region: 'image'
        """
        
        arrow_files = [os.path.join(arrow_root, f"sherlock_train_{i}.arrow")for i in range(4)]
        super().__init__(vis_processor, text_processor, arrow_files, image_key=image_key, text_column_name='caption')
        self.use_widescreen = info.get('use_widescreen', False)

        # Get image-text mapping
        clue_prompt = 'clue: '
        inference_prompt = 'inference: '
        all_clues = [clue_prompt + c for c in self.table['clue'].to_pandas().tolist()]
        self.all_texts = []
        self.index_mapper = dict() # {batch index: (image_index, sent_index)}
        j = 0
        for i in range(len(all_clues)):
            for _j in range(len(all_clues[i])):
                text = all_clues[i][_j]
                assert isinstance(text, str)
                self.index_mapper[j] = (i, _j)
                self.all_texts.append(text)
                j += 1
        self.use_inference = info.get('use_inference', False)
        if self.use_inference:
            all_inferences = [inference_prompt + c for c in self.table['caption'].to_pandas().tolist()]
            assert len(all_clues) == len(all_inferences)
            for i in range(len(all_inferences)):
                n = len(all_clues[i])
                for _k in range(len(all_inferences[i])):
                    text = all_inferences[i][_k]
                    assert isinstance(text, str)
                    self.index_mapper[j] = (i, n+_k)
                    self.all_texts.append(text)
                    j += 1

        print('='* 10)
        print('Training data')
        print('Image key:', self.image_key)
        print('Use widescreen:', self.use_widescreen)
        print('Total number of Training Image-Text Pairs:', len(self.index_mapper))
        print('='* 10)

    def __getitem__(self, index):

        img_index, caption_index = self.index_mapper[index]
        
        image_bytes = io.BytesIO(self.table[self.image_key][img_index].as_py())
        image_bytes.seek(0)
        image = Image.open(image_bytes).convert('RGB')   

        if self.use_widescreen:
            images = crop_widescreens(image) # [imgae1, imgae2]
            image = torch.stack([self.vis_processor(r) for r in images], 0)
        else:
            image = self.vis_processor(image)

        text = self.all_texts[index]
        caption = self.text_processor(text)

        return {"image": image, "text_input": caption, "image_id": img_index}

class SherlockCaptionEvalDataset(ArrowEvalDataset):
    def __init__(self, vis_processor, text_processor, arrow_root, ann_root, image_key, info, split):
        """
        arrow_root (string): directory containing *.arrow files
        ann_root (string): directory containing comparison eval annotations
        split (string): val or test
        """
        
        arrow_files = [os.path.join(arrow_root, f"sherlock_val_{i}.arrow")for i in range(1)]
        super().__init__(vis_processor, text_processor, arrow_files, image_key=image_key, text_column_name='caption')
        self.use_widescreen = info.get('use_widescreen', False)

        # Get image-text mapping
        clue_prompt = 'clue: '
        all_clues = [clue_prompt + c for c in self.table['clue'].to_pandas().tolist()]
        self.all_texts = []
        self.index_mapper = dict() # {batch index: (image_index, sent_index)}
        j = 0
        for i in range(len(all_clues)):
            for _j in range(len(all_clues[i])):
                text = all_clues[i][_j]
                assert isinstance(text, str)
                self.index_mapper[j] = (i, _j)
                self.all_texts.append(text)
                j += 1

        print('='* 10)
        print('Training data')
        print('Image key:', self.image_key)
        print('Use widescreen:', self.use_widescreen)
        print('Total number of Training Image-Text Pairs:', len(self.index_mapper))
        print('='* 10)

    def __getitem__(self, index):

        img_index, caption_index = self.index_mapper[index]
        
        image_bytes = io.BytesIO(self.table[self.image_key][img_index].as_py())
        image_bytes.seek(0)
        image = Image.open(image_bytes).convert('RGB')   

        if self.use_widescreen:
            images = crop_widescreens(image) # [imgae1, imgae2]
            image = torch.stack([self.vis_processor(r) for r in images], 0)
        else:
            image = self.vis_processor(image)

        text = self.all_texts[index]
        caption = self.text_processor(text)

        return {"image": image, "text_input": caption, "image_id": img_index}

class SherlockPublicTestDataset(ArrowEvalDataset):
    def __init__(self, vis_processor, text_processor, arrow_root, ann_root, split):
        """
        arrow_root (string): directory containing *.arrow files
        ann_root (string): directory containing comparison eval annotations
        split (string): val or test
        """
        
        arrow_files = [os.path.join(arrow_root, f"sherlock_test_{i}.arrow")for i in range(1)]
        super().__init__(vis_processor, text_processor, arrow_files, text_column_name='caption')

        # load comparison GT annotations
        ann_root = Path(ann_root)
        with open(ann_root / 'test_comparison_public/test_instances.json') as f:
            answer_key = json.load(f)
        self.annotations = answer_key['annotations']

        # map comparison annotation to image_id in arrow table
        self.index_mapper = {}
        image_ids = self.table['image_id'].to_pandas().tolist()
        for i, id in enumerate([annot['Input_iid'] for annot in self.annotations]): 
            self.index_mapper[i] = image_ids.index(id)
            
    def __len__(self):
        return len(self.index_mapper)
        
    def __getitem__(self, index):

        img_index = self.index_mapper[index]
        
        image_bytes = io.BytesIO(self.table["region_image"][img_index].as_py())
        image_bytes.seek(0)
        image = Image.open(image_bytes).convert('RGB')   
        image = self.vis_processor(image)

        candidates = self.annotations[index]['candidates']
        captions = [self.text_processor(t['prediction']) for t in candidates]
        annot_keys = ['annot1', 'annot2', 'annot12']
        scores = [[t[k] for k in  annot_keys] for t in candidates]

        return {"image": image, "text_input": captions, "image_id": index, 'scores': scores}
