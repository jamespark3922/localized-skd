"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.arrow_dataset import ArrowDataset, ArrowEvalDataset

class WitDataset(ArrowDataset):
    def __init__(self, vis_processor, text_processor, arrow_root, ann_root, image_key, info, split):
        """
        arrow_root (string): directory containing *.arrow files
        """
        arrow_files = [os.path.join(arrow_root, f"wit_train_{i}.arrow")for i in range(38)]
        super().__init__(vis_processor, text_processor, arrow_files, image_key=image_key, text_column_name='page_title')

    def get_text(self, img_index, caption_index):
        
        title = self.all_texts[img_index][caption_index]
        ref = self.table["caption_reference_description"][img_index][caption_index].as_py()
        attr = self.table["caption_attribution_description"][img_index][caption_index].as_py()

        if not ref and not attr:
            caption  = title
        else:
            ref = '' if ref is None else ref
            attr = '' if attr is None else attr
            caption = f'{ref} {attr}'
        caption = caption.strip()
        
        return caption

class WitEvalDataset(ArrowEvalDataset):
    def __init__(self, vis_processor, text_processor, arrow_root, ann_root, image_key, info, split):
        """
        arrow_root (string): directory containing *.arrow files
        split (string): val or test
        """
        
        arrow_files = [os.path.join(arrow_root, f"wit_{split}_{i}_filter.arrow")for i in range(1)]
        super().__init__(vis_processor, text_processor, arrow_files, image_key=image_key, text_column_name='page_title')

    def get_text(self, img_index, caption_index):
        
        title = self.all_texts[img_index][caption_index]
        ref = self.table["caption_reference_description"][img_index][caption_index].as_py()
        attr = self.table["caption_attribution_description"][img_index][caption_index].as_py()

        if not ref and not attr:
            caption  = title
        else:
            assert ref is not None 
            ref = '' if ref is None else ref
            attr = '' # if attr is None else attr
            caption = f'{ref} {attr}'
        caption = caption.strip()
        
        return caption
