"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

import io
from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
from torchvision import datasets
from typing import List

import pyarrow as pa


def read_arrow(arrow_path):
    table = pa.ipc.RecordBatchFileReader(
        pa.memory_map(arrow_path, "r")
    ).read_all()
    return table

class ArrowDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, arrow_files:List[str], image_key='image', text_column_name='caption'):
        """
        arrow_root (string): Root directory of pyarrow files
        """
        super().__init__(vis_processor, text_processor)

        tables = [read_arrow(arrow) for arrow in arrow_files] 
        self.table = pa.concat_tables(tables, promote=True)

        self.image_key = image_key
        self.all_texts = self.table[text_column_name].to_pandas().tolist()
        self.index_mapper = dict() # {batch index: (image_index, sent_index)}
        j = 0
        for i, texts in enumerate(self.all_texts):
            for _j in range(len(texts)):
                assert isinstance(texts[_j], str)
                self.index_mapper[j] = (i, _j)
                j += 1

    def __len__(self):
            return len(self.index_mapper)        
    
    def get_text(self, img_index, caption_index):
        return self.all_texts[img_index][caption_index]
    
    def get_image(self, img_index):
        image_bytes = io.BytesIO(self.table[self.image_key][img_index].as_py())
        image_bytes.seek(0)
        image = Image.open(image_bytes).convert('RGB')
        return image

    def __getitem__(self, index):

        img_index, caption_index = self.index_mapper[index]
        
        image = self.get_image(img_index)
        image = self.vis_processor(image)

        text = self.get_text(img_index, caption_index)
        caption = self.text_processor(text)

        return {"image": image, "text_input": caption, "image_id": img_index}


class ArrowEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, arrow_files:List[str], image_key='image', text_column_name='caption'):
        """
        arrow_root (string): Root directory of pyarrow files
        """
        super().__init__(vis_processor, text_processor)

        tables = [read_arrow(arrow) for arrow in arrow_files] 
        self.table = pa.concat_tables(tables, promote=True)

        self.image_key = image_key
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        self.all_texts = self.table[text_column_name].to_pandas().tolist()
        j = 0
        for i, texts in enumerate(self.all_texts):
            self.image.append(self.table[self.image_key][i].as_py())
            self.img2txt[i] = []
            for _j in range(len(texts)):
                assert isinstance(texts[_j], str)
                caption = self.get_text(i, _j)
                self.text.append(self.text_processor(caption))
                self.img2txt[i].append(j)
                self.txt2img[j] = i
                j += 1

    def __len__(self):
            return len(self.image)        
    
    def get_text(self, img_index, caption_index):
        return self.all_texts[img_index][caption_index]

    def __getitem__(self, index):
        
        image_bytes = io.BytesIO(self.table[self.image_key][index].as_py())
        image_bytes.seek(0)
        image = Image.open(image_bytes).convert('RGB')   
        image = self.vis_processor(image)

        return {"image": image, "image_id": index}
