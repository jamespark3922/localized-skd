"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
from glob import glob
import numpy as np
import pandas as pd
from typing import List, Dict
import io
import random
import torch
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.vc_utils import draw_region, add_tags, tokens2str, read_jsonl

# import webdataset as wds


# TODO: Take tar file as it is.
# class ImageTextWebDataset(BaseDataset):
#     """ Image-Text Similarity Web Dataset
#     """
#     def __init__(self, vis_processor, text_processor, ann_paths, info):        
#         super().__init__(vis_processor, text_processor, None, [])
        
#         self.annotation = []
#         ds = wds.DataPipeline(
#             wds.SimpleShardList(ann_paths),
#             wds.tarfile_to_samples(),
#             wds.decode('pil'),
#             wds.batched(2, collation_fn=None, partial=False),
#         )

#         print(self.__getitem__(0))

class ImageTextExtractedDataset(BaseDataset):
    """
    Image-Text Similarity Web Dataset Extracted
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, info):        
        """

        Example annot:
        {
            'LICENSE': '?',
            'NSFW': 'UNLIKELY',
            'caption': "'Yo Bex! It's Dave. Da PM'  So what was actually said in those "
                        'texts between David Cameron and Rebekah Brooks? The conversation '
                        'might have gone like this …',
            'error_message': None,
            'exif': '{"Image Make": "NIKON CORPORATION", "Image Model": "NIKON D3", '
                    '"Image Orientation": "Horizontal (normal)", "Image XResolution": '
                    '"100", "Image YResolution": "100", "Image ResolutionUnit": "Not '
                    'Absolute", "Image YCbCrPositioning": "Centered", "Image ExifOffset": '
                    '"154", "EXIF ExposureTime": "1/30", "EXIF FNumber": "4", "EXIF '
                    'ExposureProgram": "Manual", "EXIF ISOSpeedRatings": "800", "EXIF '
                    'ExifVersion": "0220", "EXIF ComponentsConfiguration": "YCbCr", "EXIF '
                    'ShutterSpeedValue": "19868/4049", "EXIF ApertureValue": "4", "EXIF '
                    'ExposureBiasValue": "-2/3", "EXIF MeteringMode": "Pattern", "EXIF '
                    'LightSource": "Unknown", "EXIF FocalLength": "34", "EXIF '
                    'FlashPixVersion": "0100", "EXIF ColorSpace": "Uncalibrated"}',
            'height': 276,
            'key': '000002507',
            'md5': '674530a6a88de47c78516e918db2237a',
            'original_height': 276,
            'original_width': 460,
            'similarity': 0.3463859558105469,
            'status': 'success',
            'url': 'http://static.guim.co.uk/sys-images/Admin/BkFill/Default_image_group/2012/5/11/1336748126012/David-Cameron-texting-sho-008.jpg',
            'width': 460
        }
        """
        super().__init__(vis_processor, text_processor, vis_root, [])
        
        self.annotation = []
        for ann_path in ann_paths:
            image_paths = glob(os.path.join(ann_path, '*.jpg'))
            for image_path in image_paths:
                annot_path = image_path.replace('.jpg', '.json')
                with open(annot_path) as f:
                    annot = json.load(f)
                    annot['image_path'] = image_path
                self.annotation.append(annot)
        # self.annotation = self.annotation[:100]

        self.info = info
        print(self.__getitem__(0))

    def load_region_image(self, datum, regions):
        # if 'VG_100K' in datum['image'] or 'vcr1images' in datum['image']:
        image_path = os.path.join(self.vis_root, datum['image']) 
        # else:
        #     image_path = os.path.join(self.vis_root, 'vcr1images', datum['image'])
        image = Image.open(image_path).convert('RGB')
        for ref_idx, region in enumerate(regions[:5]):
            box = region['boxes'][0]
            image = draw_region(image, box, ref_idx, mode='boxes')

        return image
    
    def load_image(self, datum):
        image_path = datum['image_path']
        image = Image.open(image_path).convert('RGB')

        return image
     
    def parse_region_tokens(self, tokens: List[str], regions: List[Dict]) -> str:
        """ 
        Replace region IDs in tokens in the order of `regions` with 0-index and parse into string.

        :param: tokens (List[str]): List of tokens to replace the region IDs
        :param: regions (List[Dict]): list of regions containing bounding box coordinates and the IDs.
        """

        region_dct = {}
        for idx, region in enumerate(regions):
            region_dct[int(region['name'])] = region | {'index': idx} 

        process_text = tokens2str(add_tags(tokens, region_dct, reorder_ids=True))

        return process_text

    def __getitem__(self, index):
        datum = self.annotation[index]

        # regions = self.get_regions(datum)
        image = self.load_image(datum)
        image = self.vis_processor(image)   

        text = f"What is in this image? Answer: {datum['caption']}"
        text = self.text_processor(text)

        clip_sim = datum['similarity']
 
        return {
                "image": image, 
                "text_input": text,
                "image_id": index,
                "instance_id": datum["key"],
                'similarity': clip_sim,
                'caption': datum['caption']
            }
    
    def collater(self, samples):
        image_list, text_list, image_id_list, caption_list, clip_sim_list, instance_id_list = \
            [], [], [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            text_list.append(sample["text_input"])
            image_id_list.append(sample["image_id"])
            instance_id_list.append(sample["instance_id"])
            clip_sim_list.append(sample["similarity"])
            caption_list.append(sample["caption"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": text_list,
            "image_id": torch.tensor(image_id_list, dtype=torch.int),
            "instance_id": instance_id_list,
            "similarity": clip_sim_list,
            "caption": caption_list,
        }
    

if __name__ == '__main__':
    from torchvision import transforms

    def to_image_text_pair(sample):
        return sample[0], sample[1]["caption"]

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(256, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset = ImageTextExtractedDataset(
        vis_processor=transform_train,
        text_processor=lambda x: x,
        ann_paths=["/dev/shm/laion-samples/000000/"],
        info=None,
    )

    import torch
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    
            
