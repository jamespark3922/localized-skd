"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from copy import deepcopy
import numpy as np
import os
import json

from pathlib import Path

import io
import torch
from PIL import Image, ImageFile, ImageDraw

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.arrow_dataset import ArrowDataset, ArrowEvalDataset
from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.sherlock_datasets import crop_widescreens
from lavis.datasets.vc_utils import *

class VisualCometInferenceDataset(ArrowDataset):
    """ VisualCOMET Inference Dataset
    
        Only supports event generation.
    """
    def __init__(self, vis_processor, text_processor, arrow_root, ann_root, image_key, info, split):
        """
        arrow_root (string): directory containing *.arrow files
        """
        arrow_files = [os.path.join(arrow_root, f"visualcomet_{split}_person_details_XYZ.arrow")]
        super().__init__(vis_processor, text_processor, arrow_files, image_key=image_key, text_column_name='event_XYZ')
        self.use_widescreen = False # info.get('use_widescreen', False)
        self.region_mode = info.get('region_mode', 'bbox')
        self.draw_others = info.get('draw_others', False)
        self.event_only = info.get('event_only', False)
        self.no_ambience = info.get('no_ambience', True)

        self.ann_root = Path(ann_root)
        self.all_texts = []
        self.all_mappings = []
        self.index_mapper = dict() # {batch index: (image_index, text_type)}

        subjects = [s[0] for s in self.table['subject_pids'].to_pandas().tolist()]

        # events
        event_prompt = "event: "
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

        if not self.event_only:
            # before, after, intent inferences
            prompt = {
                'before': 'before: PersonX needed to ', 
                'intent': 'intent: PersonX wanted to ',
                'after': 'after: PersonX will most likely '
            }
            for k in ['before', 'intent', 'after']:
                inf_prompt = prompt[k]
                inferences = [inf_prompt + inf for inf in self.table[k+'_id_XYZ'].to_pandas().tolist()]
                inference_mappings = self.table[k+'_id_XYZ_mapping'].to_pandas().tolist()

                for i in range(len(inferences)):
                    for _k in range(len(inferences[i])):
                        text = inferences[i][_k]
                        mapping = inference_mappings[i][_k]
                        assert isinstance(text, str)
                        assert isinstance(mapping, dict)
                        self.index_mapper[j] = (i, k)
                        self.all_texts.append(text)
                        self.all_mappings.append(mapping)
                        j += 1

            if not self.no_ambience:
                # person ambiences
                prompt = {
                    'ambience': 'looks: PersonX looks ',
                    'emotion': 'feels: PersonX feels ',
                    'identity': 'appears: PersonX appears to be ', 
                }
                for k in ['ambience', 'emotion', 'identity']:
                    ambience_prompt = prompt[k]
                    ambiences = [ambience_prompt + inf for inf in self.table[k].to_pandas().tolist()]
                    for i in range(len(ambiences)):
                        subject = subjects[i]
                        for _k in range(len(ambiences[i][:2])):  # keep only top two
                            text = ambiences[i][_k]
                            assert isinstance(text, str)
                            self.index_mapper[j] = (i, k)
                            self.all_texts.append(text)
                            self.all_mappings.append({'PersonX': f'[person{subject}]'})
                            j += 1

        print('Number of images:', len(self.table))
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
        
        if self.draw_others:
            if mapping.get('PersonY', None):
                pid = int(mapping['PersonY'])
                region = regions[pid]
                image = draw_PersonY(image, region, mode=self.region_mode)
            if mapping.get('PersonZ', None):
                pid = int(mapping['PersonZ'])
                region = regions[pid]
                image = draw_PersonZ(image, region, mode=self.region_mode)

        if self.use_widescreen:
            images = crop_widescreens(image) # [image1, image2]
            image = torch.stack([self.vis_processor(r) for r in images], 0)
        else:   
            image = self.vis_processor(image)

        return {"image": image, "text_type": text_type, "text_input": caption, "image_id": img_index}

class VisualCometInferenceEvalDataset(ArrowDataset):
    def __init__(self, vis_processor, text_processor, arrow_root, ann_root, image_key, info, split):
        """
        arrow_root (string): directory containing *.arrow files
        split (string): val or test
        """

        arrow_files = [os.path.join(arrow_root, f"visualcomet_{split}_person_details_XYZ.arrow")]
        super().__init__(vis_processor, text_processor, arrow_files, image_key=image_key, text_column_name='event_XYZ')
        self.use_widescreen = False # info.get('use_widescreen', False)
        self.region_mode = info.get('region_mode', 'bbox')
        self.draw_others = info.get('draw_others', False)
        self.event_only = info.get('event_only', False)
        self.no_ambience = info.get('no_ambience', True)

        self.ann_root = Path(ann_root)
        self.text = []
        self.all_mappings = []
        self.index_mapper = dict() # {batch index: (image_index, text_type)}

        subjects = [s[0] for s in self.table['subject_pids'].to_pandas().tolist()]

        # events
        event_prompt = "event: "
        events = self.table['event_XYZ'].to_pandas().tolist()
        event_mappings = self.table['event_XYZ_mapping'].to_pandas().tolist()
        j = 0
        for i in range(len(events)):
            text = events[i]
            mapping = event_mappings[i]
            assert isinstance(text, str)
            assert isinstance(mapping, dict)
            self.index_mapper[j] = (i, 'event', event_prompt)
            self.text.append(text)
            self.all_mappings.append(mapping)
            j += 1

        if not self.event_only:
            # before, after, intent inferences
            prompt = {
                'before': 'before: PersonX needed to ', 
                'intent': 'intent: PersonX wanted to ',
                'after': 'after: PersonX will most likely '
            }
            for k in ['before', 'intent', 'after']:
                inf_prompt, inf_text = prompt[k].split(': ')
                inf_prompt = inf_prompt + ': '
                inferences = [inf_text + inf for inf in self.table[k+'_id_XYZ'].to_pandas().tolist()]
                inference_mappings = self.table[k+'_id_XYZ_mapping'].to_pandas().tolist()

                for i in range(len(inferences)):
                    text = inferences[i][0]
                    mapping = inference_mappings[i][0]
                    assert isinstance(text, str)
                    assert isinstance(mapping, dict)
                    self.index_mapper[j] = (i, k, inf_prompt)
                    self.text.append(text)
                    self.all_mappings.append(mapping)
                    j += 1
            
            # person ambiences
            if not self.no_ambience:
                prompt = {
                    'ambience': 'looks: PersonX looks ',
                    'emotion': 'feels: PersonX feels ',
                    'identity': 'appears: PersonX appears to be ', 
                }
                for k in ['ambience', 'emotion', 'identity']:
                    amb_prompt, amb_text = prompt[k].split(': ')
                    amb_prompt = amb_prompt + ': '
                    ambiences = [amb_text + inf for inf in self.table[k].to_pandas().tolist()]
                    for i in range(len(ambiences)):
                        subject = subjects[i]
                        text = ambiences[i][0]
                        assert isinstance(text, str)
                        self.index_mapper[j] = (i, k, amb_prompt)
                        self.text.append(text)
                        self.all_mappings.append({'PersonX': f'[person{subject}]'})
                        j += 1

        print('Total number of Training Image-Text Pairs:', len(self.index_mapper))
        assert len(self.index_mapper) == len(self.text) == len(self.all_mappings)

    def __getitem__(self, index):

        img_index, text_type, prompt = self.index_mapper[index]
        
        text = self.text[index]
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
        
        if self.draw_others:
            if mapping.get('PersonY', None):
                pid = int(mapping['PersonY'])
                region = regions[pid]
                image = draw_PersonY(image, region, mode=self.region_mode)
            if mapping.get('PersonZ', None):
                pid = int(mapping['PersonZ'])
                region = regions[pid]
                image = draw_PersonZ(image, region, mode=self.region_mode)

        if self.use_widescreen:
            images = crop_widescreens(image) # [image1, image2]
            image = torch.stack([self.vis_processor(r) for r in images], 0)
        else:   
            image = self.vis_processor(image)

        return {
            "image": image, 
            "text_type": text_type, 
            "text_input": caption,
            "prompt": prompt,
             "instance_id": img_index,
             "image_id": index
        }


class VisualCometInferencePredDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, input_data, vis_root, info):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths=[])
        self.region_mode = info.get('region_mode', 'bbox')
        self.draw_others = info.get('draw_others', False)
        self.event_only = info.get('event_only', False)

        self.prompts = {
                'event': 'event: ',
        }
        if not self.event_only:
            self.prompts.update({
                'before': 'before: PersonX needed to ', 
                'intent': 'intent: PersonX wanted to ',
                'after': 'after: PersonX will most likely ',
                'ambience': 'looks: PersonX looks ',
                'emotion': 'feels: PersonX feels ',
                'identity': 'appears: PersonX appears to be ',
            })

        self.data = []
        for d in input_data:
            for prompt_type, prompt in self.prompts.items():
                instance = deepcopy(d)
                instance['text_type'] = prompt_type
                instance['prompt'] = prompt
                self.data.append(instance)

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        instance = self.data[index]
        img_path = os.path.join(self.vis_root, instance['img_fn'])
        metadata_path = os.path.join(self.vis_root, instance['metadata_fn'])
        prompt_type = instance['text_type']
        subject = instance['subject_pids'][0]
        return self.load_image(img_path, metadata_path, prompt_type, subject)

    def load_image(self, img_path: str, metadata_path: str, prompt_type: str, subject: int):
        image = Image.open(img_path).convert('RGB')
        metadata = json.load(open(metadata_path))

        # draw person region as bbox or polygon
        if self.region_mode == 'polygon':
            region = metadata['segms'][subject]
        else:
            region = metadata['boxes'][subject][:4]
        image = draw_PersonX(image, region, mode=self.region_mode)
        # if self.use_widescreen:
        #     images = crop_widescreens(image) # [image1, image2]
        #     image = torch.stack([self.vis_processor(r) for r in images], 0)
        # else:   
        image = self.vis_processor(image)
        prompt_text = self.prompts[prompt_type]
        
        return {
            "image": image, 
            "text_type": prompt_type, 
            "prompt": prompt_text,
        }