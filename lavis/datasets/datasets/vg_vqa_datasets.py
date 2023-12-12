"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os

from PIL import Image

from lavis.datasets.datasets.vqa_datasets import VQADataset
from lavis.datasets.datasets.retrieval_mc_datasets import RetrievalMCDataset

class VGVQADataset(VQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        answers = [ann["answer"]]
        # TODO this should be configured better
        weights = [0.2]

        return {
            "image": image,
            "text_input": question,
            "answers": answers,
            "weights": weights,
        }


class V7WVQADataset(RetrievalMCDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, [])
        for ann_path in ann_paths:
            with open(ann_path) as f:
                data = json.load(f)
                for datum in data:
                    for qa in datum['qa_pairs']:
                        qa['filename'] = datum['filename']
                        self.annotation.append(qa)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["filename"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        answers = ann['multiple_choices'] + [ann['answer']]
        text_input = [self.text_processor(f"{ann['question']} Answer: {answer}") for answer in answers]

        return {
            "image": image,
            "text_input": text_input,
            "image_id": ann['image_id'],
            "instance_id": ann["qa_id"],
            "label": 3, # always the last one
        }
