"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from collections import OrderedDict
import random
import json
import os
import torch
from copy import deepcopy

from PIL import Image

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionEvalDataset

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "direct_answers": "; ".join(ann["direct_answers"]),
                "choices": "; ".join(ann["choices"]),
                "correct_choice": ann["choices"][ann["correct_choice_idx"]],
                "image": sample["image"],
            }
        )


class AOKVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]
        
        split = ann["split"]
        image_name =  f"{split}2017/{ann['image_id']:012d}.jpg"
        image_path = os.path.join(self.vis_root, image_name)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        answer_key = "direct_answers"

        answer_weight = {}
        for answer in ann[answer_key]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann[answer_key])
            else:
                answer_weight[answer] = 1 / len(ann[answer_key])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        return {
            "image": image,
            "text_input": question,
            "answers": answers,
            "weights": weights,
        }

class AOKVQASyntheticDataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]
        
        split = ann["split"]
        image_name =  f"{split}_{ann['image_id']}_{ann['question_id']}_0.jpg"
        image_path = os.path.join(self.vis_root, image_name)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        answer_key = "choice_idx"

        answers = [ann['choices'][ann['choice_idx']]]
        weights = [1.0]
        # answer_weight = {}
        # for answer in [ann[answer_key]]:
        #     if answer in answer_weight.keys():
        #         answer_weight[answer] += 1 / len(ann[answer_key])
        #     else:
        #         answer_weight[answer] = 1 / len(ann[answer_key])

        # answers = list(answer_weight.keys())
        # weights = list(answer_weight.values())

        return {
            "image": image,
            "text_input": question,
            "answers": answers,
            "weights": weights,
        }


class AOKVQAEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root

        self.annotation = json.load(open(ann_paths[0]))

        answer_list_path = ann_paths[1]
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
        else:
            self.answer_list = None

        try:
            self.coco_fmt_qust_file = ann_paths[2]
            self.coco_fmt_anno_file = ann_paths[3]
        except IndexError:
            self.coco_fmt_qust_file = None
            self.coco_fmt_anno_file = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def collater(self, samples):
        (
            image_list,
            question_list,
            question_id_list,
            instance_id_list,
            choices_list,
            correct_choice_idx_list,
            direct_answers_list,
        ) = ([], [], [], [], [], [], [])

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            question_id_list.append(sample["question_id"])
            instance_id_list.append(sample["instance_id"])
            choices_list.append(sample["choices"])
            correct_choice_idx_list.append(sample["correct_choice_idx"])
            direct_answers_list.append(sample["direct_answers"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "question_id": question_id_list,
            "instance_id": instance_id_list,
            "choices": choices_list,
            "correct_choice_idx": correct_choice_idx_list,
            "direct_answers": direct_answers_list,
        }
    

    def __getitem__(self, index):
        ann = self.annotation[index]
        
        split = ann["split"]
        # if split == "val":
            # image_name = ann["image"]
        # else:
        image_name =  f"{split}2017/{ann['image_id']:012d}.jpg"
        image_path = os.path.join(self.vis_root, image_name)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        choices = ann["choices"]
        if "correct_choice_idx" in ann:
            correct_choice_idx = ann["correct_choice_idx"]
        else:
            correct_choice_idx = None

        if "direct_answers" in ann:
            direct_answers = ann["direct_answers"]
        else:
            direct_answers = None

        return {
            "image": image,
            "text_input": question,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
            "text_output": choices,
            "correct_choice_idx": correct_choice_idx,
            "direct_answers": direct_answers,
        }

class AOKVQAMCEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root

        self.annotation = json.load(open(ann_paths[0]))

        answer_list_path = ann_paths[1]
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
        else:
            self.answer_list = None

        try:
            self.coco_fmt_qust_file = ann_paths[2]
            self.coco_fmt_anno_file = ann_paths[3]
        except IndexError:
            self.coco_fmt_qust_file = None
            self.coco_fmt_anno_file = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor


        self._add_instance_ids()

 
    def __getitem__(self, index):
        ann = self.annotation[index]
        
        split = ann["split"]
        image_name =  f"{split}2017/{ann['image_id']:012d}.jpg"
        image_path = os.path.join(self.vis_root, image_name)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        # text_input = [self.text_processor(f"Answer: {c}") for c in ann["choices"]]
        text_input = [self.text_processor(f"{ann['question']} Answer: {c}") for c in ann["choices"]]

        label = ann.get("correct_choice_idx", None)
        return {
            "image": image,
            "text_input": text_input,
            "image_id": index,
            "instance_id": ann["instance_id"],
            "label": label,
        }
    
    def collater(self, samples):
        (
            image_list,
            input_list,
            image_id_list,
            instance_id_list,
            label_list,
        ) = ([], [], [], [], [])

        for sample in samples:
            image_list.append(sample["image"])
            input_list.append(sample["text_input"])
            image_id_list.append(sample["image_id"])
            label_list.append(sample["label"])
            instance_id_list.append(sample["instance_id"])

        to_return = {
            "image": torch.stack(image_list, dim=0),
            "text_input": input_list,
            "image_id": torch.tensor(image_id_list, dtype=torch.int),
            "instance_id": instance_id_list,
        }

        if label_list[0] is not None:
            to_return.update({
                "label": torch.LongTensor(label_list)
            })

        return to_return

class AOKVQAGenerationDataset(CaptionDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, info):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
     
    def __getitem__(self, index):
        ann = self.annotation[index]
        
        split = ann["split"]
        image_name =  f"{split}2017/{ann['image_id']:012d}.jpg"
        image_path = os.path.join(self.vis_root, image_name)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        
        
        answer = ann["choices"][ann["correct_choice_idx"]]

        # include rationale randomly:
        rationale = random.choice(ann['rationales'])
        if random.random() > 0.5:
            text_input = f"{ann['question']} Long Answer:"
            text_output = f"{answer} Rationale: {rationale}"
        else:
            text_input = f"{ann['question']} Answer:"
            text_output = answer
        text_input = self.text_processor(text_input)
        text_output = self.text_processor(text_output, add_prompt=False)
        
        return {
            "image": image,
            "text_input": text_input,
            "text_output": text_output,
            "image_id": ann["image_id"],
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }

class AOKVQAGenerationEvalDataset(CaptionEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, info):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root

        self.annotation = json.load(open(ann_paths[0]))
        answer_list_path = ann_paths[1]
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
        else:
            self.answer_list = None

        try:
            self.coco_fmt_qust_file = ann_paths[2]
            self.coco_fmt_anno_file = ann_paths[3]
        except IndexError:
            self.coco_fmt_qust_file = None
            self.coco_fmt_anno_file = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.is_mc = info.get('is_mc', False)
        
        self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        
        ann = self.annotation[index]
        
        split = ann["split"]
        image_name =  f"{split}2017/{ann['image_id']:012d}.jpg"
        image_path = os.path.join(self.vis_root, image_name)
        image = Image.open(image_path).convert("RGB")
        
        image = self.vis_processor(image)
        
        text_input = f"{ann['question']} Answer:"
        text_input = self.text_processor(text_input)

        text_output = [self.text_processor(t, add_prompt=False) for t in ann['choices']]

        direct_answers = ann.get('direct_answers', None)
        label = ann.get("correct_choice_idx", None)

        if "direct_answers" in ann:
            direct_answers = ann["direct_answers"]
        else:
            direct_answers = None

        return {
            "image": image,
            "text_input": text_input,
            'text_output': text_output,
            "image_id": ann["image_id"],
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],

            "choices": ann["choices"],
            "direct_answers": direct_answers,
            "label": label,

        }

    def collater(self, samples):
        (
            image_list,
            input_list,
            output_list,  # mc choice
            answer_list,
            choice_list,
            image_id_list,
            question_id_list,
            instance_id_list,
            label_list,
        ) = ([], [], [], [], [], [], [], [], [])


        for sample in samples:
            image_list.append(sample["image"])
            input_list.append(sample["text_input"])
            output_list.append(sample["text_output"])
            answer_list.append(sample["direct_answers"])
            choice_list.append(sample["choices"])
            image_id_list.append(sample["image_id"])
            question_id_list.append(sample["question_id"])
            instance_id_list.append(sample["instance_id"])
            label_list.append(sample["label"])

        to_return = {
            "image": torch.stack(image_list, dim=0),
            "text_input": input_list,
            "text_output": output_list,
            "choices": choice_list,
            "image_id": torch.tensor(image_id_list, dtype=torch.int),
            "question_id": question_id_list,
            "instance_id": instance_id_list,
            "direct_answers": answer_list,
            "label": label_list,
        }

        if label_list[0] is not None:
            to_return.update({
                "label": torch.LongTensor(label_list)
            })

        return to_return

# class AOKVQAGenerationDataset(CaptionDataset, __DisplMixin):
#     def __init__(self, vis_processor, text_processor, vis_root, ann_paths, direct_answer_weight_threshold=0.2):
#         super().__init__(vis_processor, text_processor, vis_root, ann_paths)

#         # question-direct answers as separate instances
#         # for ann in self.annotation:
#         #     answer_weight = {}
#         #     for answer in ann["direct_answers"]:
#         #         if answer in answer_weight.keys():
#         #             answer_weight[answer] += 1 / len(ann["direct_answers"])
#         #         else:
#         #             answer_weight[answer] = 1 / len(ann["direct_answers"])
#         #     for answer, weight in answer_weight.items():
#         #         if weight > direct_answer_weight_threshold:
#         #             datum = deepcopy(ann)
#         #             datum.update({
#         #                 "answer": answer,
#         #                 "weight": weight 
#         #             })
#         #             data.append(datum)
#         # self.annotation = data
        
#         # captions
#         self.im2caption = {}
#         for caption_split in ['train', 'val']:
#             caption_annotation = json.load(open(os.path.join(self.vis_root, f'annotations/captions_{caption_split}2017.json')))['annotations']
#             for annot in caption_annotation:
#                 self.im2caption[annot['image_id']] = annot['caption']
     
#     def __getitem__(self, index):
#         ann = self.annotation[index]
        
#         split = ann["split"]
#         image_name =  f"{split}2017/{ann['image_id']:012d}.jpg"
#         image_path = os.path.join(self.vis_root, image_name)
#         image = Image.open(image_path).convert("RGB")

#         image = self.vis_processor(image)
#         caption = self.im2caption[ann['image_id']]
#         answer = ann["choices"][ann["correct_choice_idx"]]
#         qa = f'{ann["question"]} Answer: {answer}'

#         return {
#             "image": image,
#             "text_input": caption,
#             "answer": qa,
#             "image_id": ann["image_id"],
#             "question_id": ann["question_id"],
#             "instance_id": ann["instance_id"],
#         }

# class AOKVQAGenerationEvalDataset(CaptionEvalDataset, __DisplMixin):
#     def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
#         """
#         vis_root (string): Root directory of images (e.g. coco/images/)
#         ann_root (string): directory to store the annotation file
#         """

#         self.vis_root = vis_root

#         self.annotation = json.load(open(ann_paths[0]))
#         answer_list_path = ann_paths[1]
#         if os.path.exists(answer_list_path):
#             self.answer_list = json.load(open(answer_list_path))
#         else:
#             self.answer_list = None

#         try:
#             self.coco_fmt_qust_file = ann_paths[2]
#             self.coco_fmt_anno_file = ann_paths[3]
#         except IndexError:
#             self.coco_fmt_qust_file = None
#             self.coco_fmt_anno_file = None

#         self.vis_processor = vis_processor
#         self.text_processor = text_processor

#         self.im2caption = {}
#         for caption_split in ['train', 'val']:
#             caption_annotation = json.load(open(os.path.join(self.vis_root, f'annotations/captions_{caption_split}2017.json')))['annotations']
#             for annot in caption_annotation:
#                 self.im2caption[annot['image_id']] = annot['caption']
        
#         self._add_instance_ids()

#     def __len__(self):
#         return len(self.annotation)

#     def __getitem__(self, index):
        
#         ann = self.annotation[index]
        
#         split = ann["split"]
#         image_name =  f"{split}2017/{ann['image_id']:012d}.jpg"
#         image_path = os.path.join(self.vis_root, image_name)
#         image = Image.open(image_path).convert("RGB")
        
#         image = self.vis_processor(image)
#         caption = self.im2caption[ann['image_id']]
#         answer = ann["choices"][ann["correct_choice_idx"]]
#         qa = f'{ann["question"]} Answer: {answer}'

#         return {
#             "image": image,
#             "text_input": caption,
#             "answer": qa,
#             "image_id": ann["image_id"],
#             "question_id": ann["question_id"],
#             "instance_id": ann["instance_id"],
#         }