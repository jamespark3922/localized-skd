"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry

from lavis.datasets.datasets.unified_vqa_datasets import UnifiedVQAGenDataset, UnifiedVQAGenEvalDataset
from lavis.datasets.datasets.unified_vqa_chatgpt_datasets import (
    UnifiedVQAChatGPTDataset, UnifiedVQAChatGPTEvalDataset, UnifiedVQAChatGPTContrastiveDataset, UnifiedVQAChatGPTContrastiveEvalDataset
)
from lavis.datasets.datasets.unified_vqa_rank_datasets import UnifiedVQARankDataset, UnifiedVQARankGenerativeDataset
from lavis.datasets.datasets.unified_vqa_binary_datasets import UnifiedVQABinaryDataset, UnifiedVQABinaryEvalDataset
from lavis.datasets.datasets.visualcomet_datasets import VisualCometDataset
from lavis.datasets.datasets.unified_train_eval_datasets import UnifiedGenerativeDataset, UnifiedGenerativeEvalDataset, UnifiedContrastiveDataset

from lavis.datasets.datasets.llava_datasets import  LLAVADataset, LLAVAContrastiveDataset


@registry.register_builder("llava")
class LLAVABuilder(BaseDatasetBuilder):
    train_dataset_cls = LLAVADataset
    eval_dataset_cls = UnifiedVQAChatGPTEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/llava/llava_defaults.yaml",
    }

@registry.register_builder("llava_contrastive")
class LLAVAContrastiveBuilder(BaseDatasetBuilder):
    train_dataset_cls = LLAVAContrastiveDataset
    eval_dataset_cls = UnifiedContrastiveDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/llava/llava_contrastive.yaml",
    }

@registry.register_builder("unified_vqa")
class UnifiedVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = UnifiedVQAGenDataset
    eval_dataset_cls = UnifiedVQAGenEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/unified_vqa/unified_vqa_defaults_boxes.yaml",
        "segms": "configs/datasets/unified_vqa/unified_vqa_segms.yaml",
        "debug": "configs/datasets/unified_vqa/unified_vqa_debug.yaml",
        "vcr_eval": "configs/datasets/unified_vqa/unified_vqa_defaults_vcr_eval.yaml",
        "default_eval": "configs/datasets/unified_vqa/unified_vqa_defaults_eval.yaml",
    }

@registry.register_builder("unified_vqa_chatgpt")
class UnifiedVQAChatGPTBuilder(BaseDatasetBuilder):
    train_dataset_cls = UnifiedVQAChatGPTDataset
    eval_dataset_cls = UnifiedVQAChatGPTEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/unified_vqa_chatgpt/unified_vqa_chatgpt_defaults.yaml",
        "debug": "configs/datasets/unified_vqa_chatgpt/unified_vqa_chatgpt_defaults_debug.yaml",
        "unfiltered": "configs/datasets/unified_vqa_chatgpt/unified_vqa_chatgpt_defaults_unfiltered.yaml",
        "unfiltered_subset": "configs/datasets/unified_vqa_chatgpt/unified_vqa_chatgpt_defaults_unfiltered_subset.yaml",
        
        "openimages": "configs/datasets/unified_vqa_chatgpt/openimages_chatgpt_defaults.yaml",
        "openimages_subset": "configs/datasets/unified_vqa_chatgpt/openimages_chatgpt_defaults_subset.yaml",
        
        "gt": "configs/datasets/unified_vqa_chatgpt/vc_sherlock_chatgpt_defaults.yaml",
    }

@registry.register_builder("unified_vqa_chatgpt_contrastive")
class UnifiedVQAChatGPTContrastiveBuilder(BaseDatasetBuilder):
    train_dataset_cls = UnifiedVQAChatGPTContrastiveDataset
    # eval_dataset_cls = UnifiedVQAChatGPTContrastiveEvalDataset
    eval_dataset_cls = UnifiedContrastiveDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/unified_vqa_chatgpt/unified_vqa_chatgpt_contrastive.yaml",
        "debug": "configs/datasets/unified_vqa_chatgpt/unified_vqa_chatgpt_contrastive_debug.yaml",
        "unfiltered": "configs/datasets/unified_vqa_chatgpt/unified_vqa_chatgpt_contrastive_unfiltered.yaml",
        "unfiltered_subset": "configs/datasets/unified_vqa_chatgpt/unified_vqa_chatgpt_contrastive_unfiltered_subset.yaml",

        "openimages": "configs/datasets/unified_vqa_chatgpt/openimages_chatgpt_contrastive.yaml",
        "openimages_subset": "configs/datasets/unified_vqa_chatgpt/openimages_chatgpt_contrastive_subset.yaml",
        
        "gt": "configs/datasets/unified_vqa_chatgpt/vc_sherlock_chatgpt_contrastive.yaml",
    }

@registry.register_builder("unified_generative")
class UnifiedGenerativeBuilder(BaseDatasetBuilder):
    train_dataset_cls = UnifiedGenerativeDataset
    eval_dataset_cls = UnifiedGenerativeEvalDataset

    DATASET_CONFIG_DICT = {
        # "default": "configs/datasets/unified_vqa/unified_vqa_defaults_vcr_eval.yaml",
        "vcr_train": "configs/datasets/unified_generative/vcr_defaults_train.yaml",
        "sherlock_train": "configs/datasets/unified_generative/sherlock_defaults_train.yaml",
        "visualcomet_train": "configs/datasets/unified_generative/visualcomet_defaults_train.yaml",

        "vcr_eval": "configs/datasets/unified_generative/vcr_defaults_eval.yaml",
        "vcr_generative_eval": "configs/datasets/unified_generative/vcr_generative_eval.yaml",

        "sherlock_eval": "configs/datasets/unified_generative/sherlock_defaults_eval.yaml",
        "visualcomet_ranking_eval": "configs/datasets/unified_generative/visualcomet_ranking_eval.yaml" ,
        "visualcomet_gen_eval": "configs/datasets/unified_generative/visualcomet_gen_eval.yaml" ,


    }

@registry.register_builder("unified_contrastive")
class UnifiedContrastiveBuilder(BaseDatasetBuilder):
    train_dataset_cls = UnifiedContrastiveDataset
    eval_dataset_cls = UnifiedContrastiveDataset

    DATASET_CONFIG_DICT = {
        # "default": "configs/datasets/unified_vqa/unified_vqa_defaults_vcr_eval.yaml",
        "vcr_train": "configs/datasets/unified_contrastive/vcr_defaults_train.yaml",
        "sherlock_train": "configs/datasets/unified_contrastive/sherlock_defaults_train.yaml",
        "visualcomet_train": "configs/datasets/unified_contrastive/visualcomet_defaults_train.yaml",
        
        "vcr_train_subset": "configs/datasets/unified_contrastive/vcr_defaults_train_subset.yaml",
        "sherlock_train_subset": "configs/datasets/unified_contrastive/sherlock_defaults_train_subset.yaml",
    
        "vcr_eval": "configs/datasets/unified_contrastive/vcr_defaults_eval.yaml",
        "sherlock_eval": "configs/datasets/unified_contrastive/sherlock_defaults_eval.yaml" ,
        "visualcomet_ranking_eval": "configs/datasets/unified_contrastive/visualcomet_ranking_eval.yaml" ,
    }

@registry.register_builder("unified_vqa_rank")
class UnifiedVQARankBuilder(BaseDatasetBuilder):
    train_dataset_cls = UnifiedVQARankDataset
    eval_dataset_cls = UnifiedVQARankDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/unified_vqa_rank/unified_vqa_rank_defaults_boxes.yaml",
        "vcr": "configs/datasets/unified_vqa_rank/unified_vqa_rank_defaults_boxes_vcr.yaml",
        "segms": "configs/datasets/unified_vqa_rank/unified_vqa_rank_segms.yaml",
        "debug": "configs/datasets/unified_vqa_rank/unified_vqa_rank_defaults_boxes_debug.yaml",
    }

@registry.register_builder("unified_vqa_binary")
class UnifiedVQABinaryBuilder(BaseDatasetBuilder):
    train_dataset_cls = UnifiedVQABinaryDataset
    eval_dataset_cls = UnifiedVQABinaryEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/unified_vqa_binary/unified_vqa_binary_defaults.yaml",
        "multi_class": "configs/datasets/unified_vqa_binary/unified_vqa_multi_class_classify.yaml",
        "default_classify_filtering": "configs/datasets/unified_vqa_binary/unified_vqa_binary_classify_balanced.yaml",
        "default_classify": "configs/datasets/unified_vqa_binary/unified_vqa_binary_classify.yaml",
        "multi_class_classify": "configs/datasets/unified_vqa_binary/unified_vqa_multi_class_classify.yaml",

        # openimages
        "openimages_classify": "configs/datasets/unified_vqa_binary/openimages_binary_defaults_classify.yaml",
        "openimages_filtering": "configs/datasets/unified_vqa_binary/openimages_binary_defaults_filtering.yaml",

        "openimages_classify_text_only": "configs/datasets/unified_vqa_binary/openimages_binary_defaults_classify_text_only.yaml",

    }