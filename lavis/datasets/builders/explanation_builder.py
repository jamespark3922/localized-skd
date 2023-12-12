"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry

from lavis.datasets.datasets.explanation_datasets import ExplanationDataset, ExplanationVCREvalDataset

@registry.register_builder("explanation")
class ExplanationBuilder(BaseDatasetBuilder):
    train_dataset_cls = ExplanationDataset
    eval_dataset_cls = ExplanationDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/explanation/explanation_defaults.yaml",
        "vqax": "configs/datasets/explanation/explanation_vqax.yaml",
        "esnlive": "configs/datasets/explanation/explanation_esnlive.yaml",
        "vcr": "configs/datasets/explanation/explanation_vcr.yaml",
    }

@registry.register_builder("explanation_vcr")
class ExplanationVCREvalBuilder(BaseDatasetBuilder):
    train_dataset_cls = ExplanationDataset
    eval_dataset_cls = ExplanationVCREvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/explanation/explanation_vcr_eval.yaml",
    }