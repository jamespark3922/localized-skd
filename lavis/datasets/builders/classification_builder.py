"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.nlvr_datasets import NLVRDataset, NLVREvalDataset
from lavis.datasets.datasets.snli_ve_datasets import SNLIVisualEntialmentDataset, SNLIVisualEntailmentMCDataset
from lavis.datasets.datasets.swig_datasets import SWIGDataset, SWIGEvalDataset

@registry.register_builder("nlvr")
class NLVRBuilder(BaseDatasetBuilder):
    train_dataset_cls = NLVRDataset
    eval_dataset_cls = NLVREvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/nlvr/defaults.yaml"}


@registry.register_builder("snli_ve")
class SNLIVisualEntailmentBuilder(BaseDatasetBuilder):
    train_dataset_cls = SNLIVisualEntialmentDataset
    eval_dataset_cls = SNLIVisualEntialmentDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/snli_ve/defaults.yaml"}

@registry.register_builder("snli_ve_mc")
class SNLIVisualEntailmentMCBuilder(BaseDatasetBuilder):
    train_dataset_cls = SNLIVisualEntailmentMCDataset
    eval_dataset_cls = SNLIVisualEntailmentMCDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/snli_ve/defaults_mc.yaml"}

@registry.register_builder("swig")
class SWIGBuilder(BaseDatasetBuilder):
    train_dataset_cls = SWIGDataset
    eval_dataset_cls = SWIGEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/swig/defaults.yaml"}
