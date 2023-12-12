"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.wit_datasets import (
    WitDataset,
    WitEvalDataset
)

from lavis.common.registry import registry

@registry.register_builder("wit_retrieval")
class WitBuilder(BaseDatasetBuilder):
    train_dataset_cls = WitDataset
    eval_dataset_cls = WitEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/wit/defaults.yaml"}