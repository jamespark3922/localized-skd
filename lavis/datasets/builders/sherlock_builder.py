"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.sherlock_datasets import (
    SherlockDataset,
    SherlockEvalDataset
)

from lavis.common.registry import registry

@registry.register_builder("sherlock_comparison")
class SherlockBuilder(BaseDatasetBuilder):
    train_dataset_cls = SherlockDataset
    eval_dataset_cls = SherlockEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/sherlock/defaults.yaml",
        "no_region": "configs/datasets/sherlock/no_region.yaml",
        "no_widescreen": "configs/datasets/sherlock/defaults_no_widescreen.yaml",
    }