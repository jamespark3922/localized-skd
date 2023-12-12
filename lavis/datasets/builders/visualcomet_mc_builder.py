"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.visualcomet_mc_datasets import (
    VisualCometMCDataset,
    VisualCometMCEvalDataset
)

from lavis.common.registry import registry

@registry.register_builder("visualcomet_mc")
class VisualCometMCBuilder(BaseDatasetBuilder):
    train_dataset_cls = VisualCometMCDataset
    eval_dataset_cls = VisualCometMCEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/visualcomet_mc/defaults.yaml",
        "no_region": "configs/datasets/visualcomet_mc/no_region.yaml",
        "no_widescreen": "configs/datasets/visualcomet_mc/no_widescreen.yaml",
    }