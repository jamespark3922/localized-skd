"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.visualcomet_rank_datasets import (
    VisualCometRankDataset, VisualCometRankEvalDataset
)

from lavis.common.registry import registry

@registry.register_builder("visualcomet_rank")
class VisualCometMCBuilder(BaseDatasetBuilder):
    train_dataset_cls = VisualCometRankDataset
    eval_dataset_cls = VisualCometRankDataset # VisualCometRankEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/rank_visualcomet/rank_vc_defaults.yaml",
        "polygon": "configs/datasets/rank_visualcomet/rank_vc_polygon.yaml",
        "crop": "configs/datasets/rank_visualcomet/rank_vc_crop.yaml",

        "action": "configs/datasets/rank_visualcomet/rank_vc_action.yaml"
    }