"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.coco_caption_datasets import (
    COCOCapDataset,
    COCOCapEvalDataset,
    NoCapsEvalDataset,
)

from lavis.datasets.datasets.visualcomet_caption_datasets import (
    VisualCometCaptionDataset,
    VisualCometCaptionEvalDataset
)

from lavis.datasets.datasets.visualcomet_inference_datasets import (
    VisualCometInferenceDataset,
    VisualCometInferenceEvalDataset
)

from lavis.datasets.datasets.sherlock_caption_datasets import (
    SherlockCaptionDataset,
    SherlockCaptionEvalDataset
)

from lavis.datasets.datasets.aok_vqa_datasets import (
    AOKVQAGenerationDataset,
    AOKVQAGenerationEvalDataset
)

# video
from lavis.common.registry import registry
from lavis.datasets.datasets.video_caption_datasets import (
    VideoCaptionDataset,
    VideoCaptionEvalDataset,
)




@registry.register_builder("coco_caption")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset
    eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_cap.yaml",
    }


@registry.register_builder("nocaps")
class COCOCapBuilder(BaseDatasetBuilder):
    eval_dataset_cls = NoCapsEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nocaps/defaults.yaml",
    }

@registry.register_builder("visualcomet_caption")
class VisualCometCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VisualCometCaptionDataset
    eval_dataset_cls = VisualCometCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/visualcomet_mc/defaults_cap.yaml",
        "inference": "configs/datasets/visualcomet_mc/defaults_cap_inference.yaml",
    }

@registry.register_builder("visualcomet_inference")
class VisualCometInfBuilder(BaseDatasetBuilder):
    train_dataset_cls = VisualCometInferenceDataset
    eval_dataset_cls = VisualCometInferenceEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/visualcomet_inf/defaults.yaml",
        "others": "configs/datasets/visualcomet_inf/others.yaml",
        "polygon": "configs/datasets/visualcomet_inf/polygon.yaml",
        "event": "configs/datasets/visualcomet_inf/event.yaml",
        "polygon_event": "configs/datasets/visualcomet_inf/polygon_event.yaml",
    }


@registry.register_builder("sherlock_caption")
class SherlockCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = SherlockCaptionDataset
    eval_dataset_cls = SherlockCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/sherlock/defaults_cap.yaml",
        "inference": "configs/datasets/sherlock/inference_cap.yaml",
    }


@registry.register_builder("msrvtt_caption")
class MSRVTTCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/defaults_cap.yaml",
    }


@registry.register_builder("msvd_caption")
class MSVDCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msvd/defaults_cap.yaml",
    }


@registry.register_builder("vatex_caption")
class VATEXCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vatex/defaults_cap.yaml",
    }

@registry.register_builder("aok_vqa_gen")
class AOKVQAGenBuilder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQAGenerationDataset
    eval_dataset_cls = AOKVQAGenerationEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/aokvqa_gen/defaults.yaml",
        "default": "configs/datasets/aokvqa_gen/mc.yaml",
    }
