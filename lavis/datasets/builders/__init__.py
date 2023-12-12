"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import load_dataset_config
from lavis.datasets.builders.caption_builder import (
    COCOCapBuilder,
    MSRVTTCapBuilder,
    MSVDCapBuilder,
    VATEXCapBuilder,
    VisualCometCapBuilder,
    SherlockCapBuilder
)
from lavis.datasets.builders.image_text_pair_builder import (
    ConceptualCaption12MBuilder,
    ConceptualCaption3MBuilder,
    VGCaptionBuilder,
    SBUCaptionBuilder,
    LaionSampleBuilder,
)
from lavis.datasets.builders.classification_builder import (
    NLVRBuilder,
    SNLIVisualEntailmentBuilder,
    SNLIVisualEntailmentMCBuilder,
    SWIGBuilder,
)
from lavis.datasets.builders.imagefolder_builder import ImageNetBuilder
from lavis.datasets.builders.video_qa_builder import MSRVTTQABuilder, MSVDQABuilder
from lavis.datasets.builders.vqa_builder import (
    COCOVQABuilder,
    OKVQABuilder,
    VGVQABuilder,
    GQABuilder,
    AOKVQABuilder,
    AOKVQAMCBuilder,
    V7WVQA_Builder
)
from lavis.datasets.builders.unified_vqa_builder import (
    UnifiedVQABuilder,
    UnifiedVQAChatGPTBuilder,
    UnifiedVQAChatGPTContrastiveBuilder,
    UnifiedGenerativeBuilder,
    UnifiedContrastiveBuilder,
    UnifiedVQARankBuilder,
    UnifiedVQABinaryBuilder,


    LLAVABuilder,
    LLAVAContrastiveBuilder
 )
from lavis.datasets.builders.rank_builder import (
    VisualCometRankDataset,
)
from lavis.datasets.builders.retrieval_builder import (
    MSRVTTRetrievalBuilder,
    DiDeMoRetrievalBuilder,
    COCORetrievalBuilder,
    Flickr30kBuilder,
)

from lavis.datasets.builders.explanation_builder import (
    ExplanationBuilder,
    ExplanationVCREvalBuilder,
)

from lavis.datasets.builders.dialogue_builder import AVSDDialBuilder

from lavis.datasets.builders.wit_builder import WitBuilder
from lavis.datasets.builders.sherlock_builder import SherlockBuilder
from lavis.datasets.builders.visualcomet_mc_builder import VisualCometMCBuilder

from lavis.common.registry import registry

__all__ = [
    "COCOCapBuilder",
    "COCORetrievalBuilder",
    "COCOVQABuilder",
    "ConceptualCaption12MBuilder",
    "ConceptualCaption3MBuilder",
    "DiDeMoRetrievalBuilder",
    "Flickr30kBuilder",
    "GQABuilder",
    "ImageNetBuilder",
    "LaionSampleBuilder",
    "MSRVTTCapBuilder",
    "MSRVTTQABuilder",
    "MSRVTTRetrievalBuilder",
    "MSVDCapBuilder",
    "MSVDQABuilder",
    "NLVRBuilder",
    "OKVQABuilder",
    "SBUCaptionBuilder",
    "SNLIVisualEntailmentBuilder",
    "SNLIVisualEntailmentMCBuilder",
    "SWIGBuilder",
    "VATEXCapBuilder",
    "VGCaptionBuilder",
    "VGVQABuilder",
    "V7WVQA_Builder",
    "AOKVQABuilder",
    "AOKVQAMCBuilder",

    "UnifiedVQABuilder",
    "UnifiedVQAChatGPTBuilder",
    "UnifiedVQAChatGPTContrastiveBuilder",

    "UnifiedVQARankBuilder",
    "UnifiedVQABinaryBuilder",

    "UnifiedGenerativeBuilder",
    "UnifiedContrastiveBuilder",
    
    "ExplanationBuilder",
    "ExplanationVCREvalBuilder",
    "AVSDDialBuilder",
    "WitBuilder",
    "SherlockBuilder",
    'VisualCometMCBuilder'
]


def load_dataset(name, cfg_path=None, vis_path=None, data_type=None):
    """
    Example

    >>> dataset = load_dataset("coco_caption", cfg=None)
    >>> splits = dataset.keys()
    >>> print([len(dataset[split]) for split in splits])

    """
    if cfg_path is None:
        cfg = None
    else:
        cfg = load_dataset_config(cfg_path)

    try:
        builder = registry.get_builder_class(name)(cfg)
    except TypeError:
        print(
            f"Dataset {name} not found. Available datasets:\n"
            + ", ".join([str(k) for k in dataset_zoo.get_names()])
        )
        exit(1)

    if vis_path is not None:
        if data_type is None:
            # use default data type in the config
            data_type = builder.config.data_type

        assert (
            data_type in builder.config.build_info
        ), f"Invalid data_type {data_type} for {name}."

        builder.config.build_info.get(data_type).storage = vis_path

    dataset = builder.build_datasets()
    return dataset


class DatasetZoo:
    def __init__(self) -> None:
        self.dataset_zoo = {
            k: list(v.DATASET_CONFIG_DICT.keys())
            for k, v in sorted(registry.mapping["builder_name_mapping"].items())
        }

    def get_names(self):
        return list(self.dataset_zoo.keys())


dataset_zoo = DatasetZoo()
