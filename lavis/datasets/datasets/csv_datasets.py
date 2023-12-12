"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from abc import abstractmethod
from lavis.datasets.datasets.base_dataset import BaseDataset
import pandas as pd

class CSVDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, [])
        
        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(pd.read_csv(ann_path).to_dict(orient='records'))
