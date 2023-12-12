"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
import logging

import numpy as np
import torch

from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask

from sklearn.metrics import precision_recall_fscore_support


@registry.register_task("multimodal_classification")
class MultimodalClassificationTask(BaseTask):
    def __init__(self, report_metric=True):
        self.report_metric = report_metric
        super().__init__()

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            report_metric=report_metric,
        )

    def valid_step(self, model, samples):
        results = []

        outputs = model.predict(samples)
        predictions = outputs["prediction"]
        scores = predictions[:, 1].cpu().numpy()
        predictions = predictions.max(1)[1].cpu().numpy()

        reformat_keys = {
            'text_input': 'text',
            'image_id': 'index',
            'instance_id': 'instance'
        }

        for i in range(len(predictions)):
            result_dict = {}
             
            for key in samples.keys():

                if key in ['image']:
                    continue
                value = samples[key][i]
                if isinstance(value, torch.Tensor):
                    value = value.cpu().numpy()
                if isinstance(value, np.ndarray) and value.size == 1:
                    value = value.item()

                # Preserve original key format
                if key in reformat_keys:
                    result_dict[reformat_keys[key]] = value
                else:
                    result_dict[key] = value

            result_dict["prediction"] = predictions[i].item()
            result_dict["score"] = scores[i].item()

            results.append(result_dict)

        return results
    
    # def valid_step(self, model, samples):
    #     results = []

    #     outputs = model.predict(samples)

    #     predictions = outputs["prediction"]
    #     scores = predictions[:,1].cpu().numpy()
    #     predictions = predictions.max(1)[1].cpu().numpy()
    #     texts = samples['text_input']
    #     indices = samples['image_id']
    #     instances = samples['instance_id']

    #     if 'label' in samples:
    #         targets = samples["label"]
    #         targets = targets.cpu().numpy()
    #         for pred, score, tgt, text, instance, index in zip(predictions, scores, targets, texts, instances, indices):
    #             if isinstance(index, torch.Tensor):
    #                 index = index.item()

    #             results.append(
    #                 {
    #                     "index": index,
    #                     "instance": instance,
    #                     'text': text,
    #                     "prediction": pred.item(),
    #                     "target": tgt.item(),
    #                     "score": score.item(),
    #                 }
    #             )
    #     else:
    #         for pred, score, text, instance, index in zip(predictions, scores, texts, instances, indices):
    #             if isinstance(index, torch.Tensor):
    #                 index = index.item()

    #             results.append(
    #                 {
    #                     "index": index,
    #                     'text': text,
    #                     "instance": instance,
    #                     "prediction": pred.item(),
    #                     "score": score.item(),
    #                 }
    #             )
    
    #     return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="index",
        )

        if self.report_metric:
            metrics = self._report_metrics(
                eval_result_file=eval_result_file, split_name=split_name
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        results = json.load(open(eval_result_file))

        predictions = np.array([res["prediction"] for res in results])
        targets = np.array([res["target"] for res in results])

        accuracy = (targets == predictions).sum() / targets.shape[0]
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        log_stats = {split_name: {k: v for k, v in metrics.items()}}

        p,r,f1,_ = precision_recall_fscore_support(predictions, targets, average='weighted')

        metrics['precision'] = p
        metrics['recall'] = r
        metrics['f1'] = f1
    
        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        logging.info(metrics)
        return metrics
