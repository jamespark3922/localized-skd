"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures


@registry.register_model("blip2_critic")
class Blip2Critic(Blip2Base):
    """
    BLIP2 Classifier using the first-stage model with Q-former and ViT.
    Only use the ITM head to perform classification.

    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        max_txt_len=32,
        qa_weight=0.5,
        r_weight=0.5
    ):

        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])
        for name, param in self.Qformer.cls.named_parameters():
            param.requires_grad = False

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.qa_head = nn.Sequential(
            nn.Linear(self.Qformer.config.hidden_size, self.Qformer.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.Qformer.config.hidden_size, 1),
        )
        self.qa_weight = qa_weight

        self.r_head = nn.Sequential(
            nn.Linear(self.Qformer.config.hidden_size, self.Qformer.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.Qformer.config.hidden_size, 1),
        )
        self.r_weight = r_weight

        self.max_txt_len = max_txt_len

    def forward(self, samples, is_train=True):
        image = samples["image"]
        text = samples["text_input"]

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        
        text_ids = text_tokens.input_ids
        text_atts = text_tokens.attention_mask

        query_tokens_itm = self.query_tokens.expand(text_ids.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            image.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts], dim=1)

        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        prediction = F.softmax(logits, dim=1) #[:,]

        # multi-task learning
        if "label" in samples:
            targets = samples["label"]
            label_loss = F.cross_entropy(logits, targets)

            if is_train:
                qa_logits = self.qa_head(vl_embeddings).mean(dim=1).squeeze(1)
                qa = samples['qa_label']
                qa_loss = F.mse_loss(qa_logits, qa)

                r_logits = self.r_head(vl_embeddings).mean(dim=1).squeeze(1)
                r = samples['r_label']
                r_loss = F.mse_loss(r_logits, qa)

                loss = label_loss + self.qa_weight * qa_loss + self.r_weight * r_loss
            else:
                loss = label_loss

            return {"loss": loss, "prediction": prediction}
        else:
            return {"prediction": prediction}

    @torch.no_grad()
    def predict(self, samples):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        return self.forward(samples, is_train=False) 

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        qa_weight = cfg.get("qa_weight", 0.5)
        r_weight = cfg.get("r_weight", 0.5)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
            qa_weight=qa_weight,
            r_weight=r_weight,
        )
        model.load_checkpoint_from_config(cfg)

        return model