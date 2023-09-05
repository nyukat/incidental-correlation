import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import heads, roberta, swin_transformer
from modules.classify import MultimodalClassifier, UnimodalClassifier
from modules.metrics import Accuracy, VQAScore
from modules.roberta import RobertaModel
from modules.swin_helpers import swin_adapt_position_encoding
from modules.vae import Vae, VanillaVAE


class FIBERTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.task = config["task"]

        resolution_after = config["image_size"]
        self.num_fuse_block = config["num_fuse_block"]
        self.num_text_layer = config["num_layers"]
        roberta.NUM_FUSE_BLOCK = swin_transformer.NUM_FUSE_BLOCK = self.num_fuse_block
        roberta.DIM_IMG = config["input_image_embed_size"]
        swin_transformer.DIM_TXT = config["input_text_embed_size"]

        self.cross_modal_text_transform = nn.Linear(config["input_text_embed_size"], config["hidden_size"])
        self.cross_modal_image_transform = nn.Linear(config["input_image_embed_size"], config["hidden_size"])

        self.cross_modal_text_transform_itc = nn.Linear(config["input_text_embed_size"], config["hidden_size"])
        self.cross_modal_image_transform_itc = nn.Linear(config["input_image_embed_size"], config["hidden_size"])

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                getattr(swin_transformer, config["vit"])(
                    pretrained=config["pretrained_vit"],
                    config=config,
                )
                RobertaModel.from_pretrained(config["tokenizer"])

            torch.distributed.barrier()

        self.vit_model = getattr(swin_transformer, config["vit"])(
            pretrained=config["pretrained_vit"],
            config=config,
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.text_transformer = RobertaModel.from_pretrained(config["tokenizer"])

        self.cross_modal_image_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_text_pooler = heads.Pooler(config["hidden_size"])
        self.itc_pooler = config["itc_pooler"]
        if self.itc_pooler:
            self.cross_modal_image_pooler_itc = heads.Pooler(config["hidden_size"])
            self.cross_modal_text_pooler_itc = heads.Pooler(config["hidden_size"])

        if "vqa" in self.task:
            x_dim = 2 * config["hidden_size"]
            image_dim = text_dim = config["hidden_size"]
            y_dim = config["vqav2_label_size"]
        elif "nlvr2" in self.task:
            x_dim = 4 * config["hidden_size"]
            image_dim = 2 * config["hidden_size"]
            text_dim = config["hidden_size"]
            y_dim = 1
        else:
            raise ValueError

        # We need to sum the losses over the targets for VQA. This also works for NLVR2, but since there's only one
        # target, it has a similar effect to squeeze().
        log_prob_fn = lambda y_pred, y_true: -F.binary_cross_entropy_with_logits(y_pred, y_true,
            reduction="none").sum(dim=1)
        nll_fn = lambda y_pred, y_true: F.binary_cross_entropy_with_logits(y_pred, y_true,
            reduction="none").sum(dim=1)

        if config["is_vanilla"]:
            self.vae = VanillaVAE(x_dim, config["hidden_dims"], config["latent_size"], y_dim, config["n_samples"],
                log_prob_fn)
        else:
            self.vae = Vae(x_dim, config["hidden_dims"], config["latent_size"], y_dim, config["n_components"],
                config["n_samples"], log_prob_fn)
        self.multimodal_classifier = MultimodalClassifier(x_dim, config["hidden_dims"], y_dim, nll_fn)
        self.unimodal_classifier = UnimodalClassifier(image_dim, text_dim, config["hidden_dims"], y_dim, nll_fn)

        self.accuracy = Accuracy()
        self.vqa_score = VQAScore()

        exclude_keys = ["image_queue", "text_queue", "queue_ptr", "queue_total", "image_input_queue", "text_input_queue",
            "text_input_mask_queue"]
        if config["load_path"] != "":
            ckpt = torch.load(config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            for key in exclude_keys:
                if key in state_dict:
                    state_dict.pop(key)
            if os.path.basename(config["load_path"]) == "fiber_pretrain.ckpt" and not config["test_only"]:
                state_dict = swin_adapt_position_encoding(
                    state_dict, before=config["resolution_before"], after=resolution_after
                )
            self.load_state_dict(state_dict, strict=False)

    def make_embeds(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        img=None,
        text_only=False,
        image_only=False,
    ):
        if not text_only:
            if img is None:
                if f"image_{image_token_type_idx - 1}" in batch:
                    imgkey = f"image_{image_token_type_idx - 1}"
                else:
                    imgkey = "image"
                img = batch[imgkey][0]

        if not image_only:
            do_mlm = "_mlm" if mask_text else ""
            text_ids = batch[f"text_ids{do_mlm}"]
            text_labels = batch[f"text_labels{do_mlm}"]
            text_masks = batch[f"text_masks"]

        # block attn
        if text_only:
            text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
            device = text_embeds.device
            input_shape = text_masks.size()
            extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, device)
            for layer_i, layer in enumerate(self.text_transformer.encoder.layer):
                text_embeds = layer(text_embeds, extend_text_masks)[0]

            text_embeds = self.cross_modal_text_transform_itc(text_embeds)

            if self.itc_pooler:
                cls_feats_text = self.cross_modal_text_pooler_itc(text_embeds)
            else:
                cls_feats_text = text_embeds[:, 0]

            cls_feats_text = cls_feats_text / cls_feats_text.norm(dim=-1, keepdim=True)

            ret = {
                "text_feats": text_embeds,
                "image_feats": None,
                "cls_feats": cls_feats_text,
                "text_labels": text_labels,
                "text_ids": text_ids,
                "text_masks": text_masks,
                "image": None,
            }

            return ret

        if image_only:
            image_embeds = self.vit_model.patch_embed(img)
            if self.vit_model.absolute_pos_embed is not None:
                image_embeds = image_embeds + self.vit_model.absolute_pos_embed
            image_embeds = self.vit_model.pos_drop(image_embeds)

            for layer_i, layer in enumerate(self.vit_model.layers):
                image_embeds = layer(image_embeds)
            image_embeds = self.vit_model.norm(image_embeds)
            image_embeds = self.cross_modal_image_transform_itc(image_embeds)
            image_feats = image_embeds

            avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
            if self.itc_pooler:
                cls_feats_image = self.cross_modal_image_pooler_itc(avg_image_feats)
            else:
                cls_feats_image = avg_image_feats[:, 0]

            cls_feats_image = cls_feats_image / cls_feats_image.norm(dim=-1, keepdim=True)

            ret = {
                "text_feats": None,
                "image_feats": image_embeds,
                "cls_feats": cls_feats_image,
                "text_labels": None,
                "text_ids": None,
                "text_masks": None,
                "image": None,
            }

            return ret

        image_embeds = self.vit_model.patch_embed(img)
        if self.vit_model.absolute_pos_embed is not None:
            image_embeds = image_embeds + self.vit_model.absolute_pos_embed
        image_embeds = self.vit_model.pos_drop(image_embeds)
        for layer_i, layer in enumerate(self.vit_model.layers[:2]):
            image_embeds = layer(image_embeds)

        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        device = text_embeds.device
        input_shape = text_masks.size()
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, device)
        num_pre_text = self.num_text_layer - self.num_fuse_block
        for layer_i, layer in enumerate(self.text_transformer.encoder.layer[:num_pre_text]):
            text_embeds = layer(text_embeds, extend_text_masks)[0]

        num_pre_block = 8 + num_pre_text
        for blk_cnt, blk in enumerate(self.vit_model.layers[2].blocks):
            if blk_cnt < num_pre_block:
                image_embeds = blk(image_embeds)
            else:
                fuse_image_embeds = blk(image_embeds, text_embeds, extend_text_masks)
                text_embeds = self.text_transformer.encoder.layer[blk_cnt - 8](
                    text_embeds, extend_text_masks, encoder_hidden_states=(image_embeds)
                )[0]
                image_embeds = fuse_image_embeds

        if self.vit_model.layers[2].downsample is not None:
            image_embeds = self.vit_model.layers[2].downsample(image_embeds)

        for blk_cnt, blk in enumerate(self.vit_model.layers[3].blocks):
            fuse_image_embeds = blk(image_embeds, text_embeds, extend_text_masks)
            text_embeds = self.text_transformer.encoder.layer[blk_cnt + 10](
                text_embeds, extend_text_masks, encoder_hidden_states=(image_embeds), last_norm=(blk_cnt == 0)
            )[0]
            image_embeds = fuse_image_embeds

        if self.vit_model.layers[3].downsample is not None:
            image_embeds = self.vit_model.layers[3].downsample(image_embeds)

        text_embeds = self.cross_modal_text_transform(text_embeds)
        image_embeds = self.cross_modal_image_transform(image_embeds)

        cls_feats_text = self.cross_modal_text_pooler(text_embeds)
        avg_image_feats = self.avgpool(image_embeds.transpose(1, 2)).view(image_embeds.size(0), 1, -1)
        cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)
        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)

        ret = {
            "text_feats": text_embeds,
            "image_feats": image_embeds,
            "cls_feats": cls_feats,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "image": img,
        }

        return ret

    def make_vqa_targets(self, batch):
        vqa_labels = batch["vqa_labels"]
        vqa_scores = batch["vqa_scores"]
        vqa_targets = torch.zeros(len(vqa_labels), self.config["vqav2_label_size"]).to(self.device)

        for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
            for l, s in zip(_label, _score):
                vqa_targets[i, l] = s

        return vqa_targets

    def make_nlvr2_targets(self, batch):
        return torch.tensor(batch["answers"]).float().to(self.device)[:, None]

    def forward(self, batch):
        if self.task == "vae_vqav2":
            x = self.make_embeds(batch)["cls_feats"]
            y_true = self.make_vqa_targets(batch)
            out = self.vae(x, y_true)
        elif self.task == "multimodal_classify_vqav2":
            x = self.make_embeds(batch)["cls_feats"]
            y_true = self.make_vqa_targets(batch)
            out = self.multimodal_classifier(x, y_true)
        elif self.task == "unimodal_classify_vqav2":
            x_image = self.make_embeds(batch, image_only=True)["cls_feats"]
            x_text = self.make_embeds(batch, text_only=True)["cls_feats"]
            y_true = self.make_vqa_targets(batch)
            out = self.unimodal_classifier(x_image, x_text, y_true)
        elif self.task == "vae_nlvr2":
            embeds1 = self.make_embeds(batch, image_token_type_idx=1)
            embeds2 = self.make_embeds(batch, image_token_type_idx=2)
            x = torch.cat([embeds1["cls_feats"], embeds2["cls_feats"]], dim=-1)
            y_true = self.make_nlvr2_targets(batch)
            out = self.vae(x, y_true)
        elif self.task == "multimodal_classify_nlvr2":
            embeds1 = self.make_embeds(batch, image_token_type_idx=1)
            embeds2 = self.make_embeds(batch, image_token_type_idx=2)
            x = torch.cat([embeds1["cls_feats"], embeds2["cls_feats"]], dim=-1)
            y_true = self.make_nlvr2_targets(batch)
            out = self.multimodal_classifier(x, y_true)
        elif self.task == "unimodal_classify_nlvr2":
            embeds1 = self.make_embeds(batch, image_only=True, image_token_type_idx=1)
            embeds2 = self.make_embeds(batch, image_only=True, image_token_type_idx=2)
            x_image = torch.cat([embeds1["cls_feats"], embeds2["cls_feats"]], dim=-1)
            x_text = self.make_embeds(batch, text_only=True)["cls_feats"]
            y_true = self.make_nlvr2_targets(batch)
            out = self.unimodal_classifier(x_image, x_text, y_true)
        else:
            raise ValueError
        return out

    def training_step(self, batch, batch_idx):
        out = self(batch)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.log("val_loss", out["loss"], on_step=False, on_epoch=True)
        if "kl" in out:
            self.log("val_kl", out["kl"], on_step=False, on_epoch=True)
        if "vqa" in self.task:
            y_true = self.make_vqa_targets(batch)
            self.vqa_score(out.pop("logits"), y_true)
        elif "nlvr2" in self.task:
            y_true = self.make_nlvr2_targets(batch)
            self.accuracy(out.pop("logits"), y_true)

    def validation_epoch_end(self, outs):
        if "vqa" in self.task:
            self.log("val_score", self.vqa_score.compute())
            self.vqa_score.reset()
        elif "nlvr2" in self.task:
            self.log("val_acc", self.accuracy.compute())
            self.accuracy.reset()

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.log("test_loss", out["loss"], on_step=False, on_epoch=True)
        if "kl" in out:
            self.log("test_kl", out["kl"], on_step=False, on_epoch=True)
        if "vqa" in self.task:
            y_true = self.make_vqa_targets(batch)
            self.vqa_score(out.pop("logits"), y_true)
        elif "nlvr2" in self.task:
            y_true = self.make_nlvr2_targets(batch)
            self.accuracy(out.pop("logits"), y_true)

    def test_epoch_end(self, outs):
        if "vqa" in self.task:
            self.log("test_score", self.vqa_score.compute())
            self.vqa_score.reset()
        elif "nlvr2" in self.task:
            self.log("test_acc", self.accuracy.compute())
            self.accuracy.reset()

    def configure_optimizers(self):
        if "vae" in self.task:
            return torch.optim.Adam(self.vae.parameters(), lr=self.config["learning_rate"])
        elif "multimodal_classify" in self.task:
            return torch.optim.Adam(self.multimodal_classifier.parameters(), lr=self.config["learning_rate"])
        elif "unimodal_classify" in self.task:
            return torch.optim.Adam(self.unimodal_classifier.parameters(), lr=self.config["learning_rate"])
        else:
            raise ValueError