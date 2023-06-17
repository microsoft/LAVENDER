from utils.lib import *
from visbackbone.video_swin import get_vidswin_model


class EncVideo(T.nn.Module):
    def __init__(self, args, hidden_size):
        super().__init__()
        self.swin = get_vidswin_model(args)
        self.latent_feat_size = self.swin.norm.normalized_shape[0]
        self.img_feature_dim = hidden_size
        self.swinbert = getattr(args, 'swinbert', False)
        self.max_size_frame = getattr(args, 'max_size_frame', 6)  # 5
        self.max_size_patch = getattr(args, 'max_size_patch', 14)  # 7

        if not self.swinbert:
            if self.latent_feat_size != self.img_feature_dim:
                self.fc = T.nn.Linear(
                    self.latent_feat_size, self.img_feature_dim)
            else:
                self.fc = None
            self.emb_cls = T.nn.Parameter(
                    0.02*T.randn(1, 1, 1, self.img_feature_dim))
            self.emb_pos = T.nn.Parameter(
                0.02*T.randn(
                    1, 1, 1+self.max_size_patch**2, self.img_feature_dim))
            self.emb_len = T.nn.Parameter(
                0.02*T.randn(
                    1, self.max_size_frame, 1, self.img_feature_dim))
            self.emb_odr = T.nn.Parameter(
                0.02*T.randn(1, 1, 1, self.img_feature_dim))
            self.norm = T.nn.LayerNorm(self.img_feature_dim)
        else:
            self.fc = T.nn.Linear(self.latent_feat_size, 512)
            self.img_embedding = T.nn.Linear(512, self.img_feature_dim)
        self.transform_normalize = None

    def forward(self, img, odr=None, vt_mask=None):
        _B, _T, _C, _H, _W = img.shape
        _h, _w = _H//32, _W//32

        if self.transform_normalize is not None:
            img = self.transform_normalize(img)

        f_img = self.swin(img.transpose(1, 2)).transpose(1, 2)
        f_img = f_img.permute(0, 1, 3, 4, 2).view(
            [_B, _T, _h*_w, self.latent_feat_size])

        if self.fc is not None:
            f_img = self.fc(f_img)

        # for swinbert initialized
        if self.swinbert:
            f_img = self.img_embedding(f_img)
            fake_cls_token = T.zeros(
                (_B, _T, 1, self.img_feature_dim), dtype=f_img.dtype,
                device=f_img.device)
            f_img = T.cat([fake_cls_token, f_img], dim=2)

            m_img = T.ones(_h*_w).long().cuda().unsqueeze(0).unsqueeze(0)
            m_img = m_img.expand([_B, _T, -1]).contiguous()
            fake_cls_mask = T.zeros((_B, _T, 1), dtype=m_img.dtype,
                                    device=m_img.device)
            m_img = T.cat([fake_cls_mask, m_img], dim=2)

            f_img = f_img.view([_B, _T*(1+_h*_w), -1])
            m_img = m_img.view([_B, _T*(1+_h*_w)])
            return f_img, m_img

        f_img = T.cat([self.emb_cls.expand([_B, _T, -1, -1]), f_img], dim=2)
        f_img += self.emb_pos.expand([_B, _T, -1, -1])[:, :, :1+_h*_w, :]

        if odr is not None:
            emb_len = []  # feed order
            for b in range(_B):
                tmp = T.cat([
                    self.emb_len[:, i:i+1, :, :]
                    if i == p else self.emb_odr
                    for i, p in enumerate(odr[b])], dim=1)
                emb_len.append(tmp)
            emb_len = T.cat(emb_len, dim=0)
            f_img += emb_len
        else:
            f_img += self.emb_len.expand([_B, -1, 1+_h*_w, -1])[:, :_T, :, :]

        f_img = self.norm(f_img).view([_B, _T*(1+_h*_w), -1])

        m_img = T.ones(1+_h*_w).long().cuda().unsqueeze(0).unsqueeze(0)
        m_img = m_img.expand([_B, _T, -1]).contiguous()
        if vt_mask is not None:
            m_img = m_img * vt_mask
        m_img = m_img.view([_B, _T*(1+_h*_w)])

        return f_img, m_img


class EncTxt(T.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        bert = transformers.AutoModel.from_pretrained(
            args.txt_backbone)
        self.emb_txt = bert.embeddings
        if args.txt_backbone_embed_only:
            self.txt_trsfr = None
            self.mask_ext = None
        else:
            self.txt_trsfr = bert.encoder
            self.mask_ext = bert.get_extended_attention_mask
        self.size_vocab = bert.config.vocab_size
        del bert

    def get_attn_mask(
            self, mask_txt, attn_mask_type="full",):
        _B, _Lt = mask_txt.shape
        if attn_mask_type == "seq2seq":
            _triangle_mask = T.tril(
                T.ones((_B, _Lt, _Lt), dtype=T.long))
            mask = _triangle_mask
            mask.detach()
            mask = mask.to(device=mask_txt.device)
        else:
            mask = mask_txt
        return mask

    def forward(self, txt, mask_txt=None, token_type_ids=None,
                position_ids=None, attn_mask_type="full"):
        f_txt = self.emb_txt(
            txt, token_type_ids=token_type_ids,
            position_ids=position_ids)
        if self.txt_trsfr is not None:
            if mask_txt is None:
                mask_txt = T.ones_like(txt)
            m_txt = self.get_attn_mask(
                mask_txt,
                attn_mask_type=attn_mask_type)
            m_txt = self.mask_ext(m_txt, m_txt.shape, m_txt.device)
            # safeguard fp16
            m_txt = m_txt.to(dtype=f_txt.dtype)
            out = self.txt_trsfr(f_txt, m_txt, output_attentions=False)
            return out['last_hidden_state']
        else:
            return f_txt


class LAVENDER_Base(T.nn.Module):
    def __init__(self, args, tokzr=None):
        super().__init__()
        self.args = args
        self.enc_txt = EncTxt(args)

        # get fusion encoder
        bert = transformers.AutoModelForMaskedLM.from_pretrained(
            self.args.fusion_encoder)
        if args.fusion_encoder_rand_init:
            config = transformers.AutoConfig.from_pretrained(
                self.args.fusion_encoder)
            bert = transformers.AutoModelForMaskedLM.from_config(config)
        self.hidden_size = bert.config.hidden_size
        self.mask_ext = bert.get_extended_attention_mask
        if isinstance(bert, transformers.RobertaForMaskedLM):
            self.trsfr = bert.roberta.encoder
        else:
            self.trsfr = bert.bert.encoder
        self.config = bert.config
        del bert
        self.enc_img = EncVideo(args, self.hidden_size)
        if args.use_checkpoint:
            self.enc_img = checkpoint_wrapper(
                self.enc_img, offload_to_cpu=True)

        self.tokzr = tokzr
        if tokzr is not None:
            (self.cls_token_id, self.sep_token_id,
             self.pad_token_id, self.mask_token_id,
             self.unk_token_id) = self.tokzr.convert_tokens_to_ids(
                [self.tokzr.cls_token,
                 self.tokzr.sep_token, self.tokzr.pad_token,
                 self.tokzr.mask_token,
                 self.tokzr.unk_token])
            self.true_token_id = self.tokzr.convert_tokens_to_ids(
                ["true"])[0]
            self.false_token_id = self.tokzr.convert_tokens_to_ids(
                ["false"])[0]

    def go_feat(self, img, txt, mask, odr=None,
                vt_mask=None, attn_mask_type="full"):
        feat_img, mask_img = self.enc_img(img, odr, vt_mask)
        feat_txt = self.enc_txt(
            txt, mask_txt=mask,
            attn_mask_type=attn_mask_type)
        mask_txt = mask
        return feat_img, mask_img, feat_txt, mask_txt

    def get_attn_mask(
            self, mask_img, mask_txt,
            attn_mask_type="full", mask_pretxt=None):
        _B, _Lv = mask_img.shape
        _, _Lt = mask_txt.shape
        device = mask_img.device
        if mask_pretxt is not None:
            _, _Ltp = mask_pretxt.shape
            full_mask = T.cat([mask_img, mask_pretxt], dim=1)
        else:
            _Ltp = 0
            full_mask = mask_img
        _L = _Lv + _Ltp + _Lt
        _Lfull = _Lv + _Ltp
        if attn_mask_type == "seq2seq":
            mask = T.zeros((_B, _L, _L), dtype=T.long)
            _triangle_mask = T.tril(
                T.ones((_B, _Lt, _Lt), dtype=T.long))
            full_mask_expand = T.ones((_B, _L, _Lfull), dtype=T.long)
            full_mask_expand = full_mask.unsqueeze(1).expand_as(
                full_mask_expand)
            mask[:, :, :_Lfull].copy_(full_mask_expand)
            mask[:, _Lfull:, _Lfull:].copy_(_triangle_mask)
            mask.detach()
            mask = mask.to(device=device)
        else:
            mask = T.cat([full_mask, mask_txt], dim=1)
        return mask

    def go_cross(
            self, feat_img, mask_img, feat_txt,
            mask_txt, attn_mask_type="full", feat_pretxt=None,
            mask_pretxt=None):
        if feat_pretxt is not None:
            assert mask_pretxt is None
            feat = T.cat(
                [feat_img, feat_pretxt, feat_txt], dim=1)
        else:
            feat = T.cat([feat_img, feat_txt], dim=1)
        mask = self.get_attn_mask(
            mask_img, mask_txt,
            attn_mask_type=attn_mask_type, mask_pretxt=mask_pretxt)
        assert feat.shape[1] == mask.shape[1],\
            f"mask and feat must have the same length, got {feat.shape[1]} " +\
            f"vs. {mask.shape[1]}"
        mask = self.mask_ext(mask, mask.shape, mask.device)
        # safeguard fp16
        mask = mask.to(dtype=feat_img.dtype)
        out = self.trsfr(feat, mask, output_attentions=True)
        return out['last_hidden_state'], out['attentions']

    def prepro_pretxt(self, task_or_prompt_txt):
        return task_or_prompt_txt

    def get_pretxt(self, mask_txt, task_name=None, prompt=None):
        txt_dim = mask_txt.dim()
        if self.args.enable_task_token:
            assert task_name is not None
            assert task_name in self.task_tok2id 
            task_id = self.task_tok2id[task_name]
            task_feat_txt = self.emb_task[task_id, :].unsqueeze(0)
            task_mask = T.ones(
                1, device=mask_txt.device, dtype=mask_txt.dtype)
            task_txt = T.zeros(
                1, device=mask_txt.device, dtype=mask_txt.dtype)
            if txt_dim > 1:
                _B, _ = mask_txt.shape
                task_txt = task_txt.unsqueeze(0).expand(_B, -1)
                task_mask = task_mask.unsqueeze(0).expand(_B, -1)
                task_feat_txt = task_feat_txt.unsqueeze(0).expand(_B, -1, -1)
            task_txt = self.prepro_pretxt(task_txt)
            return task_txt, task_mask, task_feat_txt
        elif prompt is not None and self.args.enable_prompt:
            prompt_txt, prompt_mask = prompt
            prompt_dim = prompt_txt.dim()
            if prompt_txt.dim() == 1:
                prompt_feat_txt = self.enc_txt(prompt_txt.unsqueeze(0))
            else:
                prompt_feat_txt = self.enc_txt(prompt_txt)
            if txt_dim > 1 and prompt_dim == 1:
                _B, _ = mask_txt.shape
                prompt_txt = prompt_txt.unsqueeze(
                    0).expand(_B, -1)
                prompt_mask = prompt_mask.unsqueeze(
                    0).expand(_B, -1)
                prompt_feat_txt = prompt_feat_txt.expand(_B, -1, -1)
            elif txt_dim == 1 and prompt_dim == 1:
                prompt_feat_txt = prompt_feat_txt[0]
            elif txt_dim > 1 and prompt_dim > 1 and txt_dim == prompt_dim:
                assert mask_txt.shape[0] == prompt_txt.shape[0]
            else:
                raise ValueError(
                    f"txt dim: {txt_dim}, prompt_txt dim {prompt_dim}")
            prompt_txt = self.prepro_pretxt(prompt_txt)
            return prompt_txt, prompt_mask, prompt_feat_txt
        else:
            return None, None, None

    def prepro_txt_inputs(self, txt, mask_txt,
                          feat_txt, task_name=None, prompt=None):
        # if self.args.enable_task_token:
        #     assert task_name in self.task_tok2id or task_name is None
        #     return self.add_task_token_to_txt(
        #         txt, mask_txt, feat_txt, task_name)
        # is self.args.enable_prompt:
        #     return self.add_prompt_to_txt(txt, mask_txt, feat_txt, prompt)
        pretxt_txt, pretxt_mask, pretxt_feat = self.get_pretxt(
            mask_txt, task_name, prompt)
        if pretxt_txt is not None:
            mask_txt = T.cat([pretxt_mask, mask_txt], dim=-1)
            txt = T.cat([pretxt_txt, txt], dim=-1)
            feat_txt = T.cat([pretxt_feat, feat_txt], dim=-2)
        return txt, mask_txt, feat_txt

    def add_task_token_to_txt(self, txt, mask_txt, feat_txt, task_name=None):
        if task_name is not None:
            task_id = self.task_tok2id[task_name]
            task_feat_txt = self.emb_task[task_id, :].unsqueeze(0)
            task_mask = T.ones(
                1, device=mask_txt.device, dtype=mask_txt.dtype)
            task_txt = T.zeros(
                1, device=mask_txt.device, dtype=mask_txt.dtype)
            if txt.dim() > 1:
                _B, _ = txt.shape
                task_txt = task_txt.unsqueeze(0).expand(_B, -1)
                task_mask = task_mask.unsqueeze(0).expand(_B, -1)
                task_feat_txt = task_feat_txt.unsqueeze(0).expand(_B, -1, -1)
            task_txt = self.prepro_pretxt(task_txt)
            mask_txt = T.cat([task_mask, mask_txt], dim=-1)
            txt = T.cat([task_txt, txt], dim=-1)
            feat_txt = T.cat([task_feat_txt, feat_txt], dim=-2)
        return txt, mask_txt, feat_txt

    def add_prompt_to_txt(self, txt, mask_txt, feat_txt, prompt=None):
        if prompt is not None:
            prompt_txt, prompt_mask = prompt
            if prompt_txt.dim() == 1:
                prompt_feat_txt = self.enc_txt(prompt_txt.unsqueeze(0))
            else:
                prompt_feat_txt = self.enc_txt(prompt_txt)
            if txt.dim() > 1 and prompt_txt.dim() == 1:
                _B, _ = txt.shape
                prompt_txt = prompt_txt.unsqueeze(
                    0).expand(_B, -1)
                prompt_mask = prompt_mask.unsqueeze(
                    0).expand(_B, -1)
                prompt_feat_txt = prompt_feat_txt.expand(_B, -1, -1)
            elif txt.dim() == 1 and prompt_txt.dim() == 1:
                prompt_feat_txt = prompt_feat_txt[0]
            else:
                raise ValueError(
                    f"txt dim: {txt.dim()}, prompt_txt dim {prompt_txt.dim()}")
            prompt_txt = self.prepro_pretxt(prompt_txt)
            mask_txt = T.cat([prompt_mask, mask_txt], dim=-1)
            txt = T.cat([prompt_txt, txt], dim=-1)
            feat_txt = T.cat([prompt_feat_txt, feat_txt], dim=-2)
        return txt, mask_txt, feat_txt

    def load_ckpt(self, ckpt):
        if ckpt == '':
            print('===== Finished Init LAVENDER  =====')
            return
        elif not os.path.exists(ckpt):
            print(f'Try to load pre-trained weights from {ckpt}, '
                  f'but file does not exists...')
            return
        print(f'Loading pre-trained weights from {ckpt}')
        loaded_state_dict = T.load(ckpt, map_location='cpu')
        # missing, unexpected = self.load_state_dict(
        #   loaded_state_dict, strict=False)
        filename, _ = os.path.splitext(ckpt.split("/")[-1])
        if "SwinBERT" in filename:
            self.load_SwinBERT_weight(loaded_state_dict)
        else:
            self.__load_ckpt__(loaded_state_dict)

    def __load_ckpt__(self, loaded_state_dict):
        model_keys = set([k for k in list(self.state_dict().keys())])
        load_keys = set(loaded_state_dict.keys())

        toload = {}
        mismatched_shape_keys = []
        for k in model_keys:
            if k in load_keys:
                if self.state_dict()[k].shape != loaded_state_dict[k].shape:
                    mismatched_shape_keys.append(
                        (k, loaded_state_dict[k].shape,
                         self.state_dict()[k].shape))
                else:
                    toload[k] = loaded_state_dict[k]

        print("You can ignore the keys with `position_ids` or from task heads")
        strct_loading = True
        unexpected = load_keys.difference(model_keys)
        if len(unexpected):
            strct_loading = False
            print("=========================Unexpected==================================")
            print(f"\tIn total {len(unexpected)}, {sorted(unexpected)}")

        missing = model_keys.difference(load_keys)
        if len(missing):
            strct_loading = False
            print("===========================Missing===================================")
            print(f"\tIn total {len(missing)}, {sorted(missing)}")

        if len(mismatched_shape_keys):
            strct_loading = False
            print("======================Shape Mismatched===============================")
            print(f"\tIn total {len(mismatched_shape_keys)}, "
                  f"{sorted(mismatched_shape_keys)}")

        self.load_state_dict(toload, strict=strct_loading)
        loaded_max_size_frame = getattr(
            loaded_state_dict, "enc_img.max_size_frame", 6)
        loaded_max_size_patch = getattr(
            loaded_state_dict, "enc_img.max_size_patch", 14)

        if loaded_max_size_frame < self.enc_img.max_size_frame:
            self.enc_img.emb_len.data[:, :loaded_max_size_frame].copy_(
                loaded_state_dict["enc_img.emb_len"])
        elif loaded_max_size_frame > self.enc_img.max_size_frame:
            self.enc_img.emb_len.data.copy_(
                loaded_state_dict["enc_img.emb_len"][
                    :, :self.enc_img.max_size_frame])
        else:
            print("enc_img.enc_len shape matched")

        if loaded_max_size_patch < self.enc_img.max_size_patch:
            self.enc_img.emb_pos.data[:, :, :loaded_max_size_patch].copy_(
                loaded_state_dict["enc_img.emb_pos"])
        elif loaded_max_size_patch > self.enc_img.max_size_patch:
            self.enc_img.emb_pos.data.copy_(
                loaded_state_dict["enc_img.emb_pos"][
                    :, :, :self.enc_img.max_size_patch])
        else:
            print("enc_img.emb_pos shape matched")

    def load_SwinBERT_weight(self, loaded_state_dict):
        print(f'Special loading with SwinBERT pre-trained weights')
        # model_keys = set([k for k in list(self.state_dict().keys())])
        load_keys = set(loaded_state_dict.keys())

        toload = {}
        deleted = set()
        for key in load_keys:
            if "swin.backbone" in key:
                new_key = key.replace("swin.backbone", "enc_img.swin")
                toload[new_key] = loaded_state_dict[key]
            elif "trans_encoder.bert.encoder" in key:
                new_key = key.replace("trans_encoder.bert.encoder", "trsfr")
                toload[new_key] = loaded_state_dict[key]
            elif "trans_encoder.bert.embeddings" in key:
                new_key = key.replace(
                    "trans_encoder.bert.embeddings",
                    "enc_txt.emb_txt")
                toload[new_key] = loaded_state_dict[key]
            elif key.startswith("fc."):
                new_key = key.replace(
                    "fc.",
                    "enc_img.fc.")
                toload[new_key] = loaded_state_dict[key]
            elif "trans_encoder.bert.img_embedding" in key:
                new_key = key.replace(
                    "trans_encoder.bert.img_embedding",
                    "enc_img.img_embedding")
                toload[new_key] = loaded_state_dict[key]
                toload[new_key] = loaded_state_dict[key]
            elif key.startswith("trans_encoder.cls."):
                new_key = key.replace(
                    "trans_encoder.cls.",
                    "fc_mtm.")
                toload[new_key] = loaded_state_dict[key]
            else:
                deleted.add(key)
        deleted = list(deleted)
        # fake a zero bias for fc_mtm
        toload["fc_mtm.predictions.decoder.bias"] = toload["fc_mtm.predictions.bias"]
        print("======================Keys removed from SwinBERT pretrained ===============================")
        print(f"\tIn total {len(deleted)}, {sorted(deleted)}")
        self.__load_ckpt__(toload)
