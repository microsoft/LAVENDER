import transformers
import torch as T
import torchvision as TV
# from .video_transform import Normalize


SWIN_CFG = {
    "base-in22k": "microsoft/swin-base-patch4-window7-224-in22k",
    "base": "microsoft/swin-base-patch4-window7-224",
    "tiny": "microsoft/swin-tiny-patch4-window7-224",
    "large-in22k": "microsoft/swin-large-patch4-window7-224-in22k",
    "large": "microsoft/swin-large-patch4-window7-224",
    "small": "microsoft/swin-small-patch4-window7-224"}


def get_swin_model(args):
    model_name = args.vis_backbone_size
    assert args.size_img == 224
    if args.vis_backbone_size in ["base", "large"] and args.imagenet == 22:
        model_name = f"{args.vis_backbone_size}-in22k"
    else:
        model_name = args.vis_backbone_size

    placeholder = transformers.SwinModel.from_pretrained(SWIN_CFG[model_name])

    if args.vis_backbone_init == "random":
        args.vis_backbone_pretrained_weight = None
        cfg = placeholder.config
        swin = transformers.SwinModel(cfg)
        del placeholder
    else:
        swin = placeholder
        args.vis_backbone_pretrained_weight = SWIN_CFG[model_name]
    return swin


class EncImgSwinMean(T.nn.Module):
    def __init__(self, args, hidden_size):
        super().__init__()
        assert args.temporal_fusion == "mean"
        self.swin = get_swin_model(args)
        self.img_feature_dim = hidden_size
        self.latent_feat_size = self.swin.num_features

        self.swin2bert = T.nn.Conv1d(
            self.latent_feat_size, self.img_feature_dim, 1)

        self.emb_cls = T.nn.Parameter(
            0.02*T.randn(1, 1, 1, self.img_feature_dim))
        self.emb_pos = T.nn.Parameter(
            0.02*T.randn(1, 1, 1+14**2, self.img_feature_dim))
        self.emb_len = T.nn.Parameter(
            0.02*T.randn(1, 6, 1, self.img_feature_dim))
        self.norm = T.nn.LayerNorm(self.img_feature_dim)
        # if args.img_transform == ["vid_rand_crop"]:
        #     self.transform_normalize = None
        # elif args.imagenet_norm:
        #     self.transform_normalize = TV.transforms.Normalize(
        #         [0.485, 0.456, 0.406],
        #         [0.229, 0.224, 0.225])
        # else:
        #     self.transform_normalize = TV.transforms.Normalize(
        #         [0.5, 0.5, 0.5],
        #         [0.5, 0.5, 0.5])
        self.transform_normalize = None

    def forward(self, img, odr=None):
        _B, _T, _C, _H, _W = img.shape
        _h, _w = _H//32, _W//32

        if self.transform_normalize is not None:
            img = self.transform_normalize(img)

        f_img = self.swin(
            img.flatten(0, 1), output_hidden_states=True)['hidden_states'][-1]
        f_img = self.swin2bert(f_img.permute(0, 2, 1))
        f_img = f_img.view([_B, _T, -1, _h*_w]).permute(0, 1, 3, 2)

        f_img = T.mean(f_img, dim=1, keepdim=True)
        _T = 1  # MEAN

        f_img = T.cat([self.emb_cls.expand([_B, _T, -1, -1]), f_img], dim=2)
        f_img += self.emb_pos.expand([_B, _T, -1, -1])[:, :, :1+_h*_w, :]
        f_img += self.emb_len.expand([_B, -1, 1+_h*_w, -1])[:, :_T, :, :]
        f_img = self.norm(f_img).view([_B, _T*(1+_h*_w), -1])

        m_img = T.ones(1+_h*_w).long().cuda().unsqueeze(0).unsqueeze(0)
        m_img = m_img.expand([_B, _T, -1]).contiguous().view(
            [_B, _T*(1+_h*_w)])

        return f_img, m_img


class EncImgSwinConcat(T.nn.Module):
    def __init__(self, args, hidden_size):
        super().__init__()
        self.imagenet_norm = args.imagenet_norm
        self.swin = get_swin_model(args)
        self.img_feature_dim = hidden_size
        self.latent_feat_size = self.swin.num_features

        self.swin2bert = T.nn.Conv1d(
            self.latent_feat_size, self.img_feature_dim, 1)

        self.emb_cls = T.nn.Parameter(
            0.02*T.randn(1, 1, 1, self.img_feature_dim))
        self.emb_pos = T.nn.Parameter(
            0.02*T.randn(1, 1, 1+14**2, self.img_feature_dim))
        self.emb_len = T.nn.Parameter(
            0.02*T.randn(1, 6, 1, self.img_feature_dim))
        self.emb_odr = T.nn.Parameter(
            0.02*T.randn(1, 1, 1, self.img_feature_dim))
        self.norm = T.nn.LayerNorm(self.img_feature_dim)
        # if args.img_transform == ["vid_rand_crop"]:
        #     self.transform_normalize = None
        # elif self.imagenet_norm:
        #     self.transform_normalize = TV.transforms.Normalize(
        #         [0.485, 0.456, 0.406],
        #         [0.229, 0.224, 0.225])
        # else:
        #     self.transform_normalize = TV.transforms.Normalize(
        #         [0.5, 0.5, 0.5],
        #         [0.5, 0.5, 0.5])
        self.transform_normalize = None

    def forward(self, img, odr=None, vt_mask=None):
        _B, _T, _C, _H, _W = img.shape
        _h, _w = _H//32, _W//32

        if self.transform_normalize is not None:
            img = self.transform_normalize(img)

        f_img = self.swin(
            img.flatten(0, 1), output_hidden_states=True)['hidden_states'][-1]
        f_img = self.swin2bert(f_img.permute(0, 2, 1))
        f_img = f_img.view(
            [_B, _T, self.img_feature_dim, _h*_w]).permute(0, 1, 3, 2)

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
        m_img = m_img.expand([_B, _T, -1]).contiguous().view(
            [_B, _T*(1+_h*_w)])

        return f_img, m_img
