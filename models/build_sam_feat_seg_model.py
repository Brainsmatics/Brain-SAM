import torch

from functools import partial

from .SamFeatSeg import SamFeatSeg, SegDecoderCNN, UPromptCNN
# from .sam_decoder import MaskDecoder
from segment_anything.modeling import ImageEncoderViT, PromptEncoder,TwoWayTransformer


def _build_feat_seg_model(
    img_size,
    iter_2stage,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    num_classes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam_seg = SamFeatSeg(
        iter_2stage=iter_2stage,
        img_size=img_size,
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        promptcnn_first=UPromptCNN(),
        seg_decoder_first=SegDecoderCNN(num_classes=num_classes, num_depth=4, p_channel=3, promptemd_channel=0, first_p=True),

        prompt_encoder_end=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        seg_decoder_end=SegDecoderCNN(num_classes=num_classes, num_depth=4, p_channel=0, promptemd_channel=256, first_p=False),
    )

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)

        loaded_keys = []
        for k in state_dict.keys():
            if k in sam_seg.state_dict().keys():
                loaded_keys.append(k)
        sam_seg.load_state_dict(state_dict, strict=False)
        # print("loaded keys:", loaded_keys)
        print('load keys over!')
    return sam_seg


def build_sam_vit_h_seg_cnn(num_classes=2, checkpoint=None, img_size=320, iter_2stage=1):
    return _build_feat_seg_model(
        img_size=img_size,
        iter_2stage=iter_2stage,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        num_classes=num_classes,
        checkpoint=checkpoint,
    )


build_sam_seg = build_sam_vit_h_seg_cnn


def build_sam_vit_l_seg_cnn(num_classes=2, checkpoint=None, img_size=320, iter_2stage=1):
    return _build_feat_seg_model(
        img_size=img_size,
        iter_2stage=iter_2stage,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        num_classes=num_classes,
        checkpoint=checkpoint,
    )


def build_sam_vit_b_seg_cnn(num_classes=2, checkpoint=None, img_size=320, iter_2stage=1):
    return _build_feat_seg_model(
        img_size=img_size,
        iter_2stage=iter_2stage,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        num_classes=num_classes,
        checkpoint=checkpoint,
    )


sam_feat_seg_model_registry = {
    "default": build_sam_seg,
    "vit_h": build_sam_seg,
    "vit_l": build_sam_vit_l_seg_cnn,
    "vit_b": build_sam_vit_b_seg_cnn,
}

