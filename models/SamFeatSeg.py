from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Type
# from segment_anything.modeling import ImageEncoderViT
# from segment_anything.modeling.common import LayerNorm2d
from models.unet_parts import *

class SamFeatSeg(nn.Module):
    # 在此定义模型整体相互调用
    def __init__(
        self,
        image_encoder,
        seg_decoder_first,
        promptcnn_first,
        prompt_encoder_end,
        seg_decoder_end,
        img_size,
        iter_2stage,
    ):
        super().__init__()
        self.iter_2stage = iter_2stage
        self.img_size = img_size
        self.image_encoder = image_encoder
        self.promptcnn_first = promptcnn_first
        self.seg_decoder_first = seg_decoder_first

        self.conv = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
        self.prompt_encoder_end = prompt_encoder_end
        self.seg_decoder_end = seg_decoder_end

    def forward(self, x):
        original_size = x.shape[-1]
        x = F.interpolate(
            x,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        image_embedding = self.image_encoder(x) #[B, 256, 64, 64]
        prompt_embedding = self.promptcnn_first(x) #[B, 3, 256, 256]prompt encoder
        out1 = self.seg_decoder_first(image_embedding, prompt_embedding) #fine decoder
        if out1.shape[-1] != original_size:
            out1 = F.interpolate(
                out1,
                (original_size, original_size),
                mode="bilinear",
                align_corners=False,
            )
        out1_d = out1

        for Y in range(self.iter_2stage): #最小设为1
            p2_in = F.interpolate(
                out1_d,
                (256, 256),
                mode="bilinear",
                align_corners=False,
            )
            p2_in = self.conv(p2_in)
            sparse_embeddings2, prompt_embedding2 = self.prompt_encoder_end(
                points=None,
                boxes=None,
                masks=p2_in,
            ) # [B, 256, 64, 64]
            out1_d = self.seg_decoder_end(image_embedding, prompt_embedding2)
            out1_d = F.interpolate(
                out1_d,
                (original_size, original_size),
                mode="bilinear",
                align_corners=False,
            )

        out2 = out1_d

        return out1, out2, prompt_embedding

    def get_embedding(self, x):
        original_size = x.shape[-1]
        x = F.interpolate(
            x,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        image_embedding = self.image_encoder(x)
        out = nn.functional.adaptive_avg_pool2d(image_embedding, 1).squeeze()
        return out


class SegDecoderCNN(nn.Module):
    def __init__(self,
                 num_classes=2,
                 x_embed_dim=256,
                 num_depth=4,
                 top_channel=64,
                 p_channel=3,
                 promptemd_channel=256,
                 first_p=True, #第一次提示端
                 ):
        super().__init__()
        self.first_p = first_p
        self.input_block = nn.Sequential(
            nn.Conv2d(x_embed_dim+promptemd_channel, top_channel*2**num_depth, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(top_channel * 2 ** num_depth, top_channel * 2 ** num_depth, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList()
        for i in range(num_depth):
            if num_depth > 2 > i:
                block = nn.Sequential(
                    nn.Conv2d(top_channel * 2 ** (num_depth - i), top_channel * 2 ** (num_depth - i), 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(top_channel * 2 ** (num_depth - i), top_channel * 2 ** (num_depth - i - 1), 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(top_channel * 2 ** (num_depth - i),  top_channel * 2 ** (num_depth - i), 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(top_channel * 2 ** (num_depth - i),  top_channel * 2 ** (num_depth - i), 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(top_channel * 2 ** (num_depth - i), top_channel * 2 ** (num_depth - i - 1), 2, stride=2),
                )
            self.blocks.append(block)

        self.final = nn.Sequential(
            nn.Conv2d(top_channel+p_channel, int(top_channel), 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(top_channel, num_classes, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(num_classes, num_classes, 1),
        )

    def forward(self, x, p):
        if self.first_p:
            x = self.input_block(x)
            for blk in self.blocks:
                x = blk(x)
            y = torch.cat([x, p], 1)
            y = self.final(y)
        else:
            x = torch.cat([x, p], 1)
            x = self.input_block(x)
            for blk in self.blocks:
                x = blk(x)
            y = self.final(x)

        return y

class UPromptCNN(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=False):
        super(UPromptCNN, self).__init__()

        self.bilinear = bilinear
        self.downsample = nn.Sequential(
            nn.MaxPool2d(2),
            nn.MaxPool2d(2)
        )
        self.inc = (DoubleConv(n_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        factor = 2 if bilinear else 1
        self.down3 = (Down(128, 256 // factor))
        self.up1 = (Up(256, 128 // factor, bilinear))
        self.up2 = (Up(128, 64 // factor, bilinear))
        self.up3 = (Up(64, 32, bilinear))
        self.outc = (OutConv(32, n_classes))

    def forward(self, x):
        x = self.downsample(x)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        PromptEmbedding = self.outc(x)

        return PromptEmbedding #n,3,256,256