"""Fully Convolutional Network with Stride of 8"""
from __future__ import division

from mindspore import ops
import mindspore.nn as nn

from core.models.segbase import SegBaseModel

__all__ = ['FCN', 'get_fcn', 'get_fcn_resnet50_voc',
           'get_fcn_resnet101_voc', 'get_fcn_resnet152_voc']


class FCN(SegBaseModel):
    def __init__(self, nclass, backbone='resnet50', aux=True, pretrained_base=True, **kwargs):
        super(FCN, self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)
        self.head = _FCNHead(2048, nclass, **kwargs)
        if aux:
            self.auxlayer = _FCNHead(1024, nclass, **kwargs)

        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])

    def construct(self, x):
        size = x.shape[2:]
        _, _, c3, c4 = self.base_forward(x)

        outputs = []
        x = self.head(c4)
        x = ops.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = ops.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class _FCNHead(nn.Cell):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.SequentialCell(
            nn.Conv2d(in_channels, inter_channels, 3, pad_mode='pad', padding=1, has_bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def construct(self, x):
        return self.block(x)


def get_fcn(dataset='pascal_voc', backbone='resnet50', pretrained=False, root='~/.torch/models',
            pretrained_base=False, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from core.data.dataloader import datasets
    model = FCN(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)
    return model


def get_fcn_resnet50_voc(**kwargs):
    return get_fcn('pascal_voc', 'resnet50', **kwargs)


def get_fcn_resnet101_voc(**kwargs):
    return get_fcn('pascal_voc', 'resnet101', **kwargs)


def get_fcn_resnet152_voc(**kwargs):
    return get_fcn('pascal_voc', 'resnet152', **kwargs)

if __name__ == '__main__':
    model = get_fcn_resnet50_voc()
    print(model)
