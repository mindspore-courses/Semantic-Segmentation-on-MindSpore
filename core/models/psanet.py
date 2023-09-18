"""Point-wise Spatial Attention Network"""
import mindspore
from mindspore import ops
import mindspore.nn as nn

from core.nn import _ConvBNReLU
from core.models.segbase import SegBaseModel
from core.models.fcn import _FCNHead

__all__ = ['PSANet', 'get_psanet', 'get_psanet_resnet50_voc', 'get_psanet_resnet101_voc',
           'get_psanet_resnet152_voc', 'get_psanet_resnet50_citys', 'get_psanet_resnet101_citys',
           'get_psanet_resnet152_citys']


class PSANet(SegBaseModel):
    r"""PSANet

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:
        Hengshuang Zhao, et al. "PSANet: Point-wise Spatial Attention Network for Scene Parsing."
        ECCV-2018.
    """

    def __init__(self, nclass, backbone='resnet', aux=False, pretrained_base=True, **kwargs):
        super(PSANet, self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)
        self.head = _PSAHead(nclass, **kwargs)
        if aux:
            self.auxlayer = _FCNHead(1024, nclass, **kwargs)

        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])

    def construct(self, x):
        size = x.shape[2:]
        _, _, c3, c4 = self.base_forward(x)
        outputs = list()
        x = self.head(c4)
        x = ops.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = ops.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class _PSAHead(nn.Cell):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_PSAHead, self).__init__()
        # psa_out_channels = crop_size // 8 ** 2
        self.psa = _PointwiseSpatialAttention(2048, 3600, norm_layer)

        self.conv_post = _ConvBNReLU(1024, 2048, 1, norm_layer=norm_layer)
        self.project = nn.SequentialCell(
            _ConvBNReLU(4096, 512, 3, pad_mode='pad', padding=1, norm_layer=norm_layer),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(512, nclass, 1))

    def construct(self, x):
        global_feature = self.psa(x)
        out = self.conv_post(global_feature)
        out = ops.cat([x, out], axis=1)
        out = self.project(out)

        return out


class _PointwiseSpatialAttention(nn.Cell):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_PointwiseSpatialAttention, self).__init__()
        reduced_channels = 512
        self.collect_attention = _AttentionGeneration(in_channels, reduced_channels, out_channels, norm_layer)
        self.distribute_attention = _AttentionGeneration(in_channels, reduced_channels, out_channels, norm_layer)

    def construct(self, x):
        collect_fm = self.collect_attention(x)
        distribute_fm = self.distribute_attention(x)
        psa_fm = ops.cat([collect_fm, distribute_fm], axis=1)
        return psa_fm


class _AttentionGeneration(nn.Cell):
    def __init__(self, in_channels, reduced_channels, out_channels, norm_layer, **kwargs):
        super(_AttentionGeneration, self).__init__()
        self.conv_reduce = _ConvBNReLU(in_channels, reduced_channels, 1, norm_layer=norm_layer)
        self.attention = nn.SequentialCell(
            _ConvBNReLU(reduced_channels, reduced_channels, 1, norm_layer=norm_layer),
            nn.Conv2d(reduced_channels, out_channels, 1, has_bias=False))

        self.reduced_channels = reduced_channels

    def construct(self, x):
        reduce_x = self.conv_reduce(x)
        attention = self.attention(reduce_x)
        n, c, h, w = attention.shape
        attention = attention.view(n, c, -1)
        reduce_x = reduce_x.view(n, self.reduced_channels, -1)
        fm = ops.bmm(reduce_x, ops.softmax(attention, axis=1))
        fm = fm.view(n, self.reduced_channels, h, w)

        return fm


def get_psanet(dataset='pascal_voc', backbone='resnet50', pretrained=False, root='~/.torch/models',
               pretrained_base=True, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from core.data.dataloader import datasets
    model = PSANet(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)

    return model


def get_psanet_resnet50_voc(**kwargs):
    return get_psanet('pascal_voc', 'resnet50', **kwargs)


def get_psanet_resnet101_voc(**kwargs):
    return get_psanet('pascal_voc', 'resnet101', **kwargs)


def get_psanet_resnet152_voc(**kwargs):
    return get_psanet('pascal_voc', 'resnet152', **kwargs)


def get_psanet_resnet50_citys(**kwargs):
    return get_psanet('citys', 'resnet50', **kwargs)


def get_psanet_resnet101_citys(**kwargs):
    return get_psanet('citys', 'resnet101', **kwargs)


def get_psanet_resnet152_citys(**kwargs):
    return get_psanet('citys', 'resnet152', **kwargs)


if __name__ == '__main__':
    model = get_psanet_resnet50_voc()
    img = ops.randn(1, 3, 480, 480)
    output = model(img)
    print(model)
