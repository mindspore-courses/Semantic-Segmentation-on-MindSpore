"""Context Encoding for Semantic Segmentation"""
from mindspore import nn
from mindspore import ops
from mindspore.common import Parameter
from mindspore.ops import operations as P
import mindspore.ops.functional as F
from mindspore import Tensor
import numpy as np

from core.models.segbase import SegBaseModel
from core.models.fcn import _FCNHead

__all__ = ['EncNet', 'EncModule', 'get_encnet', 'get_encnet_resnet50_ade',
           'get_encnet_resnet101_ade', 'get_encnet_resnet152_ade']


class EncNet(SegBaseModel):
    def __init__(self, nclass, backbone='resnet50', aux=True, se_loss=True, lateral=False,
                 pretrained_base=True, **kwargs):
        super(EncNet, self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)
        self.head = _EncHead(2048, nclass, se_loss=se_loss, lateral=lateral, **kwargs)
        if aux:
            self.auxlayer = _FCNHead(1024, nclass, **kwargs)

        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])

    def construct(self, x):
        size = x.shape[2:]
        features = self.base_forward(x)

        x = list(self.head(*features))
        x[0] = ops.interpolate(x[0], size, mode='bilinear', align_corners=True)
        if self.aux:
            auxout = self.auxlayer(features[2])
            auxout = ops.interpolate(auxout, size, mode='bilinear', align_corners=True)
            x.append(auxout)
        return tuple(x)


class _EncHead(nn.Cell):
    def __init__(self, in_channels, nclass, se_loss=True, lateral=True,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_EncHead, self).__init__()
        self.lateral = lateral
        self.conv5 = nn.SequentialCell(
            nn.Conv2d(in_channels, 512, 3, pad_mode='pad', padding=1, has_bias=False),
            norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU()
        )
        if lateral:
            self.connect = nn.CellList([
                nn.SequentialCell(
                    nn.Conv2d(512, 512, pad_mode='pad', padding=1, has_bias=False),
                    norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
                    nn.ReLU()),
                nn.SequentialCell(
                    nn.Conv2d(1024, 512, 1, has_bias=False),
                    norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
                    nn.ReLU()),
            ])
            self.fusion = nn.SequentialCell(
                nn.Conv2d(3 * 512, 512, 3, pad_mode='pad', padding=1, has_bias=False),
                norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
                nn.ReLU()
            )
        self.encmodule = EncModule(512, nclass, ncodes=32, se_loss=se_loss,
                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
        self.conv6 = nn.SequentialCell(
            nn.Dropout(p=0.1),
            nn.Conv2d(512, nclass, 1)
        )

    def construct(self, *inputs):
        feat = self.conv5(inputs[-1])
        if self.lateral:
            c2 = self.connect[0](inputs[1])
            c3 = self.connect[1](inputs[2])
            feat = self.fusion(ops.cat([feat, c2, c3], 1))
        outs = list(self.encmodule(feat))
        outs[0] = self.conv6(outs[0])
        return tuple(outs)


class Mean(nn.Cell):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim
        self.reduce_mean = P.ReduceMean(keep_dims=self.keep_dim)

    def construct(self, input):
        return self.reduce_mean(input, self.dim)


class EncModule(nn.Cell):
    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(EncModule, self).__init__()
        self.se_loss = se_loss
        self.encoding = nn.SequentialCell([
            nn.Conv2d(in_channels, in_channels, kernel_size=1, has_bias=False),
            norm_layer(in_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(),
            Encoding(D=in_channels, K=ncodes),
            nn.BatchNorm1d(ncodes),
            nn.ReLU(),
            Mean(dim=1)
        ])

        self.fc = nn.SequentialCell([
            nn.Dense(in_channels, in_channels),
            nn.Sigmoid()
        ])

        if self.se_loss:
            self.selayer = nn.Dense(in_channels, nclass)

    def construct(self, x):
        en = self.encoding(x)
        b, c, _, _ = P.Shape()(x)
        gamma = self.fc(en)
        y = P.Reshape()(gamma, (b, c, 1, 1))
        outputs = [F.relu(x + x * y)]
        if self.se_loss:
            outputs.append(self.selayer(en))
        return tuple(outputs)


class Encoding(nn.Cell):
    def __init__(self, D, K):
        super(Encoding, self).__init__()
        self.D, self.K = D, K
        self.codewords = Parameter(Tensor(np.random.uniform(-1, 1, (K, D)).astype(np.float32)), requires_grad=True)
        self.scale = Parameter(Tensor(np.random.uniform(-1, 0, K).astype(np.float32)), requires_grad=True)
        self.reset_params()

    def reset_params(self):
        std1 = 1. / ((self.K * self.D) ** (1 / 2))
        self.codewords.set_data(Tensor(np.random.uniform(-std1, std1, self.codewords.shape).astype(np.float32)))
        self.scale.set_data(Tensor(np.random.uniform(-1, 0, self.scale.shape).astype(np.float32)))

    def construct(self, X):
        assert (P.Shape()(X)[1] == self.D)
        B, D = P.Shape()(X)[0], self.D
        expand_dims = P.ExpandDims()
        reshape = P.Reshape()
        sum_op = P.ReduceSum(keep_dims=False)

        if len(X.shape) == 3:
            X = P.Transpose()(X, (0, 2, 1))
        elif len(X.shape) == 4:
            X = reshape(X, (B, D, -1))
            X = P.Transpose()(X, (0, 2, 1))
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')

        S = expand_dims(self.scale, 0)
        S = expand_dims(S, 0)
        S = expand_dims(S, -1)

        C = expand_dims(self.codewords, 0)
        C = expand_dims(C, 0)

        X_expanded = expand_dims(X, 2)
        X_expanded = P.BroadcastTo((B, X.shape[1], self.K, D))(X_expanded)

        SL = S * (X_expanded - C)
        SL = SL ** 2
        SL = sum_op(SL, -1)

        A = P.Softmax(axis=2)(SL)

        A_expanded = expand_dims(A, -1)
        X_expanded = expand_dims(X, 2)
        X_expanded = P.BroadcastTo((B, X.shape[1], self.K, D))(X_expanded)

        E = A_expanded * (X_expanded - C)
        E = sum_op(E, 1)

        return E


def get_encnet(dataset='pascal_voc', backbone='resnet50', pretrained=False, root='~/.torch/models',
               pretrained_base=False, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from core.data.dataloader import datasets
    model = EncNet(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)
    return model


def get_encnet_resnet50_ade(**kwargs):
    return get_encnet('ade20k', 'resnet50', **kwargs)


def get_encnet_resnet101_ade(**kwargs):
    return get_encnet('ade20k', 'resnet101', **kwargs)


def get_encnet_resnet152_ade(**kwargs):
    return get_encnet('ade20k', 'resnet152', **kwargs)


if __name__ == '__main__':
    img = ops.randn(2, 3, 224, 224)
    model = get_encnet_resnet50_ade()
    outputs = model(img)
    print(model)
