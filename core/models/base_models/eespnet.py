import math
from mindspore import Tensor, nn, ops

from core.nn import _ConvBNPReLU, _ConvBN, _BNPReLU

__all__ = ['EESP', 'EESPNet', 'eespnet']


class EESP(nn.Cell):

    def __init__(self, in_channels, out_channels, stride=1, k=4, r_lim=7, down_method='esp', norm_layer=nn.BatchNorm2d):
        super(EESP, self).__init__()
        self.stride = stride
        n = int(out_channels / k)
        n1 = out_channels - (k - 1) * n
        assert down_method in ['avg', 'esp'], 'One of these is suppported (avg or esp)'
        assert n == n1, "n(={}) and n1(={}) should be equal for Depth-wise Convolution ".format(n, n1)
        self.proj_1x1 = _ConvBNPReLU(in_channels, n, 1, stride=1, group=k, norm_layer=norm_layer)

        map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}
        self.k_sizes = list()
        for i in range(k):
            ksize = int(3 + 2 * i)
            ksize = ksize if ksize <= r_lim else 3
            self.k_sizes.append(ksize)
        self.k_sizes.sort()
        self.spp_dw = nn.CellList()
        for i in range(k):
            dilation = map_receptive_ksize[self.k_sizes[i]]
            self.spp_dw.append(nn.Conv2d(n, n, 3, stride, pad_mode='pad', padding=dilation, dilation=dilation, group=n, has_bias=False))
        self.conv_1x1_exp = _ConvBN(out_channels, out_channels, 1, 1, groups=k, norm_layer=norm_layer)
        self.br_after_cat = _BNPReLU(out_channels, norm_layer)
        self.module_act = nn.PReLU(out_channels)
        self.downAvg = True if down_method == 'avg' else False

    def construct(self, x):
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            out_k = out_k + output[k - 1]
            output.append(out_k)
        expanded = self.conv_1x1_exp(self.br_after_cat(ops.cat(output, 1)))
        del output
        if self.stride == 2 and self.downAvg:
            return expanded

        if expanded.shape == x.shape:
            expanded = expanded + x

        return self.module_act(expanded)


class DownSampler(nn.Cell):

    def __init__(self, in_channels, out_channels, k=4, r_lim=9, reinf=True, inp_reinf=3, norm_layer=None):
        super(DownSampler, self).__init__()
        channels_diff = out_channels - in_channels
        self.eesp = EESP(in_channels, channels_diff, stride=2, k=k,
                         r_lim=r_lim, down_method='avg', norm_layer=norm_layer)
        self.avg = nn.AvgPool2d(kernel_size=3, pad_mode='pad', padding=1, stride=2)
        if reinf:
            self.inp_reinf = nn.SequentialCell(
                _ConvBNPReLU(inp_reinf, inp_reinf, 3, 1, 1),
                _ConvBN(inp_reinf, out_channels, 1, 1))
        self.act = nn.PReLU(out_channels)

    def construct(self, x, x2=None):
        avg_out = self.avg(x)
        eesp_out = self.eesp(x)
        output = ops.cat([avg_out, eesp_out], 1)
        if x2 is not None:
            w1 = avg_out.shape[2]
            while True:
                x2 = ops.avg_pool2d(x2, kernel_size=3, padding=1, stride=2)
                w2 = x2.shape[2]
                if w2 == w1:
                    break
            output = output + self.inp_reinf(x2)

        return self.act(output)


class EESPNet(nn.Cell):
    def __init__(self, num_classes=1000, scale=1, reinf=True, norm_layer=nn.BatchNorm2d):
        super(EESPNet, self).__init__()
        inp_reinf = 3 if reinf else None
        reps = [0, 3, 7, 3]
        r_lim = [13, 11, 9, 7, 5]
        K = [4] * len(r_lim)

        # set out_channels
        base, levels, base_s = 32, 5, 0
        out_channels = [base] * levels
        for i in range(levels):
            if i == 0:
                base_s = int(base * scale)
                base_s = math.ceil(base_s / K[0]) * K[0]
                out_channels[i] = base if base_s > base else base_s
            else:
                out_channels[i] = base_s * pow(2, i)
        if scale <= 1.5:
            out_channels.append(1024)
        elif scale in [1.5, 2]:
            out_channels.append(1280)
        else:
            raise ValueError("Unknown scale value.")

        self.level1 = _ConvBNPReLU(3, out_channels[0], 3, 2, 1, norm_layer=norm_layer)

        self.level2_0 = DownSampler(out_channels[0], out_channels[1], k=K[0], r_lim=r_lim[0],
                                    reinf=reinf, inp_reinf=inp_reinf, norm_layer=norm_layer)

        self.level3_0 = DownSampler(out_channels[1], out_channels[2], k=K[1], r_lim=r_lim[1],
                                    reinf=reinf, inp_reinf=inp_reinf, norm_layer=norm_layer)
        self.level3 = nn.CellList()
        for i in range(reps[1]):
            self.level3.append(EESP(out_channels[2], out_channels[2], k=K[2], r_lim=r_lim[2],
                                    norm_layer=norm_layer))

        self.level4_0 = DownSampler(out_channels[2], out_channels[3], k=K[2], r_lim=r_lim[2],
                                    reinf=reinf, inp_reinf=inp_reinf, norm_layer=norm_layer)
        self.level4 = nn.CellList()
        for i in range(reps[2]):
            self.level4.append(EESP(out_channels[3], out_channels[3], k=K[3], r_lim=r_lim[3],
                                    norm_layer=norm_layer))

        self.level5_0 = DownSampler(out_channels[3], out_channels[4], k=K[3], r_lim=r_lim[3],
                                    reinf=reinf, inp_reinf=inp_reinf, norm_layer=norm_layer)
        self.level5 = nn.CellList()
        for i in range(reps[2]):
            self.level5.append(EESP(out_channels[4], out_channels[4], k=K[4], r_lim=r_lim[4],
                                    norm_layer=norm_layer))

        self.level5.append(_ConvBNPReLU(out_channels[4], out_channels[4], 3, 1, 1,
                                        groups=out_channels[4], norm_layer=norm_layer))
        self.level5.append(_ConvBNPReLU(out_channels[4], out_channels[5], 1, 1, 0,
                                        groups=K[4], norm_layer=norm_layer))

        self.fc = nn.Dense(out_channels[5], num_classes)


    def construct(self, x, seg=True):
        out_l1 = self.level1(x)

        out_l2 = self.level2_0(out_l1, x)

        out_l3_0 = self.level3_0(out_l2, x)
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)

        out_l4_0 = self.level4_0(out_l3, x)
        for i, layer in enumerate(self.level4):
            if i == 0:
                out_l4 = layer(out_l4_0)
            else:
                out_l4 = layer(out_l4)

        if not seg:
            out_l5_0 = self.level5_0(out_l4)  # down-sampled
            for i, layer in enumerate(self.level5):
                if i == 0:
                    out_l5 = layer(out_l5_0)
                else:
                    out_l5 = layer(out_l5)

            output_g = ops.adaptive_avg_pool2d(out_l5, output_size=1)
            output_g = ops.dropout(output_g, p=0.2, training=self.training)
            output_1x1 = output_g.view(output_g.shape(0), -1)

            return self.fc(output_1x1)
        return out_l1, out_l2, out_l3, out_l4


def eespnet(pretrained=False, **kwargs):
    model = EESPNet(**kwargs)
    if pretrained:
        raise ValueError("Don't support pretrained")
    return model


if __name__ == '__main__':
    img = ops.randn(1, 3, 224, 224)
    model = eespnet()
    out = model(img)
    print(img)
