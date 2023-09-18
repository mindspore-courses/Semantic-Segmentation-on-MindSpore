from mindspore import nn, ops
from mindspore.ops import operations as P


class _DenseLayer(nn.Cell):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, dilation=1):
        super(_DenseLayer, self).__init__()
        self.add = P.TensorAdd()
        self.drop_rate = drop_rate
        self.cat = P.Concat(axis=1)

        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, pad_mode='pad', padding=dilation,
                               dilation=dilation)

    def construct(self, x):
        new_features = self.norm1(x)
        new_features = self.relu1(new_features)
        new_features = self.conv1(new_features)
        new_features = self.norm2(new_features)
        new_features = self.relu2(new_features)
        new_features = self.conv2(new_features)
        if self.drop_rate > 0:
            new_features = ops.dropout(new_features, p=self.drop_rate, training=self.training)
        return self.cat((x, new_features))


class _DenseBlock(nn.Cell):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, dilation=1):
        super(_DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, dilation)
            layers.append(layer)
        self.layers = nn.SequentialCell(layers)

    def construct(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _Transition(nn.Cell):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # Keep pooling here

    def construct(self, x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)  # Apply pooling here
        return x


class DenseNet(nn.Cell):
    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16), num_init_features=64,
                 bn_size=4, drop_rate=0, num_classes=1000, dilate_scale=1):
        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.SequentialCell([
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, pad_mode='pad', padding=3),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='pad', padding=1),
        ])

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            dilation = dilate_scale if i in [2, 3] and dilate_scale == 8 else (
                2 * dilate_scale if i == 3 and dilate_scale == 16 else 1)
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate, dilation)
            self.features.append(block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features // 2)
                self.features.append(trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.append(nn.BatchNorm2d(num_features))

        # Linear layer
        self.flatten = nn.Flatten()
        self.classifier = nn.Dense(num_features, num_classes)

    def construct(self, x):
        features = self.features(x)
        out = P.ReLU()(features)
        out = nn.AdaptiveAvgPool2d((1, 1))(out)
        out = self.flatten(out)
        out = self.classifier(out)
        return out


class DilatedDenseNet(DenseNet):
    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16), num_init_features=64,
                 bn_size=4, num_classes=1000, dilate_scale=8):
        super(DilatedDenseNet, self).__init__(growth_rate, block_config, num_init_features,
                                              bn_size, num_classes)
        assert (dilate_scale == 8 or dilate_scale == 16), "dilate_scale can only set as 8 or 16"
        if dilate_scale == 8:
            self._apply_dilation(self.features[6], 2)
            self._apply_dilation(self.features[8], 4)
            self.features[7].pool = nn.Identity()  # Replace pooling with identity in transition2
            self.features[9].pool = nn.Identity()  # Replace pooling with identity in transition3
        elif dilate_scale == 16:
            self._apply_dilation(self.features[8], 2)
            self.features[9].pool = nn.Identity()  # Replace pooling with identity in transition3

    def _apply_dilation(self, block, dilate):
        for layer in block.layers:
            self._conv_dilate(layer.conv2, dilate)

    def _conv_dilate(self, m, dilate):
        if m.kernel_size == (3, 3):
            m.pad_mode = 'pad'
            m.padding = (dilate, dilate)
            m.dilation = (dilate, dilate)


# Specification
densenet_spec = {121: (64, 32, [6, 12, 24, 16]),
                 161: (96, 48, [6, 12, 36, 24]),
                 169: (64, 32, [6, 12, 32, 32]),
                 201: (64, 32, [6, 12, 48, 32])}


# Constructor
def get_densenet(num_layers, pretrained=False, **kwargs):
    r"""Densenet-BC model from the
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.

    Parameters
    ----------
    num_layers : int
        Number of layers for the variant of densenet. Options are 121, 161, 169, 201.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default $TORCH_HOME/models
        Location for keeping the model parameters.
    """
    num_init_features, growth_rate, block_config = densenet_spec[num_layers]
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    return model


def get_dilated_densenet(num_layers, dilate_scale, pretrained=False, **kwargs):
    num_init_features, growth_rate, block_config = densenet_spec[num_layers]
    model = DilatedDenseNet(growth_rate, block_config, num_init_features, dilate_scale=dilate_scale)
    return model


def densenet121(**kwargs):
    return get_densenet(121, **kwargs)


def densenet161(**kwargs):
    return get_densenet(161, **kwargs)


def densenet169(**kwargs):
    return get_densenet(169, **kwargs)


def densenet201(**kwargs):
    return get_densenet(201, **kwargs)


def dilated_densenet121(dilate_scale, **kwargs):
    return get_dilated_densenet(121, dilate_scale, **kwargs)


def dilated_densenet161(dilate_scale, **kwargs):
    return get_dilated_densenet(161, dilate_scale, **kwargs)


def dilated_densenet169(dilate_scale, **kwargs):
    return get_dilated_densenet(169, dilate_scale, **kwargs)


def dilated_densenet201(dilate_scale, **kwargs):
    return get_dilated_densenet(201, dilate_scale, **kwargs)


if __name__ == '__main__':
    img = ops.randn(2, 3, 224, 224)
    model = dilated_densenet121(8)
    outputs = model(img)
    print(model)