# This code comes from below link. Please follow assigned license
# https://github.com/tatsy/normalizing-flows-pytorch

import torch
import torch.nn as nn
import torch.nn.init as init


def safe_detach(x):
    """
    detech operation which keeps reguires_grad
    ---
    https://github.com/rtqichen/residual-flows/blob/master/lib/layers/iresblock.py
    """
    return x.detach().requires_grad_(x.requires_grad)


def weights_init_as_nearly_identity(m):
    """
    initialize weights such that the layer becomes nearly identity mapping
    """
    classname = m.__class__.__name__
    if classname.find("Linear") != -1 or classname.find("Conv") != -1:
        # check if module is wrapped by spectral norm
        if hasattr(m, "weight"):
            nn.init.constant_(m.weight, 0)
        else:
            nn.init.constant_(m.weight_bar, 0)
        nn.init.normal_(m.bias, 0, 0.01)


def anomaly_hook(self, inputs, outputs):
    """
    module hook for detecting NaN and infinity
    """
    if not isinstance(outputs, tuple):
        outputs = [outputs]
    else:
        outputs = list(outputs)

    for i, out in enumerate(outputs):
        inf_mask = torch.isinf(out)
        nan_mask = torch.isnan(out)
        if inf_mask.any():
            print("In module:", self.__class__.__name__)
            print(
                f"Found NAN in output {i} at indices: ",
                inf_mask.nonzero(),
                "where:",
                out[inf_mask.nonzero(as_tuple=False)[:, 0].unique(sorted=True)],
            )

        if nan_mask.any():
            print("In", self.__class__.__name__)
            print(
                f"Found NAN in output {i} at indices: ",
                nan_mask.nonzero(),
                "where:",
                out[nan_mask.nonzero(as_tuple=False)[:, 0].unique(sorted=True)],
            )

        if inf_mask.any() or nan_mask.any():
            raise RuntimeError("Foud INF or NAN in output of", self.___class__.__name__, "from input tensor", inputs)


def sum(tensor, dim=None, keepdim=False):
    if dim is None:
        # sum up all dim
        return torch.sum(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.sum(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dim):
                tensor.squeeze_(d - i)
        return tensor


def mean(tensor, dim=None, keepdim=False):
    if dim is None:
        # mean all dim
        return torch.mean(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dim):
                tensor.squeeze_(d - i)
        return tensor


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, : C // 2, ...], tensor[:, C // 2 :, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


def cat_feature(tensor_a, tensor_b):
    return torch.cat((tensor_a, tensor_b), dim=1)


def pixels(tensor):
    return int(tensor.size(2) * tensor.size(3))


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)
