import torch
from resnet import ResNet18


def model_loader(model_name):
    if model_name is 'RESNET18':
        net = ResNet18()
        feat_size = 512

    else:
        raise NameError('UNKNOWN MODEL NAME')
    return (net, feat_size)

