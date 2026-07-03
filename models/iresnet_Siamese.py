# Adapted from https://github.com/deepinsight/insightface
# Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 4690-4699).

import torch
import torch.nn as nn
from .iresnet import IResNet, IBasicBlock

class IResNet_LastLayerModified_Siamese(IResNet):
    def __init__(self, n_features, merge_locs, *args, **kwargs):
        super(IResNet_LastLayerModified_Siamese, self).__init__(*args, **kwargs)
        self.flatten = lambda x: torch.flatten(x, 1)
        self.fc = nn.Linear(512 * 7 * 7 * len(merge_locs), n_features)
        del self.features

        self.funcs = [self.conv1, self.bn1, self.prelu, self.layer1, self.layer2, self.layer3, self.layer4, self.bn2, self.flatten, self.dropout]
        self.func_names = ['conv1', 'bn1', 'prelu', 'stage1', 'stage2', 'stage3', 'stage4', 'bn2', 'flatten', 'fc']
        self.merge_locs = merge_locs

    def forward(self, x):
        x1 = x[:,0,:]
        x2 = x[:,1,:]
        x_features = []
        merge_locs = self.merge_locs.copy()

        
        for func, name in zip(self.funcs, self.func_names):
            if name in merge_locs:
                # Merge the two networks
                x_features.append(x1 - x2)
                merge_locs.remove(name)
            if len(self.merge_locs) == 0 or len(merge_locs) > 0:
                x1 = func(x1)
                x2 = func(x2)
            x_features = [func(x) for x in x_features]

        if 'output' not in merge_locs:
            outputs = self.fc(torch.cat(x_features, dim=1))
        else:
            # If the merge location is "output", then directly compute the difference for the outputs of the two networks
            outputs = self.fc(x2) - self.fc(x1)
        return outputs

def _iresnet_Siamese(n_features, merge_locs, basic_block, n_blocks, pretrained=False, weights_path=None, **kwargs):
    model = IResNet_LastLayerModified_Siamese(n_features, merge_locs, IBasicBlock, n_blocks, **kwargs)
    if pretrained == True:
        weights = torch.load(weights_path)
        weights = {k: v for k, v in weights.items() if k in model.state_dict() and weights[k].size() == model.state_dict()[k].size()}
        model.load_state_dict(weights, strict=False)
    return model
    
def iresnet_Siamese(n_layers, n_features, merge_locs, pretrained=False, weights_path=None, **kwargs):
    if n_layers == 18:
        n_blocks = [2, 2, 2, 2]
    elif n_layers == 34:
        n_blocks = [3, 4, 6, 3]
    elif n_layers == 50:
        n_blocks = [3, 4, 14, 3]
    elif n_layers == 100:
        n_blocks = [3, 13, 30, 3]
    elif n_layers == 200:
        n_blocks = [6, 26, 60, 6]
    else:
        raise ValueError('Invalid layer number')
    return _iresnet_Siamese(n_features, merge_locs, IBasicBlock, n_blocks, pretrained=pretrained, weights_path=weights_path, **kwargs)