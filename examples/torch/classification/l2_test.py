import torch
from main import AverageMeter, compare_layers
import matplotlib.pyplot as plt
from functools import reduce

x = torch.rand(4,2,2,2)
print(x)
print(x[[False,True,False,True]].flatten(1))
"""
m_dict = torch.load("/home/tagir/work/nncf/examples/result/pruning/experiments/resnet18/greg2_negative_coef/model/ResNet18_cifar_cifar100_filter_pruning_after_greg1.pth",
map_location=torch.device('cpu'))
res = []

pruned_layers = m_dict['weights_to_prune'].keys()
left_layers = set(m_dict['weights_to_prune'].keys())

for layer in m_dict['state_dict'].keys():
    layer_weights = m_dict['state_dict'][layer]
    for nncf_l in pruned_layers:
        if compare_layers(nncf_l, layer):
            left_layers.remove(nncf_l)
            for filter_idx in m_dict['weights_to_prune'][nncf_l]:
                res.append(layer_weights[filter_idx].norm(p=2))

print(left_layers)
print(len(res))

plt.plot(m_dict['l2_p_log'])
plt.savefig('l2_neg_resnet18_prun.png')
"""