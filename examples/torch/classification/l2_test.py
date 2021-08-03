import torch
from main import AverageMeter, compare_layers
import matplotlib.pyplot as plt
from functools import reduce

m_dict = torch.load("/home/tagir/work/nncf/examples/result/pruning/experiments/resnet50/pretune_only/ResNet50_cifar_cifar100_filter_pruning/2021-08-02__16-40-16/ResNet50_cifar_cifar100_filter_pruning_pretune.pth")
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

plt.plot(res)
plt.savefig('l2_test_resnet50.png')