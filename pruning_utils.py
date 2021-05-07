import models
from torch.nn.utils import prune
import torch


def prune_transformer_block(transformer_block, args):
    pruning_amount = float(args.pruning_amount)
    prune.ln_structured(transformer_block.fc1, name='weight', amount=pruning_amount, n=0, dim=0)
    prune.remove(transformer_block.fc1, 'weight')
    prune.ln_structured(transformer_block.fc2, name='weight', amount=pruning_amount, n=0, dim=0)
    prune.remove(transformer_block.fc2, 'weight')
    for sub_module in transformer_block.fc_delta:
        if isinstance(sub_module, torch.nn.Linear):
            prune.ln_structured(sub_module, name='weight', amount=pruning_amount, n=0, dim=0)
            prune.remove(sub_module, 'weight')
    for sub_module in transformer_block.fc_gamma:
        if isinstance(sub_module, torch.nn.Linear):
            prune.ln_structured(sub_module, name='weight', amount=pruning_amount, n=0, dim=0)
            prune.remove(sub_module, 'weight')
    return transformer_block


def prune_model(model, args):
    pruning_style = args.pruning_style
    prune_layers = []
    if pruning_style == 'bottom':
        prune_layers = [1, 2]
    if pruning_style == ' alternate':
        prune_layers = [1, 3, 5]
    if pruning_style == 'top':
        prune_layers = [4, 5]
    transformer_block_count = 0
    for idx in range(len(list(model.modules()))):
        module = list(model.modules())[idx]
        if isinstance(module, models.Hengshuang.transformer.TransformerBlock):
            transformer_block_count += 1
            if transformer_block_count in prune_layers:
                module = prune_transformer_block(module, args)
                list(model.modules())[idx] = module
    return model
