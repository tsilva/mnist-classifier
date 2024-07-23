import torch.nn as nn

LOSS_FUNCTIONS = {_class.__name__: _class for _class in [
    nn.CrossEntropyLoss
]}

def build_loss_function(config):
    """
    Factory method to build a loss function based on the specified configuration.
    """

    _id = config['id']
    _kwargs = {k: v for k, v in config.items() if k not in ["id"]}
    constructor = LOSS_FUNCTIONS[_id]
    loss_function = constructor(**_kwargs)
    return loss_function

