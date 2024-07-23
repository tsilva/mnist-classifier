import torch.optim as optim

OPTIMIZERS = {_class.__name__: _class for _class in [
    optim.Adam,
    optim.AdamW,
    optim.SGD
]}

def build_optimizer(model, config):
    """
    Factory method to build an optimizer based on the specified configuration.
    """

    _id = config['id']
    _kwargs = {k: v for k, v in config.items() if k not in ["id"]}
    constructor = OPTIMIZERS[_id]
    loss_function = constructor(model.parameters(), **_kwargs)
    return loss_function
