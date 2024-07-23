import torch.optim as optim

LR_SCHEDULERS = {_class.__name__: _class for _class in [
    optim.lr_scheduler.StepLR,
    optim.lr_scheduler.CyclicLR,
    optim.lr_scheduler.ReduceLROnPlateau,
    optim.lr_scheduler.CosineAnnealingLR,
    optim.lr_scheduler.OneCycleLR
]}

def build_lr_scheduler(optimizer, config):
    """
    Factory method to build a learning rate scheduler based on the specified configuration.
    """

    _id = config['id']
    _kwargs = {k: v for k, v in config.items() if k not in ["id"]}
    constructor = LR_SCHEDULERS[_id]
    loss_function = constructor(optimizer, **_kwargs)
    return loss_function
