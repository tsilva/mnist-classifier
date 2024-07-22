import torch.optim as optim

def build_lr_scheduler(optimizer, scheduler_config):
    """
    Factory method to build a learning rate scheduler based on the specified configuration.
    """

    if not scheduler_config: return None
    scheduler_id = scheduler_config['id']
    scheduler_params = scheduler_config.get('params', {})
    scheduler = { # TODO: infer names from class names, apply to other factories
        "StepLR": optim.lr_scheduler.StepLR,
        "CyclicLR": optim.lr_scheduler.CyclicLR,
        "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
        "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
        "OneCycleLR": optim.lr_scheduler.OneCycleLR
    }[scheduler_id](optimizer, **scheduler_params)
    return scheduler

