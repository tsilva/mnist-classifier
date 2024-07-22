import torch.optim as optim

def build_optimizer(model, optimizer_config):
    """
    Factory method to build an optimizer based on the specified configuration.
    """

    optimizer_id = optimizer_config['id']
    optimizer_params = optimizer_config.get('params', {})
    optimizer = {
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
        "SGD": optim.SGD
    }[optimizer_id](model.parameters(), **optimizer_params)
    return optimizer

