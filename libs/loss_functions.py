import torch.nn as nn

def build_loss_function(loss_function_config):
    """
    Factory method to build a loss function based on the specified configuration.
    """

    loss_function_id = loss_function_config['id']
    loss_function_params = loss_function_config.get('params', {})
    loss_function = {
        "CrossEntropyLoss": nn.CrossEntropyLoss
    }[loss_function_id](**loss_function_params)
    return loss_function
