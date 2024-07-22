import wandb

def parse_wandb_sweep_config(sweep_config):
    """
    Parse the sweep configuration from W&B to a dictionary.
    Mapping dependent parameters from the flattened structure of the sweep config to the nested structure of our config.
    """

    config = {}
    for key, value in sweep_config.items():
        if key == "method": continue

        if isinstance(value, dict) and "id" in value:
            _id = value["id"]
            _id_l = _id.lower()
            params = {k.replace(f"{_id_l}_", ""): v for k, v in value.items() if k.startswith(_id_l)}
            value = {"id": _id, "params": params}
        
        config[key] = value
    return config

class OptionalWandbContext:

    def __init__(self, use_wandb, *args, **kwargs):
        self.use_wandb = use_wandb
        self.args = args
        self.kwargs = kwargs
        self.run = None

    def __enter__(self):
        if self.use_wandb:
            self.run = wandb.init(*self.args, **self.kwargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_wandb and self.run:
            self.run.finish()

    def log(self, *args, **kwargs):
        if self.use_wandb:
            wandb.log(*args, **kwargs)

    def watch(self, *args, **kwargs):
        if self.use_wandb:
            wandb.watch(*args, **kwargs)

    def save(self, *args, **kwargs):
        if self.use_wandb:
            wandb.save(*args, **kwargs)

    def Artifact(self, *args, **kwargs):
        if self.use_wandb:
            return wandb.Artifact(*args, **kwargs)
        return None

    def log_artifact(self, *args, **kwargs):
        if self.use_wandb and self.run:
            self.run.log_artifact(*args, **kwargs)
    