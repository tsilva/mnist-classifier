import wandb

def parse_wandb_sweep_config(sweep_config):
    """
    Parse the sweep configuration from W&B to a dictionary.
    Mapping dependent parameters from the flattened structure of the sweep config to the nested structure of our config.
    Handling special list-like keys with the prefix "list__".
    """
    def parse_dict(d):
        config = {}
        list_keys = {}

        for key, value in d.items():
            if key == "method":
                continue
            
            if key.startswith("list__"):
                # Process list-like keys
                parts = key.split("__")
                list_name = parts[1]
                index = int(parts[2]) - 1  # Convert to zero-based index

                if list_name not in list_keys:
                    list_keys[list_name] = []
                
                # Ensure the list is large enough
                while len(list_keys[list_name]) <= index:
                    list_keys[list_name].append(None)
                
                list_keys[list_name][index] = value
            else:
                if isinstance(value, dict) and "id" in value:
                    _id = value["id"]
                    _id_l = _id.lower()
                    params = {k.replace(f"{_id_l}_", ""): v for k, v in value.items() if k.startswith(_id_l)}
                    value = {"id": _id, "params": params}
                elif isinstance(value, dict):
                    value = parse_dict(value)
                
                config[key] = value

        # Merge list-like keys into the config
        for list_name, list_values in list_keys.items():
            config[list_name] = list_values

        return config

    return parse_dict(sweep_config)

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
    