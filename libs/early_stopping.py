class BasicPatienceEarlyStopping:
    """
    Early stopping detector to stop training when the model stops improving.
    """

    def __init__(self, metric, goal="maximize", patience=5, min_delta=0):
        self.metric = metric
        self.goal = goal
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop_triggered = False
        self.is_better_score_fn = lambda x, y: x > y if self.goal == "maximize" else x < y

    def __call__(self, metrics):
        assert self.early_stop_triggered == False, "Early stopping is already triggered"
        score = metrics[self.metric]
        best_score_threshold = self.best_score + self.min_delta if self.best_score is not None else None
        if self.best_score is None or self.is_better_score_fn(score, best_score_threshold): self.best_score, self.counter = score, 0
        else: self.early_stop_triggered, self.counter = self.counter >= self.patience, self.counter + 1
        return self.early_stop_triggered, self.best_score, self.counter, self.patience

EARLY_STOPPINGS = {_class.__name__: _class for _class in [
    BasicPatienceEarlyStopping
]}

def build_early_stopping(config):
    _id = config['id']
    metric = config["metric"]
    _kwargs = {k: v for k, v in config.items() if k not in ["id", "metric"]}
    constructor = EARLY_STOPPINGS[_id]
    loss_function = constructor(metric, **_kwargs)
    return loss_function
