class BasicPatienceEarlyStopping:
    """
    Early stopping detector to stop training when the model stops improving.
    """

    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop_triggered = False

    def __call__(self, score):
        assert self.early_stop_triggered == False, "Early stopping is already triggered"
        best_score_threshold = self.best_score + self.min_delta if self.best_score is not None else None
        if self.best_score is None or score >= best_score_threshold: self.best_score, self.counter = score, 0
        else: self.early_stop_triggered, self.counter = self.counter >= self.patience, self.counter + 1
        return self.early_stop_triggered, self.best_score, self.counter, self.patience


def build_early_stopping(early_stopping_config):
    if not early_stopping_config: return None
    early_stopping_id = early_stopping_config['id']
    early_stopping_params = early_stopping_config.get('params', {})
    early_stopping = {
        "BasicPatienceEarlyStopping": BasicPatienceEarlyStopping
    }[early_stopping_id](**early_stopping_params)
    return early_stopping
