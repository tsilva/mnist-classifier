# TODO

- [ ] Add data augmentation support (configurable in hyperparams)
- [ ] Add support for saving model as ONNX
- [ ] Add support for visualizing model activation maps
- [ ] Add support for checkpointing during training and resuming from checkpoints
- [ ] BUG: Crashing processes are not sending logs to W&B
- [ ] Figure out how to use multiple seeds when testing an hyperparameter configuration during sweep (if I just do a mean of the score, each intermediate run will still be logged to the same run)