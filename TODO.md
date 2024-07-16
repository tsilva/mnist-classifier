# TODO

- [ ] Add support for training bagging ensemble
- [ ] Add support for training stacking ensemble
- [ ] Add support for training boosting ensemble
- [ ] Log early stopping metrics
- [ ] Save model metadata along with model file (eg: hyperparameters, training results, etc)
- [ ] BUG: GPU is still not maxed out during training, figure out bottleneck

- [ ] Add NVIDIA DALI support for faster data loading
- [ ] Add support for saving model as ONNX
- [ ] Add support for visualizing model activation maps
- [ ] Add support for checkpointing during training and resuming from checkpoints
- [ ] BUG: Crashing processes are not sending logs to W&B
- [ ] Figure out how to use multiple seeds when testing an hyperparameter configuration during sweep (if I just do a mean of the score, each intermediate run will still be logged to the same run)