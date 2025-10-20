from utils import *
from training_static_masks import *


CONFIG = {
    "n_epochs": 200,
    "batch_size": 32,
    "model_learning_rate": 0.0001,
    "mask_learning_rate": 0.5,
    "lambda_init": 0.01,
    "lambda_factor": 1.5,
    "lambda_patience": 5,
    "lambda_treshold": 0.2,
    "training_id": "cnn_mnist_run_SGD_01" # Descriptive name for the training run
}

grid_training(training_loop, CONFIG, "lambda_factor", 1.1 , 1.5, 0.1, optimizer_class=optim.SGD)


    