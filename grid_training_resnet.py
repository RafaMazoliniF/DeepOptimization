from utils import *
from training_static_masks import *

CONFIG = {
    "n_epochs": 200,
    "batch_size": 128,
    "model_learning_rate": 0.00001,
    "mask_learning_rate": 0.001,
    "lambda_init": 0.01,
    "lambda_factor": 1.5,
    "lambda_patience": 5,
    "lambda_treshold": 0.2,
    "training_id": "resNet_cifar10_run_ADAM_02"
}

grid_training(training_loop, CONFIG, "mask_learning_rate", 0.001 ,0.1, 10, optimizer_class=optim.Adam)


    