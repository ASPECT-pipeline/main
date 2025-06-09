import numpy as np

comp_model_setup = {
    "metrics": ["mse"],  # must be in custom_objects in NN_losses_metrics_activations.py

    # important for HP tuning and early stopping
    "monitoring": {"objective": "val_loss",  # if not loss, must be included in custom_objects and metrics
                   "direction": "min"  # minimise or maximise the objective (for HP tuning)?
                   },

    "trim_mean_cut": 0.2,  # parameter of trim_mean in evaluation

    "model_subdir": "composition"  # subdirectory where to save models
}


