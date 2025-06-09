tax_model_setup = {
    "metrics": ["f1_score"],  # must be in custom_objects in custom_objects in NN_losses_metrics_activations.py

    # important for HP tuning and early stopping
    "monitoring": {"objective": "val_f1_score",  # if is not loss, must be included in custom_objects and metrics
                   "direction": "max"  # minimise or maximise the objective (for HP tuning)?
                   },

    "trim_mean_cut": 0.2,  # parameter of trim_mean in evaluation

    "model_subdir": "taxonomy"  # subdirectory where to save models
}
