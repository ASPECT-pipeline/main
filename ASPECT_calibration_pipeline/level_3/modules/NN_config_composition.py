import numpy as np
from modules.NN_config_parse import gimme_used_quantities

comp_output_setup = {
    "minerals": np.array([True,  # olivine
                          True,  # orthopyroxene
                          True,  # clinopyroxene
                          False]),  # plagioclase

    "endmembers": [[True, True],  # Fa, Fo; OL
                   [True, True, False],  # Fs, En, Wo; OPX
                   [True, True, True],  # Fs, En, Wo; CPX
                   [False, False, False]]  # An, Ab, Or; PLG
}


comp_model_setup = {
    "metrics": ["mse"],  # must be in custom_objects in NN_losses_metrics_activations.py

    # important for HP tuning and early stopping
    "monitoring": {"objective": "val_loss",  # if not loss, must be included in custom_objects and metrics
                   "direction": "min"  # minimise or maximise the objective (for HP tuning)?
                   },

    "trim_mean_cut": 0.2,  # parameter of trim_mean in evaluation

    "model_subdir": "composition"  # subdirectory where to save models
}


# used minerals and end-members
minerals_used, endmembers_used = gimme_used_quantities(**comp_output_setup)
