import numpy as np 
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from tensorflow.python.framework.ops import EagerTensor
import tensorflow.keras.backend as K
from tensorflow.keras.activations import relu, sigmoid, softmax
from typing import Callable
from NN_config_parse import (gimme_num_minerals, gimme_endmember_counts)
from utilities_spectra import gimme_indices
from _constants import _wp
from sklearn.metrics import f1_score as f1_sklearn



def gimme_penalisation_setup(penalised_mineral: str, used_minerals: np.ndarray | None = None,
                             used_endmembers: list[list[bool]] | None = None) -> tuple[float, int, list[dict], int]:
    # if used_minerals is None: used_minerals = minerals_used # Not found in GitHub
    # if used_endmembers is None: used_endmembers = endmembers_used # Not found in GitHub
    if used_minerals is None: print(f'used_minerals is None. Check NN_losses_metrics_activations.py')
    if used_endmembers is None: print(f'used_endmembers is None. Check NN_losses_metrics_activations.py')

    count_endmembers = gimme_endmember_counts(used_endmembers)
    used_minerals_int = K.cast(used_minerals, dtype=tf.int32)

    num_minerals = tf.numpy_function(lambda minerals: K.cast(gimme_num_minerals(minerals), dtype=tf.int32),
                                     inp=[used_minerals_int], Tout=tf.int32)

    beta = 5.  # Penalisation of the forbidden regions

    # These are needed due to "abandoned" regions
    if penalised_mineral == "orthopyroxene":
        OPX_Wo_limit = 0.10  # 10% Wo max

        i, j = 1, 2  # OPX, Wo (indices in used_endmembers)
        if used_endmembers[i][j]:
            mineral_position = K.sum(used_minerals_int[:i])
            indices = (num_minerals + K.cast(K.sum(count_endmembers[:i]), dtype=tf.int32) +
                       K.sum(K.cast(used_endmembers[i][:j], dtype=tf.int32)),)
            limits = (OPX_Wo_limit,)
            boundaries = ("upper",)

            use_penalisation = 1.

        else:
            # Use the first index which is always present
            mineral_position, indices, limits, boundaries, use_penalisation = 0, (0,), (1.,), ("upper",), 0

    if penalised_mineral == "clinopyroxene":
        CPX_Wo_limit = 0.60  # 60% Wo max

        i, j = 2, 2  # CPX, Wo (indices in used_endmembers)
        if used_endmembers[i][j]:
            mineral_position = K.sum(used_minerals_int[:i])
            indices = (num_minerals + K.cast(K.sum(count_endmembers[:i]), dtype=tf.int32) +
                       K.sum(K.cast(used_endmembers[i][:j], dtype=tf.int32)),)
            limits = (CPX_Wo_limit, )
            boundaries = ("upper", )

            use_penalisation = 1.

        else:
            # Use the first index which is always present
            mineral_position, indices, limits, boundaries, use_penalisation = 0, (0,), (1.,), ("upper",), 0

    if penalised_mineral == "plagioclase":
        # This roughly delimits the forbidden region
        PLG_An_limit = 0.15  # 15% upper limit
        PLG_Ab_limit = 0.50  # 50% lower limit
        PLG_Or_limit = 0.15  # 15% upper limit

        i = 3  # PLG
        if np.all(used_endmembers[i]):
            mineral_position = K.sum(used_minerals_int[:i])
            indices = tuple((num_minerals + K.cast(K.sum(count_endmembers[:i]), dtype=tf.int32) +
                             K.sum(K.cast(used_endmembers[i][:j], dtype=tf.int32))) for j in range(3))
            limits = (PLG_An_limit, PLG_Ab_limit, PLG_Or_limit)
            boundaries = ("upper", "lower", "upper")

            use_penalisation = 1.

        elif used_endmembers[i][0] and used_endmembers[i][2]:  # no Ab; An lower than 15% or larger than 85%
            mineral_position = K.sum(used_minerals_int[:i])
            indices = (num_minerals + K.cast(K.sum(count_endmembers[:i]), dtype=tf.int32) + 0,
                       num_minerals + K.cast(K.sum(count_endmembers[:i]), dtype=tf.int32) + 0
                       )
            limits = (PLG_An_limit, 1. - PLG_Or_limit)
            boundaries = ("upper", "lower")

            use_penalisation = 1.

        else:
            # Use the first index which is always present
            mineral_position, indices, limits, boundaries, use_penalisation = 0, (0,), (1.,), ("upper",), 0

    setup = [{"index": indices[i],
              "limit": limits[i],
              "boundary": boundaries[i]} for i in range(len(indices))]

    if num_minerals == 0:  # no multiplication with w_true in the penalisation
        mineral_position = -1

    return beta, mineral_position, setup, use_penalisation



def penalisation_function(y_true: tf.Tensor, y_pred: tf.Tensor, penalised_mineral: str,
                          used_minerals: np.ndarray | None = None,
                          used_endmembers: list[list[bool]] | None = None) -> tf.Tensor:
     # if used_minerals is None: used_minerals = minerals_used # Not found in GitHub
    # if used_endmembers is None: used_endmembers = endmembers_used # Not found in GitHub
    if used_minerals is None: print(f'used_minerals is None. Check NN_losses_metrics_activations.py')
    if used_endmembers is None: print(f'used_endmembers is None. Check NN_losses_metrics_activations.py')


    beta, mineral_position, setup, use_penalisation = gimme_penalisation_setup(penalised_mineral=penalised_mineral,
                                                                               used_minerals=used_minerals,
                                                                               used_endmembers=used_endmembers)

    if mineral_position < 0:  # no minerals
        w_true = 1.
    else:
        w_true = y_true[:, mineral_position]

    # This is here to penalize the region with a non-existent solid solution
    dists = [relu(y_pred[:, s["index"]] - s["limit"]) if s["boundary"] == "upper" else
             relu(s["limit"] - y_pred[:, s["index"]]) for s in setup]

    # closest distance from the borderline
    dist = tnp.min(dists, axis=0)
    penalisation = K.sum(dist * w_true)

    return beta * penalisation * use_penalisation

def create_class_weight(y_true: tf.Tensor, mu: float = 0.15) -> tf.Tensor:
    # another option is to use class_weight.compute_class_weight from sklearn.utils
    if mu <= 0.:
        raise ValueError(f'"mu" must be a positive number but is {mu}.')

    counts = K.cast(K.sum(y_true, axis=0), dtype=_wp)
    total = K.cast(K.sum(counts), dtype=_wp)

    counts = tf.where(counts > 0., counts, 1.)

    weights = K.log(mu * total / counts)
    weights = tf.where(weights >= 1., weights, 1.)

    return weights

def cross_entropy_base(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    # Scale predictions so that the class probabilities of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())

    # calc
    loss = -y_true * K.log(y_pred)

    return loss

def my_mse_loss(used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None,
                alpha: float | None = 1.) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    # if used_minerals is None: used_minerals = minerals_used # Not found in GitHub
    # if used_endmembers is None: used_endmembers = endmembers_used # Not found in GitHub
    if used_minerals is None: print(f'used_minerals is None. Check NN_losses_metrics_activations.py')
    if used_endmembers is None: print(f'used_endmembers is None. Check NN_losses_metrics_activations.py')
    if alpha is None: alpha = 1.

    if alpha < 0.:
        raise ValueError(f'"alpha" must be a non-negative number but is {alpha}.')

    num_minerals = gimme_num_minerals(used_minerals)

    if num_minerals and np.sum(gimme_endmember_counts(used_endmembers)) > 0:
        @tf.function
        def mse_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            indices = gimme_indices(used_minerals, used_endmembers, return_mineral_indices=True)
            start, stop = indices[0, 1:]

            w_true, w_pred = y_true[:, start:stop], y_pred[:, start:stop]
            w_square = K.sum(K.square(w_true - w_pred))

            wz = 0.0

            for i, start, stop in indices[1:]:
                z_true, z_pred = y_true[:, start:stop], y_pred[:, start:stop]

                z_square = K.square(z_true - z_pred)
                wz += K.sum(K.transpose(K.transpose(z_square) * w_true[:, i]))

            wz += penalisation_function(y_true, y_pred, "orthopyroxene",
                                        used_minerals=used_minerals, used_endmembers=used_endmembers)
            wz += penalisation_function(y_true, y_pred, "clinopyroxene",
                                        used_minerals=used_minerals, used_endmembers=used_endmembers)
            wz += penalisation_function(y_true, y_pred, "plagioclase",
                                        used_minerals=used_minerals, used_endmembers=used_endmembers)

            return w_square + alpha * wz

    elif num_minerals == 0:
        @tf.function
        def mse_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            wz = 0.0

            for start, stop in gimme_indices(used_minerals, used_endmembers):
                z_true, z_pred = y_true[:, start:stop], y_pred[:, start:stop]

                z_square = K.square(z_true - z_pred)
                wz += K.sum(z_square)

            wz += penalisation_function(y_true, y_pred, "orthopyroxene",
                                        used_minerals=used_minerals, used_endmembers=used_endmembers)
            wz += penalisation_function(y_true, y_pred, "clinopyroxene",
                                        used_minerals=used_minerals, used_endmembers=used_endmembers)
            wz += penalisation_function(y_true, y_pred, "plagioclase",
                                        used_minerals=used_minerals, used_endmembers=used_endmembers)

            return wz

    else:
        @tf.function
        def mse_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            return K.sum(K.square(y_true - y_pred))

    return mse_loss

def my_focal_loss(gamma: float = 2., use_weights: bool = True) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    @tf.function
    def focal_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        weights = create_class_weight(y_true=y_true) if use_weights else 1.
        return K.sum(weights * K.pow(1. - y_pred, gamma) * cross_entropy_base(y_true, y_pred), axis=-1)

    return focal_loss

def gimme_composition_loss(used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None,
                 alpha: float | None = 1.):
    # None for taxonomy models
    # if used_minerals is None: used_minerals = minerals_used # Not found in GitHub
    # if used_endmembers is None: used_endmembers = endmembers_used # Not found in GitHub
    if used_minerals is None: print(f'used_minerals is None. Check NN_losses_metrics_activations.py')
    if used_endmembers is None: print(f'used_endmembers is None. Check NN_losses_metrics_activations.py')

    if alpha is None: alpha = 1.

    loss_composition = my_mse_loss(used_minerals=used_minerals, used_endmembers=used_endmembers, alpha=alpha)
    return loss_composition

def gimme_taxonomy_loss(use_weights: bool | None = None):
    if use_weights is None: use_weights = True

    loss_taxonomy = my_focal_loss(gamma=2., use_weights=use_weights)
    return loss_taxonomy

def delete_wtrue_zero_samples(z_true_part: tf.Tensor, z_pred_part: tf.Tensor,
                              w_true_part: tf.Tensor) -> tuple[tf.Tensor, ...]:
    # This is only needed if num_minerals > 0

    mask = tf.greater(w_true_part, 0.)

    if K.any(mask):
        mask = K.reshape(mask, (tf.size(mask), 1))
        mask = tf.repeat(mask, repeats=K.shape(z_true_part)[1], axis=-1)

        z_true_clean = tf.where(mask, z_true_part, tf.fill(K.shape(z_true_part), np.nan))
        z_pred_clean = tf.where(mask, z_pred_part, tf.fill(K.shape(z_pred_part), np.nan))

    else:  # this is here if mask is empty due to low batch_size
        z_true_clean = tf.fill(K.shape(z_true_part), 0.0)
        z_pred_clean = tf.fill(K.shape(z_pred_part), 0.0)

    return z_true_clean, z_pred_clean

def clean_ytrue_ypred(y_true: tf.Tensor, y_pred: tf.Tensor,
                      used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None,
                      cleaning: bool = True, all_to_one: bool = False) -> tuple[tf.Tensor, ...]:
    # cleaning = False is important for classification models

    # if used_minerals is None: used_minerals = minerals_used # Not found in GitHub
    # if used_endmembers is None: used_endmembers = endmembers_used # Not found in GitHub
    if used_minerals is None: print(f'used_minerals is None. Check NN_losses_metrics_activations.py')
    if used_endmembers is None: print(f'used_endmembers is None. Check NN_losses_metrics_activations.py')

    # If numpy, convert to tensor
    if isinstance(y_true, np.ndarray):
        y_true, y_pred = K.constant(y_true), K.constant(y_pred)

    if gimme_num_minerals(used_minerals) > 0 and cleaning:
        for i, start, stop in gimme_indices(used_minerals, used_endmembers, return_mineral_indices=True):
            if i < 0:  # Contribution of modal compositions
                y_true_clean, y_pred_clean = y_true[:, start:stop], y_pred[:, start:stop]
            else:  # Contribution of chemical compositions
                # If the mineral is not present, we put there some values due to normalisation.
                # These are artificial and should not enter the MSE.
                z_true_clean, z_pred_clean = delete_wtrue_zero_samples(y_true[:, start:stop], y_pred[:, start:stop],
                                                                       y_true[:, i])

                y_true_clean = K.concatenate((y_true_clean, z_true_clean), axis=-1)
                y_pred_clean = K.concatenate((y_pred_clean, z_pred_clean), axis=-1)
    else:
        y_true_clean, y_pred_clean = y_true, y_pred

    if all_to_one:
        y_true_clean = K.reshape(y_true_clean, (tf.size(y_true_clean), 1))
        y_pred_clean = K.reshape(y_pred_clean, (tf.size(y_pred_clean), 1))

    return y_true_clean * 100., y_pred_clean * 100.

def my_softmax(used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None
               ) -> Callable[[tf.Tensor], tf.Tensor]:
    # if used_minerals is None: used_minerals = minerals_used # Not found in GitHub
    # if used_endmembers is None: used_endmembers = endmembers_used # Not found in GitHub
    if used_minerals is None: print(f'used_minerals is None. Check NN_losses_metrics_activations.py')
    if used_endmembers is None: print(f'used_endmembers is None. Check NN_losses_metrics_activations.py')

    @tf.function
    def softmax_norm(x: tf.Tensor) -> tf.Tensor:
        x_new = K.zeros_like(x[:, 0:0])

        for start, stop in gimme_indices(used_minerals, used_endmembers):
            tmp = softmax(x[..., start:stop])
            x_new = K.concatenate([x_new, tmp], axis=-1)

        return x_new

    return softmax_norm

def my_sigmoid(used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None
               ) -> Callable[[tf.Tensor], tf.Tensor]:
    # if used_minerals is None: used_minerals = minerals_used # Not found in GitHub
    # if used_endmembers is None: used_endmembers = endmembers_used # Not found in GitHub
    if used_minerals is None: print(f'used_minerals is None. Check NN_losses_metrics_activations.py')
    if used_endmembers is None: print(f'used_endmembers is None. Check NN_losses_metrics_activations.py')

    @tf.function
    def sigmoid_norm(x: tf.Tensor) -> tf.Tensor:
        x_new = K.zeros_like(x[:, 0:0])

        for start, stop in gimme_indices(used_minerals, used_endmembers):
            # tmp = K.clip(sigmoid(x[..., start:stop]), K.epsilon(), None)  # avoid zero sum
            tmp = sigmoid(x[..., start:stop])
            tmp /= K.clip(K.sum(tmp, axis=-1, keepdims=True), K.epsilon(), None)  # normalisation to unit sum

            x_new = K.concatenate([x_new, tmp], axis=-1)

        return x_new

    return sigmoid_norm

def my_relu(used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None
            ) -> Callable[[tf.Tensor], tf.Tensor]:
    # if used_minerals is None: used_minerals = minerals_used # Not found in GitHub
    # if used_endmembers is None: used_endmembers = endmembers_used # Not found in GitHub
    if used_minerals is None: print(f'used_minerals is None. Check NN_losses_metrics_activations.py')
    if used_endmembers is None: print(f'used_endmembers is None. Check NN_losses_metrics_activations.py')

    @tf.function
    def relu_norm(x: tf.Tensor) -> tf.Tensor:
        x_new = K.zeros_like(x[:, 0:0])

        for start, stop in gimme_indices(used_minerals, used_endmembers):
            tmp = K.clip(relu(x[..., start:stop]), K.epsilon(), None)  # avoid zero sum
            tmp /= K.clip(K.sum(tmp, axis=-1, keepdims=True), K.epsilon(), None)  # normalisation to unit sum

            x_new = K.concatenate([x_new, tmp], axis=-1)

        return x_new

    return relu_norm

def my_plu(alpha: float = 0.1, c: float = 1.0, used_minerals: np.ndarray | None = None,
           used_endmembers: list[list[bool]] | None = None) -> Callable[[tf.Tensor], tf.Tensor]:
    # if used_minerals is None: used_minerals = minerals_used # Not found in GitHub
    # if used_endmembers is None: used_endmembers = endmembers_used # Not found in GitHub
    if used_minerals is None: print(f'used_minerals is None. Check NN_losses_metrics_activations.py')
    if used_endmembers is None: print(f'used_endmembers is None. Check NN_losses_metrics_activations.py')

    @tf.function
    def plu_norm(x: tf.Tensor) -> tf.Tensor:
        # https://arxiv.org/pdf/1809.09534.pdf
        # This function does not return non-negative numbers. You should not use it for composition models.

        x_new = K.zeros_like(x[:, 0:0])

        for start, stop in gimme_indices(used_minerals, used_endmembers):
            tmp = relu(x[..., start:stop] + c) - c - (1. - alpha) * relu(x[..., start:stop] - c) - alpha * relu(
                -x[..., start:stop] - c)
            norm = K.sum(tmp, axis=-1, keepdims=True)
            # clip all numbers that are close to zero to signed K.epsilon()
            tmp /= tf.where(K.abs(norm) > K.epsilon(), norm, K.sign(norm) * K.epsilon())  # normalisation to unit sum

            x_new = K.concatenate([x_new, tmp], axis=-1)

        return x_new

    return plu_norm

def my_ae(used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None,
          cleaning: bool = True, all_to_one: bool = False) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    # if used_minerals is None: used_minerals = minerals_used # Not found in GitHub
    # if used_endmembers is None: used_endmembers = endmembers_used # Not found in GitHub
    if used_minerals is None: print(f'used_minerals is None. Check NN_losses_metrics_activations.py')
    if used_endmembers is None: print(f'used_endmembers is None. Check NN_losses_metrics_activations.py')


    def ae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        yt, yp = clean_ytrue_ypred(y_true, y_pred, used_minerals, used_endmembers, cleaning, all_to_one)
        return K.abs(yt - yp)

    return ae


def my_mae(used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None,
           cleaning: bool = True, all_to_one: bool = False) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    # if used_minerals is None: used_minerals = minerals_used # Not found in GitHub
    # if used_endmembers is None: used_endmembers = endmembers_used # Not found in GitHub
    if used_minerals is None: print(f'used_minerals is None. Check NN_losses_metrics_activations.py')
    if used_endmembers is None: print(f'used_endmembers is None. Check NN_losses_metrics_activations.py')

    def mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        abs_error = my_ae(used_minerals, used_endmembers, cleaning, all_to_one)(y_true, y_pred)
        return tnp.nanmean(abs_error, axis=0)

    return mae

def my_mse(used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None,
           cleaning: bool = True, all_to_one: bool = False) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    # if used_minerals is None: used_minerals = minerals_used # Not found in GitHub
    # if used_endmembers is None: used_endmembers = endmembers_used # Not found in GitHub
    if used_minerals is None: print(f'used_minerals is None. Check NN_losses_metrics_activations.py')
    if used_endmembers is None: print(f'used_endmembers is None. Check NN_losses_metrics_activations.py')

    def mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        yt, yp = clean_ytrue_ypred(y_true, y_pred, used_minerals, used_endmembers, cleaning, all_to_one)
        return tnp.nanmean(K.square(yt - yp), axis=0)

    return mse

def my_sse(used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None,
           cleaning: bool = True, all_to_one: bool = False) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    # if used_minerals is None: used_minerals = minerals_used # Not found in GitHub
    # if used_endmembers is None: used_endmembers = endmembers_used # Not found in GitHub
    if used_minerals is None: print(f'used_minerals is None. Check NN_losses_metrics_activations.py')
    if used_endmembers is None: print(f'used_endmembers is None. Check NN_losses_metrics_activations.py')

    def sse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        yt, yp = clean_ytrue_ypred(y_true, y_pred, used_minerals, used_endmembers, cleaning, all_to_one)
        return tnp.nansum(K.square(yt - yp), axis=0)

    return sse

def my_rmse(used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None,
            cleaning: bool = True, all_to_one: bool = False) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    # if used_minerals is None: used_minerals = minerals_used # Not found in GitHub
    # if used_endmembers is None: used_endmembers = endmembers_used # Not found in GitHub
    if used_minerals is None: print(f'used_minerals is None. Check NN_losses_metrics_activations.py')
    if used_endmembers is None: print(f'used_endmembers is None. Check NN_losses_metrics_activations.py')

    def rmse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return K.sqrt(my_mse(used_minerals, used_endmembers, cleaning, all_to_one)(y_true, y_pred))

    return rmse

def my_Lp_norm(p_coef: float, used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None,
               cleaning: bool = True, all_to_one: bool = False) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    if p_coef < 1.:
        raise ValueError("p_coef >= 1 in Lp_norm.")

     # if used_minerals is None: used_minerals = minerals_used # Not found in GitHub
    # if used_endmembers is None: used_endmembers = endmembers_used # Not found in GitHub
    if used_minerals is None: print(f'used_minerals is None. Check NN_losses_metrics_activations.py')
    if used_endmembers is None: print(f'used_endmembers is None. Check NN_losses_metrics_activations.py')

    def Lp_norm(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        abs_error = my_ae(used_minerals, used_endmembers, cleaning, all_to_one)(y_true, y_pred)
        return K.pow(tnp.nansum(K.pow(abs_error, p_coef), axis=0), 1. / p_coef)

    return Lp_norm


def my_quantile(percentile: np.ndarray | float, used_minerals: np.ndarray | None = None,
                used_endmembers: list[list[bool]] | None = None, cleaning: bool = True,
                all_to_one: bool = False) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    if not np.all(np.logical_and(percentile >= 0., percentile <= 100.)):
        raise ValueError("Percentile must be in the range [0, 100].")

    # if used_minerals is None: used_minerals = minerals_used # Not found in GitHub
    # if used_endmembers is None: used_endmembers = endmembers_used # Not found in GitHub
    if used_minerals is None: print(f'used_minerals is None. Check NN_losses_metrics_activations.py')
    if used_endmembers is None: print(f'used_endmembers is None. Check NN_losses_metrics_activations.py')

    def quantile(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        abs_error = my_ae(used_minerals, used_endmembers, cleaning, all_to_one)(y_true, y_pred)

        return tf.numpy_function(lambda error, perc:
                                 K.cast(np.nanpercentile(error, perc, method="median_unbiased", axis=0), dtype=_wp),
                                 inp=[abs_error, percentile], Tout=_wp)

    return quantile


def my_r2(used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None,
          cleaning: bool = True, all_to_one: bool = False) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
     # if used_minerals is None: used_minerals = minerals_used # Not found in GitHub
    # if used_endmembers is None: used_endmembers = endmembers_used # Not found in GitHub
    if used_minerals is None: print(f'used_minerals is None. Check NN_losses_metrics_activations.py')
    if used_endmembers is None: print(f'used_endmembers is None. Check NN_losses_metrics_activations.py')

    def r2(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        yt, yp = clean_ytrue_ypred(y_true, y_pred, used_minerals, used_endmembers, cleaning, all_to_one)

        SS_res = tnp.nansum(K.square(yt - yp), axis=0)
        SS_tot = tnp.nansum(K.square(yt - tnp.nanmean(yt, axis=0)), axis=0)

        SS_tot = K.clip(SS_tot, K.epsilon(), None)

        return 1.0 - SS_res / SS_tot

    return r2


def my_sam(used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None,
           cleaning: bool = True, all_to_one: bool = False) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
     # if used_minerals is None: used_minerals = minerals_used # Not found in GitHub
    # if used_endmembers is None: used_endmembers = endmembers_used # Not found in GitHub
    if used_minerals is None: print(f'used_minerals is None. Check NN_losses_metrics_activations.py')
    if used_endmembers is None: print(f'used_endmembers is None. Check NN_losses_metrics_activations.py')

    def sam(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        yt, yp = clean_ytrue_ypred(y_true, y_pred, used_minerals, used_endmembers, cleaning, all_to_one)

        s1_s2_norm = K.sqrt(tnp.nansum(K.square(yt), axis=0)) * K.sqrt(tnp.nansum(K.square(yp), axis=0))
        sum_s1_s2 = tnp.nansum(yt * yp, axis=0)

        s1_s2_norm = K.clip(s1_s2_norm, K.epsilon(), None)

        return tf.math.acos(sum_s1_s2 / s1_s2_norm)

    return sam


def my_f1_score(all_to_one: bool = False) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    def f1_score(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        average = "micro" if all_to_one else None

        return tf.numpy_function(lambda true, pred:
                                 K.cast(np.reshape(f1_sklearn(K.argmax(true), K.argmax(pred),
                                                              average=average),(-1,)), dtype=_wp),
                                 inp=[y_true, y_pred], Tout=_wp)

    return f1_score


def create_custom_objects(used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None,
                          alpha: float | None = 1., use_weights: bool | None = True,
                          p_coef: float = 1.5, percentile: float = 50.,
                          cleaning: bool = True, all_to_one: bool = True) -> dict:
    # if used_minerals is None: used_minerals = minerals_used # Not found in GitHub
    # if used_endmembers is None: used_endmembers = endmembers_used # Not found in GitHub
    if used_minerals is None: print(f'used_minerals is None. Check NN_losses_metrics_activations.py')
    if used_endmembers is None: print(f'used_endmembers is None. Check NN_losses_metrics_activations.py')
    if alpha is None: alpha = 1.
    if use_weights is None: use_weights = True

    # losses
    loss_composition = gimme_composition_loss(used_minerals=used_minerals, used_endmembers=used_endmembers, alpha=alpha)
    loss_taxonomy = gimme_taxonomy_loss(use_weights=use_weights)

    loss_composition_name = loss_composition.__name__
    loss_taxonomy_name = loss_taxonomy.__name__

    # activations
    my_softmax_norm = my_softmax(used_minerals=used_minerals, used_endmembers=used_endmembers)
    my_sigmoid_norm = my_sigmoid(used_minerals=used_minerals, used_endmembers=used_endmembers)
    my_relu_norm = my_relu(used_minerals=used_minerals, used_endmembers=used_endmembers)
    my_plu_norm = my_plu(used_minerals=used_minerals, used_endmembers=used_endmembers)

    my_softmax_name, my_sigmoid_name = my_softmax_norm.__name__, my_sigmoid_norm.__name__
    my_relu_name, my_plu_name = my_relu_norm.__name__, my_plu_norm.__name__

    # metrics
    mae = my_mae(used_minerals=used_minerals, used_endmembers=used_endmembers,
                 cleaning=cleaning, all_to_one=all_to_one)
    mse = my_mse(used_minerals=used_minerals, used_endmembers=used_endmembers,
                 cleaning=cleaning, all_to_one=all_to_one)
    sse = my_sse(used_minerals=used_minerals, used_endmembers=used_endmembers,
                 cleaning=cleaning, all_to_one=all_to_one)
    rmse = my_rmse(used_minerals=used_minerals, used_endmembers=used_endmembers,
                   cleaning=cleaning, all_to_one=all_to_one)
    Lp_norm = my_Lp_norm(p_coef=p_coef, used_minerals=used_minerals, used_endmembers=used_endmembers,
                         cleaning=cleaning, all_to_one=all_to_one)
    quantile = my_quantile(percentile=percentile, used_minerals=used_minerals, used_endmembers=used_endmembers,
                           cleaning=cleaning, all_to_one=all_to_one)
    r2 = my_r2(used_minerals=used_minerals, used_endmembers=used_endmembers,
               cleaning=cleaning, all_to_one=all_to_one)
    sam = my_sam(used_minerals=used_minerals, used_endmembers=used_endmembers,
                 cleaning=cleaning, all_to_one=all_to_one)
    f1_score = my_f1_score(all_to_one=all_to_one)

    mae_name, sse_name, mse_name, rmse_name = mae.__name__, sse.__name__, mse.__name__, rmse.__name__
    Lp_norm_name, quantile_name, r2_name, sam_name = Lp_norm.__name__, quantile.__name__, r2.__name__, sam.__name__
    f1_name = f1_score.__name__

    custom_objects = {loss_composition_name: loss_composition,
                      loss_taxonomy_name: loss_taxonomy,
                      #
                      my_softmax_name: my_softmax_norm,
                      my_sigmoid_name: my_sigmoid_norm,
                      my_relu_name: my_relu_norm,
                      my_plu_name: my_plu_norm,
                      #
                      mse_name: mse,
                      rmse_name: rmse,
                      quantile_name: quantile,
                      mae_name: mae,
                      Lp_norm_name: Lp_norm,
                      r2_name: r2,
                      sam_name: sam,
                      sse_name: sse,
                      f1_name: f1_score,
                      # back compatibility
                      "loss": loss_composition,
                      "my_sigmoid": my_sigmoid_norm,
                      "my_softmax": my_softmax_norm,
                      "my_plu": my_plu_norm,
                      "my_relu": my_relu_norm,
                      }

    return custom_objects