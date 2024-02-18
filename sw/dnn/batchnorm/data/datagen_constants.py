from enum import Enum

# base
impl_opt_level_uid = "impl_opt_level"
ifmap_uid = "ifmap"
current_mean_uid = "current_mean"
current_var_uid = "current_var"
running_mean_uid = "running_mean"
running_var_uid = "running_var"
weight_uid = "weight"
bias_uid = "bias"

# backward eval input
# backward training input
grad_ofmap_uid = "grad_ofmap"

# forward eval input
beta_uid = "beta"
gamma_uid = "gamma"

# forward eval output
ofmap_uid = "ofmap"

# backward eval output
grad_ifmap_uid = "grad_ifmap"
grad_weight_uid = "grad_weight"
grad_bias_uid = "grad_bias"

# backward training output
grad_ifmap_training_uid = "grad_ifmap_training"
grad_weight_training_uid = "grad_weight_training"
grad_bias_training_uid = "grad_bias_training"


class BatchNormMode(Enum):
    FORWARD_EVAL = 0
    FORWARD_TRAINING = 1
    BACKWARD_EVAL = 2
    BACKWARD_TRAINING = 3


struct_decls = {
    BatchNormMode.FORWARD_EVAL: ("batchnorm_layer_t", "forward_eval_layer"),
    BatchNormMode.FORWARD_TRAINING: (
        "batchnorm_training_layer_t",
        "forward_training_layer",
    ),
    BatchNormMode.BACKWARD_EVAL: ("batchnorm_backward_layer_t", "backward_eval_layer"),
    BatchNormMode.BACKWARD_TRAINING: (
        "batchnorm_backward_training_layer_t",
        "backward_training_layer",
    ),
}
