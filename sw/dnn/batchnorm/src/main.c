// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "batchnorm_impl_params.h"
#include "dnn.h"

#include "data.h"
int main() {
    if (is_forward && is_training) {
        switch (forward_training_layer.dtype) {
            case FP64:
                switch (impl_opt_level) {
                    case SINGLE_CORE_OPT:
                        batchnorm_forward_training_single_core_opt_fp64(
                            &forward_training_layer);
                        break;
                    case MULTICORE_OPT:
                        batchnorm_forward_training_multicore_fp64(&forward_training_layer, temp);
                        break;
                    default:
                        return 1;
                }
                break;
            default:
                return 1;
        }
    } else if (is_forward && !is_training) {
        switch (forward_eval_layer.dtype) {
            case FP64:
                switch (impl_opt_level) {
                    case SINGLE_CORE_OPT:
                        batchnorm_forward_single_core_opt_fp64(
                            &forward_eval_layer);
                        break;
                    case MULTICORE_OPT:
                        batchnorm_forward_multicore_fp64(&forward_eval_layer);
                        break;
                    default:
                        return 1;
                }
                break;
            default:
                return 1;
        }
    } else if (!is_forward && is_training) {
        switch (backward_training_layer.dtype) {
            case FP64:
                switch (impl_opt_level) {
                    case SINGLE_CORE:
                        batchnorm_backward_training_single_core(
                            &backward_training_layer);
                        break;
                    case SINGLE_CORE_OPT:
                        batchnorm_backward_training_single_core_opt_fp64(
                            &backward_training_layer);
                        break;
                    case MULTICORE_OPT:
                        batchnorm_backward_training_multicore_fp64(&backward_training_layer);
                        break;
                    default:
                        return 1;
                }
                break;
            case FP32:
                switch (impl_opt_level) {
                    case SINGLE_CORE_OPT:
                        batchnorm_backward_training_single_core_opt_fp32(
                            &backward_training_layer);
                        break;
                    case MULTICORE_OPT:
                        batchnorm_backward_training_multicore_fp32(
                            &backward_training_layer);
                        break;
                    default:
                        return 1;
                }
                break;
            default:
                return 1;
        }
    } else {
        switch (backward_eval_layer.dtype) {
            case FP64:
                switch (impl_opt_level) {
                    case SINGLE_CORE:
                        batchnorm_backward_single_core(&backward_eval_layer);
                        break;
                    case SINGLE_CORE_OPT:
                        batchnorm_backward_single_core_opt_fp64(
                            &backward_eval_layer);
                        break;
                    case MULTICORE_OPT:
                        batchnorm_backward_multicore_fp64(&backward_eval_layer);
                        break;
                    default:
                        return 1;
                }
                break;
            case FP32:
                switch (impl_opt_level) {
                    case SINGLE_CORE_OPT:
                        batchnorm_backward_single_core_opt_fp32(
                            &backward_eval_layer, temp);
                        break;
                    case MULTICORE_OPT:
                        batchnorm_backward_multicore_fp32(
                            &backward_eval_layer);
                        break;
                    default:
                        return 1;
                }
                break;
            case FP16:
                switch (impl_opt_level) {
                    case SINGLE_CORE_OPT:
                        batchnorm_backward_single_core_opt_fp16(
                            &backward_eval_layer, temp);
                        break;
                    default:
                        return 1;
                }
                break;
            default:
                return 1;
        }
    }

    snrt_global_barrier();

    return 0;
}
