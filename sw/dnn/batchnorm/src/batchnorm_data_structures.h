#pragma once
#include <stdint.h>
#include "dnn.h"
typedef struct {
    uint32_t CI;
    uint32_t IH;
    uint32_t IW;
    uint32_t TILE_CI;
    double *ifmap;
    double *ofmap;

    double *running_mean;
    double *running_var;
    double *weight;
    double *bias;

    float eps;
    precision_t dtype;
} batchnorm_layer_t;

typedef struct {
    uint32_t CI;
    uint32_t IH;
    uint32_t IW;

    const double *ifmap;

    double *ofmap;

    double *running_mean;
    double *running_var;

    double *weight;
    double *bias;

    float eps;
    float momentum;
    precision_t dtype;
} batchnorm_training_layer_t;

typedef struct {
    uint32_t CI;
    uint32_t IH;
    uint32_t IW;

    const double *ifmap;
    const double *grad_ofmap;
    const double *running_mean;
    const double *running_var;
    const double *weight;

    double *grad_ifmap;
    double *grad_weight;
    double *grad_bias;

    float eps;
    precision_t dtype;
} batchnorm_backward_layer_t;

typedef struct {
    uint32_t num_points_work_in_tile;  // distinct from tile size
    union {
        uint32_t work_mod_unroll;  // Generic name for dma
        uint32_t work_mod_1;
        uint32_t work_mod_2;
        uint32_t work_mod_3;
        uint32_t work_mod_4;
    };
    union {
        uint32_t work_div_unroll_sub_1;  // Generic name for dma
        uint32_t work_div_1_sub_1;
        uint32_t work_div_2_sub_1;
        uint32_t work_div_3_sub_1;
        uint32_t work_div_4_sub_1;
    };
} __attribute__((aligned(sizeof(double)))) dm_comm_t;

typedef struct {
    uint32_t CI;
    uint32_t IH;
    uint32_t IW;
    // uint32_t TILE_CI;

    double const *ifmap;
    double const *grad_ofmap;
    double const *current_mean;
    double const *current_var;
    double const *weight;

    double *grad_ifmap;
    double *grad_weight;
    double *grad_bias;

    float eps;
    precision_t dtype;
} batchnorm_backward_training_layer_t;
