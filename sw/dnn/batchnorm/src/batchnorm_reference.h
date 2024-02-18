#pragma once

#include <math.h>
#include "batchnorm_data_structures.h"
#include "batchnorm_utils.h"
#include "snrt.h"

static inline void batchnorm_forward_single_core_opt_fp64(
    batchnorm_layer_t *l) {
    uint32_t start = snrt_mcycle();
    // data is in NHWC format
    const uint32_t num_clusters =
        snrt_cluster_num();  // how many clusters are there in total?
    const uint32_t cluster_id = snrt_cluster_idx();  // which cluster are we?
    const uint32_t num_compute_cores =
        snrt_cluster_compute_core_num();  // how many compute cores per cluster?
    const uint32_t compute_id =
        snrt_cluster_core_idx();  // which core are we in this cluster

    // Calculate output dimensions

    // thought: this is so much contention
    const uint32_t N = 1;
    const uint32_t H = l->IH;
    const uint32_t W = l->IW;
    const uint32_t C = l->CI;
    uint32_t num_points = N * H * W;

    // dataflow:
    double *ptr = (double *)snrt_l1_start_addr();
    double *gamma_scratch = ptr;
    double *running_var_scratch = gamma_scratch;
    ptr += C;
    double *beta_scratch = ptr;
    double *running_mean_scratch = beta_scratch;
    ptr += C;
    double *weight_scratch = ptr;
    ptr += C;
    double *bias_scratch = ptr;
    ptr += C;

    // Dynamically compute tile sizes
    double *used_tcdm_end_addr =
        (double *)(snrt_l1_end_addr() -
                   ((1 << SNRT_LOG2_STACK_SIZE) + 8) *
                       (snrt_cluster_core_num() + 1));  // + 1 for safety
    ptrdiff_t space_left = used_tcdm_end_addr - ptr;  // 64 for shifting buffer
    // first 2: ofmap (overliad ish with ifmap)
    // second 2: double buffer
    // C: there are C channels per point
    ptrdiff_t tile_size_in_points = (space_left) / (2 * 2 * C);

    ptrdiff_t ofmap_len = tile_size_in_points * C * 2, ifmap_len = ofmap_len;
    // want to ensure tile stride (in doubles)
    ptrdiff_t tile_stride_in_doubles = tile_size_in_points * C;

    double *ofmap_scratch = ptr;
    ptr += tile_stride_in_doubles * 2;
    double *ifmap_scratch = ofmap_scratch;

    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT0,
                                     SNRT_PERF_CNT_ICACHE_STALL);
    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT1,
                                     SNRT_PERF_CNT_TCDM_CONGESTED);
    snrt_dma_txid_t load_running_mean, load_running_var, load_weight, load_bias;
    uint32_t start_dma_load = snrt_mcycle();
    // load running_var, initiate the rest
    if (snrt_is_dm_core()) {
        load_running_mean = snrt_dma_start_1d(
            running_mean_scratch, l->running_mean, C * sizeof(double));
        load_running_var = snrt_dma_start_1d(
            running_var_scratch, l->running_var, C * sizeof(double));
        load_weight =
            snrt_dma_start_1d(weight_scratch, l->weight, C * sizeof(double));
        load_bias =
            snrt_dma_start_1d(bias_scratch, l->bias, C * sizeof(double));
        snrt_dma_wait_all();
        snrt_dma_start_1d(ifmap_scratch, l->ifmap,
                          num_points * C * sizeof(double));
    } else {
        // first ssr: read from running_var, then running_mean
        // second ssr: write to gamma, then to beta
        // third ssr: read from weight, then bias
        snrt_ssr_loop_2d(SNRT_SSR_DM_ALL, 2, C, C * sizeof(double),
                         sizeof(double));
    }
    uint32_t end_dma_load = SNRT_SECTIONED_MCYCLE();
    // this is necessary to ensure that dm_comm has been written to
    snrt_cluster_hw_barrier();

    // compute invstd, load weight and running_mean in
    uint32_t start_gamma_beta_calc = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
    } else if (compute_id == 0) {
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, running_var_scratch);
        snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, gamma_scratch);
        snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, weight_scratch);
        register double eps = l->eps;
        snrt_ssr_enable();
        // Reduce the four variables running_mean, running_var, weight, bias
        // into an fmadd for main loop see
        // https://github.com/pytorch/pytorch/blob/3acaf8564da4c2f0faaea33ce4572ad9b3715b47/aten/src/ATen/native/cpu/batch_norm_kernel.cpp#L45-L55
        asm volatile(
            "frep.o %[n_frep], 5, 0, 0 \n"
            "fadd.d ft3, ft0, %[eps]\n"
            "fsqrt.d ft3, ft3\n"
            "fdiv.d ft3, ft2, ft3\n"
            "fsgnj.d ft1, ft3, ft3\n"
            "fnmsub.d ft1, ft0, ft3, ft2\n"
            :
            : [eps] "fr"(eps),
              [n_frep] "r"(C - 1)  // we repeat n_frep+1 times
            : "ft0", "ft1", "ft2", "ft3");

        __builtin_ssr_barrier(SNRT_SSR_DM1);
        snrt_ssr_disable();
    }
    uint32_t end_gamma_beta_calc = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    uint32_t start_main_loop = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        // finish loads
        snrt_dma_wait_all();

        // notify ready
        snrt_cluster_hw_barrier();
        // wait for compute to be done
        snrt_cluster_hw_barrier();
        snrt_dma_start_1d(l->ofmap, ofmap_scratch,
                          num_points * C * sizeof(double));
    } else {
        if (compute_id == 0) {
            batchnorm_forward_fp64_no_loop(
                ifmap_scratch, ofmap_scratch, gamma_scratch, beta_scratch,
                C * sizeof(double), num_points, num_points - 1, C, 1);
        } else {
            snrt_cluster_hw_barrier();
        }
        snrt_cluster_hw_barrier();
    }

    uint32_t end_main_loop = SNRT_SECTIONED_MCYCLE();

    // wait for all transactions to complete
    uint32_t start_dma_writeback = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_wait_all();
    } else {
    }
    snrt_cluster_hw_barrier();
    uint32_t done = snrt_mcycle();
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT0);
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT1);
}

static inline void batchnorm_forward_training_single_core_opt_fp64(
    batchnorm_training_layer_t *l) {
    uint32_t start = snrt_mcycle();
    // data is in NHWC format
    const uint32_t num_clusters =
        snrt_cluster_num();  // how many clusters are there in total?
    const uint32_t cluster_id = snrt_cluster_idx();  // which cluster are we?
    const uint32_t num_compute_cores =
        snrt_cluster_compute_core_num();  // how many compute cores per cluster?
    const uint32_t compute_id =
        snrt_cluster_core_idx();  // which core are we in this cluster

    // Calculate output dimensions

    // thought: this is so much contention
    const uint32_t N = 1;
    const uint32_t H = l->IH;
    const uint32_t W = l->IW;
    const uint32_t C = l->CI;
    uint32_t num_points = N * H * W;
    const double momentum = l->momentum;
    const double one_sub_momentum = 1 - momentum;
    const double momentum_times_unbiased_correction =
        momentum * ((double)num_points / (double)(num_points - 1));

    // dataflow:
    void *raw_ptr = (void *)snrt_l1_start_addr();
    dm_comm_t *dm_comm = (dm_comm_t *)raw_ptr;
    raw_ptr += sizeof(dm_comm_t);
    double *ptr = (double *)raw_ptr;
    // want to compute
    // running_var = running_var * (momentum) + current_var * (1-momentum)
    // running_mean = running_mean * (momentum) + current_mean * (1-momentum)
    // gamma = weight / sqrt(current_var + eps)
    // beta = bias - running_mean * gamma
    double *current_var_scratch = ptr;
    ptr += C;
    double *current_mean_scratch = ptr;
    ptr += C;
    double *running_var_scratch = ptr;
    ptr += C;
    double *running_mean_scratch = ptr;
    ptr += C;
    double *gamma_scratch = ptr;
    double *weight_scratch = ptr;
    ptr += C;
    double *beta_scratch = ptr;
    double *bias_scratch = ptr;
    ptr += C;

    // Dynamically compute tile sizes
    double *used_tcdm_end_addr =
        (double *)(snrt_l1_end_addr() -
                   ((1 << SNRT_LOG2_STACK_SIZE) + 8) *
                       (snrt_cluster_core_num() + 1));  // + 1 for safety
    ptrdiff_t space_left = used_tcdm_end_addr - ptr;  // 64 for shifting buffer
    // first 2: ofmap (overlaid ish with ifmap)
    // second 2: double buffer
    // C: there are C channels per point
    ptrdiff_t tile_size_in_points = (space_left) / (2 * C);

    ptrdiff_t ofmap_len = tile_size_in_points * C * 2, ifmap_len = ofmap_len;
    // want to ensure tile stride (in doubles)
    ptrdiff_t tile_stride_in_doubles = tile_size_in_points * C;

    double *ofmap_scratch = ptr;
    double *ifmap_scratch = ofmap_scratch;

    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT0,
                                     SNRT_PERF_CNT_ICACHE_STALL);
    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT1,
                                     SNRT_PERF_CNT_TCDM_CONGESTED);
    snrt_dma_txid_t load_running_mean, load_running_var, load_weight, load_bias,
        load_ifmap;
    uint32_t start_dma_load = snrt_mcycle();
    if (snrt_is_dm_core()) {
        load_ifmap = snrt_dma_start_1d(ifmap_scratch, l->ifmap,
                                       num_points * C * sizeof(double));
        load_running_mean = snrt_dma_start_1d(
            running_mean_scratch, l->running_mean, C * sizeof(double));
        load_running_var = snrt_dma_start_1d(
            running_var_scratch, l->running_var, C * sizeof(double));
        load_weight =
            snrt_dma_start_1d(weight_scratch, l->weight, C * sizeof(double));
        load_bias =
            snrt_dma_start_1d(bias_scratch, l->bias, C * sizeof(double));
    }
    uint32_t end_dma_load = SNRT_SECTIONED_MCYCLE();
    uint32_t start_statistic = SNRT_SECTIONED_MCYCLE();
    // no looping needed
    if (snrt_is_dm_core()) {
        // finish loads
        snrt_dma_wait(load_ifmap);

        // notify ready
        snrt_cluster_hw_barrier();
        // wait for compute to be done
        snrt_cluster_hw_barrier();
    } else {
        if (compute_id == 0) {
            batchnorm_collect_statistics_fp64_no_loop(
                ifmap_scratch, current_mean_scratch, current_var_scratch,
                num_points, C * sizeof(double), num_points, num_points / 4 - 1,
                num_points % 4, C, 1);
        } else {
            // wait for dma to be ready
            snrt_cluster_hw_barrier();
        }
        // notify compute done
        snrt_cluster_hw_barrier();
    }

    // need to collect current_mean = sum(x) / N
    // need to collect current_var = sum((x-current_mean)^2) / N
    uint32_t end_statistic = SNRT_SECTIONED_MCYCLE();

    uint32_t start_gamma_beta_calc = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_wait_all();
        // notify that the rest has been loaded in
        snrt_cluster_hw_barrier();
    } else {
        snrt_cluster_hw_barrier();
        if (compute_id == 0) {
            snrt_ssr_loop_2d(SNRT_SSR_DM1, 4, C, C * sizeof(double),
                             sizeof(double));
            snrt_ssr_loop_2d(SNRT_SSR_DM2, 4, C, C * sizeof(double),
                             sizeof(double));
            snrt_ssr_loop_2d(SNRT_SSR_DM0, 2, C, C * sizeof(double),
                             sizeof(double));
            snrt_ssr_repeat(SNRT_SSR_DM0, 2);
            // consume current var (twice) then current mean (twice)
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, current_var_scratch);
            // write to running_var, running_mean, gamma, beta
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, running_var_scratch);
            // consume running_var, running_mean, weight, bias
            snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, running_var_scratch);
            register double eps = l->eps;
            snrt_ssr_enable();
            // Reduce the four variables running_mean, running_var, weight, bias
            // into an fmadd for main loop see
            // https://github.com/pytorch/pytorch/blob/3acaf8564da4c2f0faaea33ce4572ad9b3715b47/aten/src/ATen/native/cpu/batch_norm_kernel.cpp#L45-L55
            // Moreover, update statistics for running_mean and running_var.
            // Use unbiased estimator for updating statistic, but biased
            // estimator for gamma/beta
            asm volatile(
                "frep.o %[n_frep], 9, 0, 0 \n"
                // current_var + eps
                "fadd.d ft3, ft0, %[eps]\n"
                // current_var * momentum
                "fmul.d ft4, ft0, %[momentum_times_unbiased_correction]\n"
                // current_mean * momentum
                "fmul.d ft5, ft0, %[momentum]\n"
                // sqrt(current_var + eps)
                "fsqrt.d ft3, ft3\n"
                // running_var update
                "fmadd.d ft1, ft2, %[one_sub_momentum], ft4\n"
                // running_mean update
                "fmadd.d ft1, ft2, %[one_sub_momentum], ft5\n"
                // weight / sqrt()
                "fdiv.d ft3, ft2, ft3\n"
                // write gamma (weight already consumed)
                "fsgnj.d ft1, ft3, ft3\n"
                // consume bias, current_mean and write beta
                "fnmsub.d ft1, ft0, ft3, ft2\n"
                :
                : [eps] "fr"(eps), [one_sub_momentum] "fr"(one_sub_momentum),
                  [momentum] "fr"(momentum),
                  [momentum_times_unbiased_correction] "fr"(
                      momentum_times_unbiased_correction),
                  [n_frep] "r"(C - 1)  // we repeat n_frep+1 times
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5");

            snrt_fpu_fence();
            snrt_ssr_repeat(SNRT_SSR_DM0, 1);
            __builtin_ssr_barrier(SNRT_SSR_DM1);
            snrt_ssr_disable();
        }
    }
    // calc gamma beta, dma loads in first tile for ofmap
    uint32_t end_gamma_beta_calc = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();
    uint32_t start_ofmap_calc = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        // finish loads
        snrt_dma_wait_all();

        // notify ready
        snrt_cluster_hw_barrier();
        // wait for compute to be done
        snrt_cluster_hw_barrier();
        snrt_dma_start_1d(l->ofmap, ofmap_scratch,
                          num_points * C * sizeof(double));
    } else {
        if (compute_id == 0) {
            batchnorm_forward_fp64_no_loop(
                ifmap_scratch, ofmap_scratch, gamma_scratch, beta_scratch,
                C * sizeof(double), num_points, num_points - 1, C, 1);
        } else {
            snrt_cluster_hw_barrier();
        }
        snrt_cluster_hw_barrier();
    }

    uint32_t end_ofmap_calc = SNRT_SECTIONED_MCYCLE();

    // collect stats

    // update running mean and running var
    // pass << batch mean and batch var >> as parameters to compute beta/gamma
    // call batchnorm_layer

    // wait for all transactions to complete
    uint32_t start_dma_writeback = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->running_var, running_var_scratch,
                          C * sizeof(double));
        snrt_dma_start_1d(l->running_mean, running_mean_scratch,
                          C * sizeof(double));
        snrt_dma_wait_all();
    } else {
    }
    snrt_cluster_hw_barrier();
    uint32_t done = snrt_mcycle();
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT0);
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT1);
}

// No DMA, SSR, or FREP. Still uses TCDM.
static inline void batchnorm_backward_single_core(
    batchnorm_backward_layer_t *l) {
    uint32_t kernel_start = snrt_mcycle();
    // data is in HWC format
    const uint32_t num_clusters =
        snrt_cluster_num();  // how many clusters are there in total? currently
                             // 1 in the config i think
    const uint32_t cluster_id = snrt_cluster_idx();  // which cluster are we?
    const uint32_t num_compute_cores =
        snrt_cluster_compute_core_num();  // how many compute cores per cluster?
    const uint32_t compute_id =
        snrt_cluster_core_idx();  // which core are we in this cluster
    // Calculate output dimensions
    uint32_t N = 1;
    uint32_t H = l->IH;
    uint32_t W = l->IW;
    uint32_t C = l->CI;
    double eps = l->eps;
    uint32_t num_points = N * H * W;

    uint32_t buffer_len = l->CI;
    uint32_t invstd_scratch_len = l->CI;
    uint32_t weight_times_invstd_len = l->CI,
             running_mean_times_invstd_len = l->CI;

    double *ptr = (double *)snrt_l1_start_addr();
    double *invstd_scratch = ptr;
    ptr += invstd_scratch_len;
    double *weight_times_invstd_scratch = ptr;
    ptr += weight_times_invstd_len;
    double *running_mean_times_invstd_scratch = ptr;
    ptr += running_mean_times_invstd_len;
    uint32_t start_dma_load = snrt_mcycle();
    uint32_t end_dma_load = SNRT_SECTIONED_MCYCLE();
    uint32_t start_invstd_computations = SNRT_SECTIONED_MCYCLE();
    if (compute_id == 0) {
        for (uint32_t channel = 0; channel < C; ++channel) {
            double invstd = 1 / sqrt(l->running_var[channel] + eps);
            invstd_scratch[channel] = invstd;
            weight_times_invstd_scratch[channel] = invstd * l->weight[channel];
            running_mean_times_invstd_scratch[channel] =
                invstd * l->running_mean[channel];
        }
    }
    uint32_t end_invstd_computations = SNRT_SECTIONED_MCYCLE();
    uint32_t start_running_var_weight_inplace_mul = SNRT_SECTIONED_MCYCLE();
    uint32_t end_running_var_weight_inplace_mul = SNRT_SECTIONED_MCYCLE();

    uint32_t start_main_loop = SNRT_SECTIONED_MCYCLE();
    if (compute_id == 0) {
        for (uint32_t i = 0; i < num_points; i += 1) {
            for (uint32_t channel = 0; channel < C; ++channel) {
                double dy = l->grad_ofmap[i * C + channel];
                double x = l->ifmap[i * C + channel];
                l->grad_bias[channel] += dy;
                l->grad_weight[channel] +=
                    dy * (x * invstd_scratch[channel] -
                          running_mean_times_invstd_scratch[channel]);
                l->grad_ifmap[i * C + channel] =
                    dy * weight_times_invstd_scratch[channel];
            }
        }
    }
    uint32_t end_main_loop = SNRT_SECTIONED_MCYCLE();
    uint32_t start_grad_bias_weight_reduction = SNRT_SECTIONED_MCYCLE();
    uint32_t end_grad_bias_weight_reduction = SNRT_SECTIONED_MCYCLE();
    uint32_t start_dma_writeback = SNRT_SECTIONED_MCYCLE();
    uint32_t end_dma_writeback = SNRT_SECTIONED_MCYCLE();
    uint32_t done = snrt_mcycle();
}

// uses DMA, SSR, FREP
static inline void batchnorm_backward_single_core_opt_fp64(
    batchnorm_backward_layer_t *l) {
    uint32_t kernel_start = snrt_mcycle();
    // data is in NHWC format
    const uint32_t num_clusters =
        snrt_cluster_num();  // how many clusters are there in total? currently
                             // 1 in the config i think
    const uint32_t cluster_id = snrt_cluster_idx();  // which cluster are we?
    const uint32_t num_compute_cores =
        snrt_cluster_compute_core_num();  // how many compute cores per cluster?
    const uint32_t compute_id =
        snrt_cluster_core_idx();  // which core are we in this cluster

    // keep the dma core and one compute core

    uint32_t N = 1;
    uint32_t H = l->IH;
    uint32_t W = l->IW;
    uint32_t C = l->CI;
    // ignore tiling for now
    uint32_t num_points = N * H * W;
    uint32_t num_doubles = num_points * C;
    uint32_t num_bytes = num_doubles * sizeof(double);
    uint32_t point_size_in_bytes = C * sizeof(double);

    ptrdiff_t grad_bias_scratch_len = C, grad_weight_scratch_len = C;

    // dataflow:
    double *ptr = (double *)snrt_l1_start_addr();
    double *invstd_scratch = ptr;
    ptr += C;
    double *weight_scratch = ptr;
    ptr += C;
    double *running_mean_scratch = ptr;
    ptr += C;
    double *grad_bias_scratch = ptr;
    ptr += grad_bias_scratch_len;
    double *grad_weight_scratch = ptr;
    ptr += grad_weight_scratch_len;
    ptrdiff_t grad_ofmap_len = num_points * C, grad_ifmap_len = grad_ofmap_len,
              ifmap_len = grad_ifmap_len;

    double *grad_ofmap_scratch = ptr;
    ptr += grad_ofmap_len;
    double *ifmap_scratch = ptr;
    ptr += ifmap_len;
    double *grad_ifmap_scratch = ifmap_scratch;  // reuse the buffer

    snrt_dma_txid_t running_var_load, weight_load, running_mean_load,
        grad_ofmap_load, ifmap_load, grad_ifmap_write;

    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT0,
                                     SNRT_PERF_CNT_ICACHE_STALL);
    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT1,
                                     SNRT_PERF_CNT_TCDM_CONGESTED);
    uint32_t start_dma_load = snrt_mcycle();
    // load running_var, initiate the rest
    if (snrt_is_dm_core()) {
        weight_load =
            snrt_dma_start_1d(weight_scratch, l->weight, point_size_in_bytes);
        running_mean_load = snrt_dma_start_1d(
            running_mean_scratch, l->running_mean, point_size_in_bytes);
        running_var_load = snrt_dma_start_1d(invstd_scratch, l->running_var,
                                             C * sizeof(double));
        snrt_dma_wait_all();
    } else if (compute_id == 0) {
        // PRECONFIGURE: operations on arrays of size C, split by core.
        snrt_ssr_loop_2d(SNRT_SSR_DM_ALL, 3, C, C * sizeof(double),
                         sizeof(double));
    }
    uint32_t end_dma_load = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    // compute invstd, load weight and running_mean in
    uint32_t start_invstd_calc = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        // load first tile in. We can do this here because sqrt/div are really
        // slow.
        grad_ofmap_load =
            snrt_dma_start_1d(grad_ofmap_scratch, l->grad_ofmap, num_bytes);
        ifmap_load = snrt_dma_start_1d(ifmap_scratch, l->ifmap, num_bytes);

        snrt_dma_wait_all();
    } else if (compute_id == 0) {
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, invstd_scratch);
        snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, invstd_scratch);
        register double eps = l->eps;  // any value in dma'ing this? idk
        const register double ONE = 1;
        snrt_ssr_enable();
        // unrolling does not help because fsqrt and fdiv are not pipelined
        asm volatile(
            "frep.o %[n_frep], 6, 0, 0 \n"
            "fadd.d ft3, ft0, %[eps]\n"
            "fsqrt.d ft3, ft3\n"
            "fdiv.d ft3, %[ONE], ft3\n"
            // write back invstd
            "fsgnj.d ft1, ft3, ft3\n"
            // write back weight * invstd
            "fmul.d ft1, ft0, ft3\n"
            // write back running mean * invstd
            "fmul.d ft1, ft0, ft3\n"
            :
            : [eps] "fr"(eps), [ONE] "fr"(ONE),
              [n_frep] "r"(C - 1)  // we repeat n_frep+1 times
            : "ft0", "ft1", "ft2", "ft3");

        snrt_fpu_fence();                     // thought: do we need this?
        __builtin_ssr_barrier(SNRT_SSR_DM1);  // thought: do we need this?
        snrt_ssr_disable();
    }
    uint32_t end_invstd_calc = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    uint32_t start_running_var_weight_inplace_mul = SNRT_SECTIONED_MCYCLE();
    uint32_t end_running_var_weight_inplace_mul = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();
    uint32_t start_main_loop = SNRT_SECTIONED_MCYCLE();
    // compute grad_weight, grad_bias, grad_ifmap
    if (snrt_is_dm_core()) {
    } else if (compute_id == 0) {
        batchnorm_backward_fp64_no_loop(grad_ofmap_scratch, grad_ifmap_scratch,
                                        ifmap_scratch, running_mean_scratch,
                                        weight_scratch, invstd_scratch,
                                        grad_bias_scratch, grad_weight_scratch,
                                        C, num_points, num_points % 2, C, 1);
    }
    uint32_t end_main_loop = SNRT_SECTIONED_MCYCLE();
    // don't need second reduction
    snrt_cluster_hw_barrier();
    uint32_t start_grad_bias_weight_reduction = SNRT_SECTIONED_MCYCLE();
    uint32_t end_grad_bias_weight_reduction = SNRT_SECTIONED_MCYCLE();
    // write back grad_bias and grad_weight. then wait for all transactions to
    // complete
    uint32_t start_dma_writeback = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->grad_bias, grad_bias_scratch, C * sizeof(double));
        snrt_dma_start_1d(l->grad_weight, grad_weight_scratch,
                          C * sizeof(double));
        snrt_dma_start_1d(l->grad_ifmap, grad_ifmap_scratch, num_bytes);
        snrt_dma_wait_all();
    } else if (compute_id == 0) {
    }
    uint32_t end_dma_writeback = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();
    uint32_t done = snrt_mcycle();
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT0);
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT1);
}

// uses DMA, SSR, FREP
static inline void batchnorm_backward_single_core_opt_fp32(
    batchnorm_backward_layer_t *l, double *temp) {
    uint32_t kernel_start = snrt_mcycle();
    // data is in NHWC format
    const uint32_t num_clusters =
        snrt_cluster_num();  // how many clusters are there in total? currently
                             // 1 in the config i think
    const uint32_t cluster_id = snrt_cluster_idx();  // which cluster are we?
    const uint32_t num_compute_cores =
        snrt_cluster_compute_core_num();  // how many compute cores per cluster?
    const uint32_t compute_id =
        snrt_cluster_core_idx();  // which core are we in this cluster

    // keep the dma core and one compute core

    uint32_t N = 1;
    uint32_t H = l->IH;
    uint32_t W = l->IW;
    uint32_t C = l->CI;
    // assume C is a multiple of 2 for now
    uint32_t num_points = N * H * W;
    precision_t dtype_bytes = l->dtype;
    uint32_t num_dtypes_per_double = (FP64 / dtype_bytes);

    // including padding for non-aligned points.
    // Write input with padded stride since ssr only supports 64-bit width r/w
    uint32_t num_doubles_per_point = ceildiv(C, num_dtypes_per_double);
    uint32_t num_doubles = num_points * num_doubles_per_point;
    bool is_point_aligned_to_8_byte_boundary = C % num_dtypes_per_double == 0;
    uint32_t num_bytes_per_aligned_point =
        num_doubles_per_point * sizeof(double);
    uint32_t num_bytes_in_point_aligned_ifmap = num_doubles * sizeof(double);

    // not including padding for non-aligned points.
    // When is_point_aligned_to_8_byte_boundary == true, the expressions above
    // are equivalent. Output should be written according to the packed version
    uint32_t num_bytes_per_packed_point = C * dtype_bytes;

    ptrdiff_t grad_bias_scratch_len = num_doubles_per_point,
              grad_weight_scratch_len = num_doubles_per_point;

    // dataflow:
    v2s *ptr = (v2s *)snrt_l1_start_addr();
    v2s *invstd_scratch = ptr;
    ptr += num_doubles_per_point;
    v2s *weight_scratch = ptr;
    ptr += num_doubles_per_point;
    v2s *running_mean_scratch = ptr;
    ptr += num_doubles_per_point;
    v2s *grad_bias_scratch = ptr;
    ptr += grad_bias_scratch_len;
    v2s *grad_weight_scratch = ptr;
    ptr += grad_weight_scratch_len;

    ptrdiff_t grad_ofmap_len = num_doubles, grad_ifmap_len = grad_ofmap_len,
              ifmap_len = grad_ifmap_len;

    v2s *grad_ofmap_scratch = ptr;
    ptr += grad_ofmap_len;
    v2s *ifmap_scratch = ptr;
    ptr += ifmap_len;
    v2s *grad_ifmap_scratch = ifmap_scratch;  // reuse the buffer

    snrt_dma_txid_t running_var_load, weight_load, running_mean_load,
        grad_ofmap_load, ifmap_load, grad_ifmap_write;

    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT0,
                                     SNRT_PERF_CNT_ICACHE_STALL);
    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT1,
                                     SNRT_PERF_CNT_TCDM_CONGESTED);
    uint32_t start_dma_load = snrt_mcycle();
    // load running_var, initiate the rest
    if (snrt_is_dm_core()) {
        // Initiate loads for everything but only wait for the running var load.
        // Q: is it better to wait then initiate the rest? we'll see
        running_var_load = snrt_dma_start_1d(invstd_scratch, l->running_var,
                                             num_bytes_per_packed_point);
        weight_load = snrt_dma_start_1d(weight_scratch, l->weight,
                                        num_bytes_per_packed_point);
        running_mean_load = snrt_dma_start_1d(
            running_mean_scratch, l->running_mean, num_bytes_per_packed_point);
        snrt_dma_wait_all();
    } else if (compute_id == 0) {
        // PRECONFIGURE: operations on arrays of size C, split by core.
        snrt_ssr_loop_2d(SNRT_SSR_DM_ALL, 2, num_doubles_per_point,
                         num_bytes_per_aligned_point, sizeof(double));
    }
    uint32_t end_dma_load = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    // compute invstd, load weight and running_mean in
    uint32_t start_invstd_calc = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        // load first tile in. We can do this here because sqrt/div are really
        // slow.
        grad_ofmap_load = initiate_dma_1d_or_2d(
            grad_ofmap_scratch, l->grad_ofmap, num_bytes_per_packed_point,
            num_bytes_per_aligned_point, num_bytes_per_packed_point, num_points,
            is_point_aligned_to_8_byte_boundary);
        ifmap_load = initiate_dma_1d_or_2d(
            ifmap_scratch, l->ifmap, dtype_bytes * C,
            num_bytes_per_aligned_point, num_bytes_per_packed_point, num_points,
            is_point_aligned_to_8_byte_boundary);

        snrt_dma_wait_all();
    } else if (compute_id == 0) {
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, invstd_scratch);
        snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, invstd_scratch);
        register float eps = l->eps;
        register float ONE = 1;
        snrt_ssr_enable();
        // unrolling does not help because fsqrt and fdiv are not pipelined
        asm volatile(
            "vfcpka.s.s %[ONE],%[ONE],%[ONE]\n"  // duplicate the 1
            "frep.o %[n_frep], 5, 0, 0 \n"
            "vfadd.r.s ft3, ft0, %[eps]\n"
            "vfsqrt.s ft3, ft3\n"
            "vfdiv.s ft3, %[ONE], ft3\n"
            // write back invstd
            "vfsgnj.s ft1, ft3, ft3\n"
            // write back weight * invstd
            "vfmul.s ft1, ft0, ft3\n"
            : [ONE] "+fr"(ONE)
            : [eps] "fr"(eps),
              [n_frep] "r"(num_doubles_per_point -
                           1)  // we repeat n_frep+1 times
            : "ft0", "ft1", "ft2", "ft3");

        snrt_fpu_fence();                     // thought: do we need this?
        __builtin_ssr_barrier(SNRT_SSR_DM1);  // thought: do we need this?
        snrt_ssr_disable();
    }
    uint32_t end_invstd_calc = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    // compute weight*invstd and running_mean*invstd

    // computing invstd scratch and using it for weight: can we do it in 1 frep?
    // load running var: 1 ssr
    // write running var: 1 ssr
    // load weight: 1 ssr
    // write weight: 1 ssr
    // answer: not really. Still worth precomputing I think
    uint32_t start_running_var_weight_inplace_mul = SNRT_SECTIONED_MCYCLE();
    uint32_t end_running_var_weight_inplace_mul = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();
    uint32_t start_main_loop = SNRT_SECTIONED_MCYCLE();
    // compute grad_weight, grad_bias, grad_ifmap
    if (snrt_is_dm_core()) {
    } else if (compute_id == 0) {
        batchnorm_backward_fp32_no_loop(
            grad_ofmap_scratch, grad_ifmap_scratch, ifmap_scratch,
            running_mean_scratch, weight_scratch, invstd_scratch,
            grad_bias_scratch, grad_weight_scratch, num_bytes_per_aligned_point,
            num_points, num_points % 2, num_doubles_per_point, 1);
    }
    uint32_t end_main_loop = SNRT_SECTIONED_MCYCLE();
    // don't need second reduction
    snrt_cluster_hw_barrier();
    uint32_t start_grad_bias_weight_reduction = SNRT_SECTIONED_MCYCLE();
    uint32_t end_grad_bias_weight_reduction = SNRT_SECTIONED_MCYCLE();
    // write back grad_bias and grad_weight. then wait for all transactions to
    // complete
    uint32_t start_dma_writeback = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->grad_bias, grad_bias_scratch,
                          num_bytes_per_packed_point);
        snrt_dma_start_1d(l->grad_weight, grad_weight_scratch,
                          num_bytes_per_packed_point);
        initiate_dma_1d_or_2d(
            l->grad_ifmap, grad_ifmap_scratch, num_bytes_per_packed_point,
            num_bytes_per_packed_point, num_bytes_per_aligned_point, num_points,
            is_point_aligned_to_8_byte_boundary);
        snrt_dma_wait_all();
    } else if (compute_id == 0) {
    }
    uint32_t end_dma_writeback = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();
    uint32_t done = snrt_mcycle();
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT0);
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT1);
}

// uses DMA, SSR, FREP
static inline void batchnorm_backward_single_core_opt_fp16(
    batchnorm_backward_layer_t *l, double *temp) {
    uint32_t kernel_start = snrt_mcycle();
    // data is in NHWC format
    const uint32_t num_clusters =
        snrt_cluster_num();  // how many clusters are there in total? currently
                             // 1 in the config i think
    const uint32_t cluster_id = snrt_cluster_idx();  // which cluster are we?
    const uint32_t num_compute_cores =
        snrt_cluster_compute_core_num();  // how many compute cores per cluster?
    const uint32_t compute_id =
        snrt_cluster_core_idx();  // which core are we in this cluster

    // keep the dma core and one compute core

    uint32_t N = 1;
    uint32_t H = l->IH;
    uint32_t W = l->IW;
    uint32_t C = l->CI;
    // assume C is a multiple of 2 for now
    uint32_t num_points = N * H * W;
    precision_t dtype_bytes = l->dtype;
    uint32_t num_dtypes_per_double = (FP64 / dtype_bytes);

    // including padding for non-aligned points.
    // Write input with padded stride since ssr only supports 64-bit width r/w
    uint32_t num_doubles_per_point = ceildiv(C, num_dtypes_per_double);
    uint32_t num_doubles = num_points * num_doubles_per_point;
    bool is_point_aligned_to_8_byte_boundary = C % num_dtypes_per_double == 0;
    uint32_t num_bytes_per_aligned_point =
        num_doubles_per_point * sizeof(double);
    uint32_t num_bytes_in_point_aligned_ifmap = num_doubles * sizeof(double);

    // not including padding for non-aligned points.
    // When is_point_aligned_to_8_byte_boundary == true, the expressions above
    // are equivalent. Output should be written according to the packed version
    uint32_t num_bytes_per_packed_point = C * dtype_bytes;

    ptrdiff_t grad_bias_scratch_len = num_doubles_per_point,
              grad_weight_scratch_len = num_doubles_per_point;

    // dataflow:
    v4s *ptr = (v4s *)snrt_l1_start_addr();
    v4s *weight_scratch = ptr;
    ptr += num_doubles_per_point;
    v4s *invstd_scratch = ptr;
    ptr += num_doubles_per_point;
    v4s *running_mean_scratch = ptr;
    ptr += num_doubles_per_point;
    v4s *grad_bias_scratch = ptr;
    ptr += grad_bias_scratch_len;
    v4s *grad_weight_scratch = ptr;
    ptr += grad_weight_scratch_len;

    ptrdiff_t grad_ofmap_len = num_doubles, grad_ifmap_len = grad_ofmap_len,
              ifmap_len = grad_ifmap_len;

    v4s *grad_ofmap_scratch = ptr;
    ptr += grad_ofmap_len;
    v4s *ifmap_scratch = ptr;
    ptr += ifmap_len;
    v4s *grad_ifmap_scratch = ifmap_scratch;  // reuse the buffer

    snrt_dma_txid_t running_var_load, weight_load, running_mean_load,
        grad_ofmap_load, ifmap_load, grad_ifmap_write;

    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT0,
                                     SNRT_PERF_CNT_ICACHE_STALL);
    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT1,
                                     SNRT_PERF_CNT_TCDM_CONGESTED);
    uint32_t start_dma_load = snrt_mcycle();
    // load running_var, initiate the rest
    if (snrt_is_dm_core()) {
        // Initiate loads for everything but only wait for the running var load.
        // Q: is it better to wait then initiate the rest? we'll see
        running_var_load = snrt_dma_start_1d(invstd_scratch, l->running_var,
                                             num_bytes_per_packed_point);
        snrt_dma_wait(running_var_load);
    } else if (compute_id == 0) {
        // PRECONFIGURE: operations on arrays of size C, split by core.
        snrt_ssr_loop_1d(SNRT_SSR_DM_ALL, num_doubles_per_point,
                         sizeof(double));
    }
    uint32_t end_dma_load = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    // compute invstd, load weight and running_mean in
    uint32_t start_invstd_calc = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        weight_load = snrt_dma_start_1d(weight_scratch, l->weight,
                                        num_bytes_per_packed_point);
        running_mean_load = snrt_dma_start_1d(
            running_mean_scratch, l->running_mean, num_bytes_per_packed_point);
        // load first tile in. We can do this here because sqrt/div are really
        // slow.
        grad_ofmap_load = initiate_dma_1d_or_2d(
            grad_ofmap_scratch, l->grad_ofmap, num_bytes_per_packed_point,
            num_bytes_per_aligned_point, num_bytes_per_packed_point, num_points,
            is_point_aligned_to_8_byte_boundary);
        ifmap_load = initiate_dma_1d_or_2d(
            ifmap_scratch, l->ifmap, dtype_bytes * C,
            num_bytes_per_aligned_point, num_bytes_per_packed_point, num_points,
            is_point_aligned_to_8_byte_boundary);

        snrt_dma_wait_all();
    } else if (compute_id == 0) {
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, invstd_scratch);
        snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, invstd_scratch);
        register float eps = l->eps;
        register float ONE = 1;
        register v2s eps_half;
        register v4s ONE_VEC;
        snrt_ssr_enable();
        // unrolling does not help because fsqrt and fdiv are not pipelined
        asm volatile(
            "fcvt.h.s %[eps_half], %[eps]\n"
            "vfcpka.h.s %[ONE_VEC],%[ONE],%[ONE]\n"  // duplicate the 1
            "vfcpkb.h.s %[ONE_VEC],%[ONE],%[ONE]\n"  // duplicate the 1
            "frep.o %[n_frep], 3, 0, 0 \n"
            "vfadd.r.h ft3, ft0, %[eps_half]\n"
            "vfsqrt.h ft3, ft3\n"
            "vfdiv.h ft1, %[ONE_VEC], ft3\n"
            : [ONE_VEC] "+fr"(ONE_VEC.f64), [eps_half] "+&fr"(eps_half.f64)
            : [eps] "fr"(eps), [ONE] "fr"(ONE),
              [n_frep] "r"(num_doubles_per_point -
                           1)  // we repeat n_frep+1 times
            : "ft0", "ft1", "ft2", "ft3");

        snrt_fpu_fence();                     // thought: do we need this?
        __builtin_ssr_barrier(SNRT_SSR_DM1);  // thought: do we need this?
        snrt_ssr_disable();
    }
    uint32_t end_invstd_calc = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    // compute weight*invstd and running_mean*invstd

    // computing invstd scratch and using it for weight: can we do it in 1 frep?
    // load running var: 1 ssr
    // write running var: 1 ssr
    // load weight: 1 ssr
    // write weight: 1 ssr
    // answer: not really. Still worth precomputing I think
    uint32_t start_running_var_weight_inplace_mul = SNRT_SECTIONED_MCYCLE();
    uint32_t end_running_var_weight_inplace_mul = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();
    uint32_t start_main_loop = SNRT_SECTIONED_MCYCLE();
    // compute grad_weight, grad_bias, grad_ifmap
    if (snrt_is_dm_core()) {
    } else if (compute_id == 0) {
        batchnorm_backward_fp16_no_loop(
            grad_ofmap_scratch, grad_ifmap_scratch, ifmap_scratch,
            running_mean_scratch, weight_scratch, invstd_scratch,
            grad_bias_scratch, grad_weight_scratch, num_bytes_per_aligned_point,
            num_points, num_points % 2, num_doubles_per_point, 1);
    }
    uint32_t end_main_loop = SNRT_SECTIONED_MCYCLE();
    // don't need second reduction
    snrt_cluster_hw_barrier();
    uint32_t start_grad_bias_weight_reduction = SNRT_SECTIONED_MCYCLE();
    uint32_t end_grad_bias_weight_reduction = SNRT_SECTIONED_MCYCLE();
    // write back grad_bias and grad_weight. then wait for all transactions to
    // complete
    uint32_t start_dma_writeback = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->grad_bias, grad_bias_scratch,
                          num_bytes_per_packed_point);
        snrt_dma_start_1d(l->grad_weight, grad_weight_scratch,
                          num_bytes_per_packed_point);
        initiate_dma_1d_or_2d(
            l->grad_ifmap, grad_ifmap_scratch, num_bytes_per_packed_point,
            num_bytes_per_packed_point, num_bytes_per_aligned_point, num_points,
            is_point_aligned_to_8_byte_boundary);
        snrt_dma_wait_all();
    } else if (compute_id == 0) {
    }
    uint32_t end_dma_writeback = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();
    uint32_t done = snrt_mcycle();
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT0);
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT1);
}

// No DMA, SSR, or FREP. Still uses TCDM.
static inline void batchnorm_backward_training_single_core(
    batchnorm_backward_training_layer_t *l) {
    uint32_t kernel_start = snrt_mcycle();
    // data is in HWC format
    const uint32_t num_clusters =
        snrt_cluster_num();  // how many clusters are there in total? currently
                             // 1 in the config i think
    const uint32_t cluster_id = snrt_cluster_idx();  // which cluster are we?
    const uint32_t num_compute_cores =
        snrt_cluster_compute_core_num();  // how many compute cores per cluster?
    const uint32_t compute_id =
        snrt_cluster_core_idx();  // which core are we in this cluster
    // Calculate output dimensions
    uint32_t N = 1;
    uint32_t H = l->IH;
    uint32_t W = l->IW;
    uint32_t C = l->CI;
    double eps = l->eps;
    uint32_t num_points = N * H * W;

    uint32_t invstd_scratch_len = l->CI, k_scratch_len = l->CI,
             grad_mean_scratch_len = l->CI, dx_scratch_len = l->CI;

    double *ptr = (double *)snrt_l1_start_addr();
    double *invstd_scratch = ptr;
    ptr += invstd_scratch_len;
    double *k_scratch = ptr;
    ptr += k_scratch_len;
    double *grad_mean_scratch = ptr;
    ptr += grad_mean_scratch_len;
    uint32_t start_dma_load = snrt_mcycle();
    uint32_t end_dma_load = SNRT_SECTIONED_MCYCLE();
    uint32_t start_compute_invstd_load = SNRT_SECTIONED_MCYCLE();
    if (compute_id == 0) {
        for (uint32_t channel = 0; channel < C; ++channel) {
            double invstd = 1 / sqrt(l->current_var[channel] + eps);
            double sum = 0;
            double dotp = 0;
            invstd_scratch[channel] = invstd;
            for (uint32_t i = 0; i < num_points; i++) {
                double dy = l->grad_ofmap[i * C + channel];
                double x = l->ifmap[i * C + channel];
                double curr_mean = l->current_mean[channel];
                sum += dy;
                dotp += dy * (x - curr_mean);
            }
            k_scratch[channel] = dotp * invstd * invstd / num_points;
            grad_mean_scratch[channel] = sum / num_points;
            l->grad_bias[channel] = sum;
            l->grad_weight[channel] = dotp * invstd;
        }
    }
    uint32_t end_compute_invstd_load = SNRT_SECTIONED_MCYCLE();
    uint32_t start_compute_sum_dotp_reduction_1 = SNRT_SECTIONED_MCYCLE();
    uint32_t end_compute_sum_dotp_reduction_1 = SNRT_SECTIONED_MCYCLE();
    uint32_t start_compute_sum_dotp_reduction_2 = SNRT_SECTIONED_MCYCLE();
    uint32_t end_compute_sum_dotp_reduction_2 = SNRT_SECTIONED_MCYCLE();
    uint32_t start_compute_k_grad_mean = SNRT_SECTIONED_MCYCLE();
    uint32_t end_compute_k_grad_mean = SNRT_SECTIONED_MCYCLE();
    uint32_t start_compute_grad_ifmap = SNRT_SECTIONED_MCYCLE();
    uint32_t end_compute_grad_ifmap = SNRT_SECTIONED_MCYCLE();

    uint32_t start_main_loop = SNRT_SECTIONED_MCYCLE();
    if (compute_id == 0) {
        for (uint32_t channel = 0; channel < C; ++channel) {
            double invstd = invstd_scratch[channel];
            double k = k_scratch[channel];
            double curr_mean = l->current_mean[channel];
            double grad_mean = grad_mean_scratch[channel];
            double weight = l->weight[channel];
            for (uint32_t i = 0; i < num_points; i += 1) {
                double dy = l->grad_ofmap[i * C + channel];
                double x = l->ifmap[i * C + channel];
                double dx = (x - curr_mean) * k;
                l->grad_ifmap[i * C + channel] =
                    (dy - grad_mean - dx) * invstd * weight;
            }
        }
    }
    uint32_t end_main_loop = SNRT_SECTIONED_MCYCLE();
    uint32_t start_grad_bias_weight_reduction = SNRT_SECTIONED_MCYCLE();
    uint32_t end_grad_bias_weight_reduction = SNRT_SECTIONED_MCYCLE();
    uint32_t start_dma_writeback = SNRT_SECTIONED_MCYCLE();
    uint32_t end_dma_writeback = SNRT_SECTIONED_MCYCLE();
    uint32_t done = snrt_mcycle();
}

// uses DMA, SSR, FREP
static inline void batchnorm_backward_training_single_core_opt_fp64(
    batchnorm_backward_training_layer_t *l) {
    uint32_t kernel_start = snrt_mcycle();
    // data is in HWC format
    const uint32_t compute_id = snrt_cluster_core_idx();

    // keep the dma core and one compute core
    uint32_t N = 1;
    uint32_t H = l->IH;
    uint32_t W = l->IW;
    uint32_t C = l->CI;
    // ignore tiling for now
    uint32_t num_points = N * H * W;
    double num_points_inv = 1 / ((double)num_points);
    uint32_t num_doubles = num_points * C;
    uint32_t num_bytes = num_doubles * sizeof(double);
    uint32_t point_size_in_bytes = C * sizeof(double);

    double *ptr = (double *)snrt_l1_start_addr();

    double *dotp_scratch = ptr;
    ptr += C;
    double *grad_bias_scratch = ptr;
    ptr += C;
    double *current_mean_scratch = ptr;
    ptr += C;
    double *invstd_scratch = ptr;
    ptr += C;
    double *weight_times_invstd_scratch = ptr;
    ptr += C;
    double *grad_weight_scratch = ptr;
    ptr += C;
    double *k_scratch = ptr;
    ptr += C;
    double *winvstd_times_meank_sub_dmean_scratch = ptr;
    ptr += C;

    ptrdiff_t grad_ofmap_len = num_points * C, grad_ifmap_len = grad_ofmap_len,
              ifmap_len = grad_ifmap_len;

    double *grad_ofmap_scratch = ptr;
    ptr += grad_ofmap_len;
    double *ifmap_scratch = ptr;
    ptr += ifmap_len;
    double *grad_ifmap_scratch = ifmap_scratch;  // reuse the buffer

    // Load data
    snrt_dma_txid_t invstd_load, curr_var_load, weight_load, curr_mean_load,
        grad_ofmap_load, ifmap_load;

    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT0,
                                     SNRT_PERF_CNT_ICACHE_STALL);
    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT1,
                                     SNRT_PERF_CNT_TCDM_CONGESTED);
    uint32_t start_dma_load = snrt_mcycle();
    if (snrt_is_dm_core()) {
        curr_var_load = snrt_dma_start_1d(invstd_scratch, l->current_var,
                                          point_size_in_bytes);
        weight_load = snrt_dma_start_1d(weight_times_invstd_scratch, l->weight,
                                        point_size_in_bytes);
        grad_ofmap_load =
            snrt_dma_start_1d(grad_ofmap_scratch, l->grad_ofmap, num_bytes);
        ifmap_load = snrt_dma_start_1d(ifmap_scratch, l->ifmap, num_bytes);
        curr_mean_load = snrt_dma_start_1d(
            current_mean_scratch, l->current_mean, point_size_in_bytes);
        snrt_dma_wait(curr_var_load);
        snrt_dma_wait(weight_load);
    } else if (compute_id == 0) {
        snrt_ssr_loop_2d(SNRT_SSR_DM_ALL, 2, C, point_size_in_bytes,
                         sizeof(double));
    }
    uint32_t end_dma_load = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    uint32_t start_invstd_computations = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_wait(grad_ofmap_load);
        snrt_dma_wait(ifmap_load);
        snrt_dma_wait(curr_mean_load);
    } else if (compute_id == 0) {
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, invstd_scratch);
        snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, invstd_scratch);
        register double eps = l->eps;
        const register double ONE = 1;
        snrt_ssr_enable();
        asm volatile(
            "frep.o %[n_frep], 5, 0, 0 \n"
            "fadd.d ft3, ft0, %[eps]\n"
            "fsqrt.d ft3, ft3\n"
            "fdiv.d ft3, %[ONE], ft3\n"
            // write out invstd
            "fsgnj.d ft1, ft3, ft3\n"
            // write out invstd * weight
            "fmul.d ft1, ft0, ft3\n"
            :
            : [eps] "fr"(eps), [ONE] "fr"(ONE), [n_frep] "r"(C - 1)
            : "ft0", "ft1", "ft2", "ft3");

        snrt_fpu_fence();
        __builtin_ssr_barrier(SNRT_SSR_DM1);
        snrt_ssr_disable();
    }
    uint32_t end_invstd_computations = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    uint32_t start_compute_sum_dotp_reduction = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
    } else if (compute_id == 0) {
        batchnorm_backward_training_fp64_no_loop_1(
            grad_ofmap_scratch, ifmap_scratch, current_mean_scratch,
            grad_bias_scratch, dotp_scratch, C, num_points, num_points % 3,
            num_points / 3 - 1, C, 1);
    }
    uint32_t end_compute_sum_dotp_reduction = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    uint32_t start_compute_k_grad_mean = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->grad_bias, grad_bias_scratch, point_size_in_bytes);
    } else if (compute_id == 0) {
        // need to compute
        // weight * invstd - can be done in pre-comp
        // grad_weight <- invstd*dotp
        // k <- grad_weight * invstd / num_points
        // (temp) grad_mean <- grad_bias / num_points
        // (mean * k - grad_mean) * (weight*invstd)
        //    = (mean * k)*weight*invstd-grad_mean * (weight*invstd)

        snrt_ssr_loop_2d(SNRT_SSR_DM0, 2, C, point_size_in_bytes,
                         sizeof(double));
        snrt_ssr_repeat(SNRT_SSR_DM0, 2);
        snrt_ssr_loop_2d(SNRT_SSR_DM1, 3, C, point_size_in_bytes,
                         sizeof(double));
        snrt_ssr_loop_2d(SNRT_SSR_DM2, 3, C, point_size_in_bytes,
                         sizeof(double));
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, invstd_scratch);
        snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, grad_weight_scratch);
        snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, dotp_scratch);
        snrt_ssr_enable();

        asm volatile(
            "frep.o %[n_frep], 9, 0, 0 \n"
            // grad_weight = invstd*dotp
            "fmul.d ft3, ft0, ft2\n"
            // grad_mean = grad_bias / num_points
            "fmul.d ft4, ft2, %[num_points_inv]\n"
            // ft5 = invstd / num_points
            "fmul.d ft5, ft0, %[num_points_inv]\n"
            // write out grad_weight
            "fsgnj.d ft1, ft3, ft3\n"
            // k = grad_weight * invstd / num_points
            "fmul.d ft6, ft3, ft5\n"
            // ft7 = grad_mean * (weight*invstd)
            "fmul.d ft7, ft4, ft0\n"
            // write out k
            "fsgnj.d ft1, ft6, ft6\n"
            // ft8 = mean * k
            "fmul.d ft8, ft2, ft6\n"
            // write out (mean*k)*(weight*invstd) - (grad_mean)*(weight*invstd)
            "fmsub.d ft1, ft8, ft0, ft7\n"
            :
            : [n_frep] "r"(C - 1), [num_points_inv] "fr"(num_points_inv)
            : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "ft8");
        snrt_fpu_fence();
        __builtin_ssr_barrier(SNRT_SSR_DM1);
        snrt_ssr_disable();
        snrt_ssr_repeat(SNRT_SSR_DM0, 1);
    }
    uint32_t end_compute_k_grad_mean = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    uint32_t start_compute_grad_ifmap = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->grad_weight, grad_weight_scratch,
                          point_size_in_bytes);
    } else if (compute_id == 0) {
        batchnorm_backward_training_fp64_no_loop_2(
            grad_ofmap_scratch, grad_ifmap_scratch, ifmap_scratch,
            weight_times_invstd_scratch, k_scratch,
            winvstd_times_meank_sub_dmean_scratch, C, num_points,
            num_points % 4, num_points / 4 - 1, C, 1);
    }
    uint32_t end_compute_grad_ifmap = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    uint32_t start_dma_writeback = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->grad_ifmap, grad_ifmap_scratch, num_bytes);
        snrt_dma_wait_all();
    }
    uint32_t end_dma_writeback = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();
    uint32_t done = snrt_mcycle();
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT0);
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT1);
}

static inline void batchnorm_backward_training_single_core_opt_fp32(
    batchnorm_backward_training_layer_t *l) {
    uint32_t kernel_start = snrt_mcycle();
    const uint32_t compute_id = snrt_cluster_core_idx();

    uint32_t N = 1;
    uint32_t H = l->IH;
    uint32_t W = l->IW;
    uint32_t C = l->CI;
    uint32_t num_points = N * H * W;
    float num_points_inv = (1/(float)num_points);
    precision_t dtype_bytes = l->dtype;
    uint32_t num_dtypes_per_double = (FP64 / dtype_bytes);

    uint32_t num_doubles_per_point = ceildiv(C, num_dtypes_per_double);
    uint32_t num_doubles = num_points * num_doubles_per_point;
    bool is_point_aligned_to_8_byte_boundary = C % num_dtypes_per_double == 0;
    uint32_t num_bytes_per_aligned_point =
        num_doubles_per_point * sizeof(double);
    uint32_t num_bytes_in_point_aligned_ifmap = num_doubles * sizeof(double);

    // not including padding for non-aligned points.
    // When is_point_aligned_to_8_byte_boundary == true, the expressions above
    // are equivalent. Output should be written according to the packed version
    uint32_t num_bytes_per_packed_point = C * dtype_bytes;

    // dataflow:
    v2s *ptr = (v2s *)snrt_l1_start_addr();

    v2s *dotp_scratch = ptr;
    ptr += num_doubles_per_point;
    v2s *grad_bias_scratch = ptr;
    ptr += num_doubles_per_point;
    v2s *current_mean_scratch = ptr;
    ptr += num_doubles_per_point;
    v2s *invstd_scratch = ptr;
    ptr += num_doubles_per_point;
    v2s *weight_times_invstd_scratch = ptr;
    ptr += num_doubles_per_point;
    v2s *grad_weight_scratch = ptr;
    ptr += num_doubles_per_point;
    v2s *k_scratch = ptr;
    ptr += num_doubles_per_point;
    v2s *winvstd_times_meank_sub_dmean_scratch = ptr;
    ptr += num_doubles_per_point;

    ptrdiff_t grad_ofmap_len = num_points * C, grad_ifmap_len = grad_ofmap_len,
              ifmap_len = grad_ifmap_len;

    v2s *grad_ofmap_scratch = ptr;
    ptr += grad_ofmap_len;
    v2s *ifmap_scratch = ptr;
    ptr += ifmap_len;
    v2s *grad_ifmap_scratch = ifmap_scratch;  // reuse the buffer

    snrt_dma_txid_t invstd_load, curr_var_load, weight_load, curr_mean_load,
        grad_ofmap_load, ifmap_load;

    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT0,
                                     SNRT_PERF_CNT_ICACHE_STALL);
    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT1,
                                     SNRT_PERF_CNT_TCDM_CONGESTED);
    uint32_t start_dma_load = snrt_mcycle();
    if (snrt_is_dm_core()) {
        curr_var_load = snrt_dma_start_1d(invstd_scratch, l->current_var,
                                          num_bytes_per_packed_point);
        weight_load = snrt_dma_start_1d(weight_times_invstd_scratch, l->weight,
                                        num_bytes_per_packed_point);
        snrt_dma_wait(curr_var_load);
        snrt_dma_wait(weight_load);
    } else if (compute_id == 0) {
        snrt_ssr_loop_2d(SNRT_SSR_DM_ALL, 2, num_doubles_per_point,
                         num_bytes_per_aligned_point,
                         sizeof(double));
    }
    uint32_t end_dma_load = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    uint32_t start_invstd_computations = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        grad_ofmap_load = initiate_dma_1d_or_2d(
            grad_ofmap_scratch, l->grad_ofmap, num_bytes_per_packed_point,
            num_bytes_per_aligned_point, num_bytes_per_packed_point, num_points,
            is_point_aligned_to_8_byte_boundary);
        ifmap_load = initiate_dma_1d_or_2d(
            ifmap_scratch, l->ifmap, dtype_bytes * C,
            num_bytes_per_aligned_point, num_bytes_per_packed_point, num_points,
            is_point_aligned_to_8_byte_boundary);
        curr_mean_load = snrt_dma_start_1d(
            current_mean_scratch, l->current_mean, num_bytes_per_packed_point);
        snrt_dma_wait_all();
    } else if (compute_id == 0) {
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, invstd_scratch);
        snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, invstd_scratch);
        register float eps = l->eps;
        const register float ONE = 1;
        snrt_ssr_enable();
        asm volatile(
            "vfcpka.s.s %[ONE],%[ONE],%[ONE]\n"  // duplicate the 1
            "frep.o %[n_frep], 5, 0, 0 \n"
            "vfadd.r.s ft3, ft0, %[eps]\n"
            "vfsqrt.s ft3, ft3\n"
            "vfdiv.s ft3, %[ONE], ft3\n"
            // write out invstd
            "vfsgnj.s ft1, ft3, ft3\n"
            // write out invstd * weight
            "vfmul.s ft1, ft0, ft3\n"
            :
            : [eps] "fr"(eps), [ONE] "fr"(ONE),
              [n_frep] "r"(num_doubles_per_point - 1)
            : "ft0", "ft1", "ft2", "ft3");

        snrt_fpu_fence();
        __builtin_ssr_barrier(SNRT_SSR_DM1);
        snrt_ssr_disable();
    }
    uint32_t end_invstd_computations = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    uint32_t start_compute_sum_dotp_reduction = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
    } else if (compute_id == 0) {
        batchnorm_backward_training_fp32_no_loop_1(
            grad_ofmap_scratch, ifmap_scratch, current_mean_scratch,
            grad_bias_scratch, dotp_scratch, num_bytes_per_aligned_point,
            num_points, num_points % 3, num_points / 3 - 1,
            num_doubles_per_point, 1);
    }
    uint32_t end_compute_sum_dotp_reduction = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    uint32_t start_compute_k_grad_mean = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->grad_bias, grad_bias_scratch,
                          num_bytes_per_packed_point);
    } else if (compute_id == 0) {
        // register v2s num_points_reg;
        register v2s num_points_inv_reg asm("ft6");  // can consider fcvt instead
        asm volatile(
            "vfcpka.s.s %[num_points_inv_reg],%[num_points_inv],%[num_points_inv]\n"  // duplicate the num_points
            : [num_points_inv_reg] "=fr"(num_points_inv_reg.f64)
            : [num_points_inv] "fr"(num_points_inv)
            : "ft0", "ft1", "ft2");
        
        snrt_ssr_loop_2d(SNRT_SSR_DM0, 2, num_doubles_per_point,
                         num_bytes_per_aligned_point,
                         sizeof(double));
        snrt_ssr_repeat(SNRT_SSR_DM0, 2);
        snrt_ssr_loop_2d(SNRT_SSR_DM1, 3, num_doubles_per_point,
                         num_bytes_per_aligned_point,
                         sizeof(double));
        snrt_ssr_loop_2d(SNRT_SSR_DM2, 3, num_doubles_per_point,
                         num_bytes_per_aligned_point,
                         sizeof(double));
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, invstd_scratch);
        snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, grad_weight_scratch);
        snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, dotp_scratch);
        snrt_ssr_enable();

        asm volatile(
            "frep.o %[n_frep], 10, 0, 0 \n"
            // grad_weight = invstd*dotp
            "vfmul.s ft3, ft0, ft2\n"
            // grad_mean = grad_bias / num_points
            "vfmul.s ft4, ft2, %[num_points_inv_reg]\n"
            // ft5 = invstd / num_points
            "vfmul.s ft5, ft0, %[num_points_inv_reg]\n"
            // write out grad_weight
            "vfsgnj.s ft1, ft3, ft3\n"
            // k = grad_weight * invstd / num_points
            "vfmul.s ft6, ft3, ft5\n"
            // ft7 = grad_mean * (weight*invstd)
            "vfmul.s ft7, ft4, ft0\n"
            // write out k
            "vfsgnj.s ft1, ft6, ft6\n"
            // ft8 = mean * k
            "vfmul.s ft8, ft2, ft6\n"
            // ft8 = (mean * k) * (weight * invstd)
            "vfmul.s ft8, ft8, ft0\n"
            // write out (mean*k)*(weight*invstd) - (grad_mean)*(weight*invstd)
            "vfsub.s ft1, ft8, ft7\n"
            :
            : [n_frep] "r"(num_doubles_per_point - 1), [num_points_inv_reg] "fr"(num_points_inv_reg.f64)
            : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "ft8");
        snrt_fpu_fence();
        __builtin_ssr_barrier(SNRT_SSR_DM1);
        snrt_ssr_disable();
        snrt_ssr_repeat(SNRT_SSR_DM0, 1);
    }
    uint32_t end_compute_k_grad_mean = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    uint32_t start_compute_grad_ifmap = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->grad_weight, grad_weight_scratch,
                          num_bytes_per_packed_point);
    } else if (compute_id == 0) {
        batchnorm_backward_training_fp32_no_loop_2(
            grad_ofmap_scratch, grad_ifmap_scratch, ifmap_scratch,
            weight_times_invstd_scratch, k_scratch,
            winvstd_times_meank_sub_dmean_scratch, num_bytes_per_aligned_point,
            num_points, num_points % 4, num_points / 4 - 1, num_doubles_per_point, 1);
    }
    uint32_t end_compute_grad_ifmap = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    uint32_t start_dma_writeback = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        initiate_dma_1d_or_2d(l->grad_ifmap, grad_ifmap_scratch,
                              dtype_bytes * C, num_bytes_per_packed_point,
                              num_bytes_per_aligned_point, num_points,
                              is_point_aligned_to_8_byte_boundary);
        snrt_dma_wait_all();
    }
    uint32_t end_dma_writeback = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();
    uint32_t done = snrt_mcycle();
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT0);
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT1);
}