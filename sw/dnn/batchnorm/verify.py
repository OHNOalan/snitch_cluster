#!/usr/bin/env python3
# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

import sys
from pathlib import Path
from operator import itemgetter

import numpy as np
import torch
from data.golden_models import (
    golden_model_backward,
    golden_model_backward_training,
    golden_model_forward_eval,
    golden_model_forward_training,
)
from data.datagen_constants import (
    ifmap_uid,
    running_mean_uid,
    running_var_uid,
    weight_uid,
    bias_uid,
    grad_ofmap_uid,
    ofmap_uid,
    grad_ifmap_uid,
    grad_weight_uid,
    grad_bias_uid,
    grad_ifmap_training_uid,
    grad_weight_training_uid,
    grad_bias_training_uid,
    BatchNormMode,
    struct_decls,
)
import functools

from typing import Optional, Tuple, Iterable, Dict

sys.path.append(str(Path(__file__).parent / "../../../util/sim/"))
import verification  # noqa: E402
from elf import Elf  # noqa: E402
from data_utils import (  # noqa: E402
    bytes_to_float,
    bytes_to_int,
    bytes_to_struct,
    floating_point_torch_type,
)

# Adapted from https://github.com/numpy/numpy/issues/10161#issuecomment-852783433
RTOL_FOR_PREC = {"64": 1e-9, "32": 1e-5, "16": 1e-2}
ATOL_FOR_PREC = {
    "64": 1e-9,
    "32": 2e-5,
    "16": 0,
}  # ?? I just want to make it pass and report errors later

PRECISION_T = {8: "64", 4: "32", 2: "16", 1: "8"}

NUMPY_T = {"64": np.float64, "32": np.float32, "16": np.float16}


errors_filepath = Path(__file__).resolve().parent / "batchnorm_verify_results"


def extract_torch_tensor_from_elf(elf, symbol, prec, shape: tuple):
    numpy_rep = bytes_to_float(elf.get_symbol_contents(symbol), prec).reshape(shape)
    return torch.tensor(numpy_rep)


def extract_torch_tensor_from_simulation(raw_results, symbol, prec, shape: tuple):
    return torch.from_numpy(bytes_to_float(raw_results[symbol], prec).reshape(shape))


def extract_torch_tensors_from_elf(
    elf,
    prec,
    tensor_shapes: Dict[str, Tuple],
    requires_grad_uids: Optional[Iterable[str]] = None,
):
    if requires_grad_uids is None:
        requires_grad_uids = []
    extracted_tensors = {
        uid: extract_torch_tensor_from_elf(elf, uid, prec, shape)
        for uid, shape in tensor_shapes
    }
    for uid in requires_grad_uids:
        extracted_tensors[uid].requires_grad = True
        extracted_tensors[uid].retain_grad()
    return extracted_tensors


def check_correctness(test, golden, actual, prec):
    golden = golden.detach().numpy().flatten()
    actual = actual.detach().numpy().flatten()
    fail = not np.allclose(
        actual,
        golden,
        rtol=RTOL_FOR_PREC[prec],
        atol=ATOL_FOR_PREC[prec],
        equal_nan=False,
    )
    absolute_err = np.absolute(golden - actual)
    relative_err = absolute_err / np.absolute(golden)
    if fail:
        print(f"FAIL: {test} verification failed.")
    else:
        print(f"{test} verification passed.")
    verification.dump_results_to_csv(
        [
            golden,
            actual,
            absolute_err,
            relative_err,
            RTOL_FOR_PREC[prec] * np.abs(golden) + ATOL_FOR_PREC[prec],
        ],
        errors_filepath / f"{test}.csv",
    )
    return int(fail)


def channels_last_to_contiguous_wrapper(model_fn):
    @functools.wraps(model_fn)
    def model_fn_nhwc_to_nchw_wrapper(*args, **nhwc_model_inputs):
        model_kwargs = {}
        for k, v in nhwc_model_inputs.items():
            if isinstance(v, torch.Tensor) and v.ndim == 4:
                # NHWC to NCHW
                model_kwargs[k] = v.permute(0, 3, 1, 2)
            else:
                model_kwargs[k] = v

        nchw_result_tensors = model_fn(**model_kwargs)
        if not isinstance(nchw_result_tensors, tuple):
            # single result
            nchw_result_tensors = (nchw_result_tensors,)
        nhwc_result_tensors = []
        for result in nchw_result_tensors:
            if isinstance(result, torch.Tensor) and result.ndim == 4:
                # NCHW to NHWC
                nhwc_result_tensors.append(result.permute(0, 2, 3, 1))
            else:
                nhwc_result_tensors.append(result)
        return nhwc_result_tensors

    return model_fn_nhwc_to_nchw_wrapper


def verify_forward_eval(elf):
    layer_struct = {
        "CI": "I",
        "IH": "I",
        "IW": "I",
        "TILE_CI": "I",
        "ifmap": "I",
        "ofmap": "I",
        "running_mean": "I",
        "running_var": "I",
        "weight": "I",
        "bias": "I",
        "eps": "f",
        "dtype": "I",
    }
    layer = bytes_to_struct(
        elf.get_symbol_contents(struct_decls[BatchNormMode.FORWARD_EVAL][1]),
        layer_struct,
    )

    C, H, W, eps, dtype = itemgetter("CI", "IH", "IW", "eps", "dtype")(layer)
    prec = PRECISION_T[dtype]

    input_tensor_shapes = [
        (ifmap_uid, (1, H, W, C)),
        (running_mean_uid, (C,)),
        (running_var_uid, (C,)),
        (weight_uid, (C,)),
        (bias_uid, (C,)),
    ]
    model_kwargs = extract_torch_tensors_from_elf(elf, prec, input_tensor_shapes)
    model_kwargs["eps"] = eps
    model_kwargs["dtype"] = floating_point_torch_type(prec)

    # the order of this is the order that it should appear in the model
    simulation_output_defs = [
        (ofmap_uid, (1, H, W, C)),
    ]

    return prec, model_kwargs, simulation_output_defs, golden_model_forward_eval


def verify_forward_training(elf):
    layer_struct = {
        "CI": "I",
        "IH": "I",
        "IW": "I",
        "ifmap": "I",
        "ofmap": "I",
        "running_mean": "I",
        "running_var": "I",
        "weight": "I",
        "bias": "I",
        "eps": "f",
        "momentum": "f",
        "dtype": "I",
    }
    layer = bytes_to_struct(
        elf.get_symbol_contents(struct_decls[BatchNormMode.FORWARD_TRAINING][1]),
        layer_struct,
    )

    C, H, W, eps, dtype, momentum = itemgetter(
        "CI", "IH", "IW", "eps", "dtype", "momentum"
    )(layer)
    prec = PRECISION_T[dtype]

    input_tensor_shapes = [
        (ifmap_uid, (1, H, W, C)),
        (running_mean_uid, (C,)),
        (running_var_uid, (C,)),
        (weight_uid, (C,)),
        (bias_uid, (C,)),
    ]
    model_kwargs = extract_torch_tensors_from_elf(elf, prec, input_tensor_shapes)
    model_kwargs["eps"] = eps
    model_kwargs["dtype"] = floating_point_torch_type(prec)
    model_kwargs["momentum"] = momentum

    # the order of this is the order that it should appear in the model
    simulation_output_defs = [
        (ofmap_uid, (1, H, W, C)),
        (running_mean_uid, (C,)),
        (running_var_uid, (C,)),
    ]

    return prec, model_kwargs, simulation_output_defs, golden_model_forward_training


def verify_backward_eval(elf):
    backward_layer_struct = {
        "CI": "I",
        "IH": "I",
        "IW": "I",
        "ifmap": "I",
        "grad_ofmap": "I",
        "running_mean": "I",
        "running_var": "I",
        "weight": "I",
        "grad_ifmap": "I",
        "grad_weight": "I",
        "grad_bias": "I",
        "eps": "f",
        "dtype": "I",
    }
    layer = bytes_to_struct(
        elf.get_symbol_contents(struct_decls[BatchNormMode.BACKWARD_EVAL][1]),
        backward_layer_struct,
    )

    C, H, W, eps, dtype = itemgetter("CI", "IH", "IW", "eps", "dtype")(layer)
    prec = PRECISION_T[dtype]

    input_tensor_shapes = [
        (ifmap_uid, (1, H, W, C)),
        (running_mean_uid, (C,)),
        (running_var_uid, (C,)),
        (weight_uid, (C,)),
        (bias_uid, (C,)),
        (grad_ofmap_uid, (1, H, W, C)),
    ]
    model_kwargs = extract_torch_tensors_from_elf(
        elf, prec, input_tensor_shapes, requires_grad_uids=(ifmap_uid,)
    )
    model_kwargs["eps"] = eps
    model_kwargs["dtype"] = floating_point_torch_type(prec)

    # the order of this is the order that it should appear in the model
    simulation_output_defs = [
        (grad_ifmap_uid, (1, H, W, C)),
        (grad_weight_uid, (C,)),
        (grad_bias_uid, (C,)),
    ]

    return prec, model_kwargs, simulation_output_defs, golden_model_backward


def verify_backward_training(elf):
    backward_training_layer_struct = {
        "CI": "I",
        "IH": "I",
        "IW": "I",
        "ifmap": "I",
        "grad_ofmap": "I",
        "current_mean": "I",
        "current_var": "I",
        "weight": "I",
        "grad_ifmap": "I",
        "grad_weight": "I",
        "grad_bias": "I",
        "eps": "f",
        "dtype": "I",
    }

    layer = bytes_to_struct(
        elf.get_symbol_contents(struct_decls[BatchNormMode.BACKWARD_TRAINING][1]),
        backward_training_layer_struct,
    )

    C, H, W, eps, dtype = itemgetter("CI", "IH", "IW", "eps", "dtype")(layer)
    prec = PRECISION_T[dtype]

    input_tensor_shapes = [
        (ifmap_uid, (1, H, W, C)),
        (weight_uid, (C,)),
        (bias_uid, (C,)),
        (grad_ofmap_uid, (1, H, W, C)),
    ]
    model_kwargs = extract_torch_tensors_from_elf(
        elf, prec, input_tensor_shapes, requires_grad_uids=(ifmap_uid,)
    )
    model_kwargs["eps"] = eps
    model_kwargs["dtype"] = floating_point_torch_type(prec)

    # the order of this is the order that it should appear in the model
    simulation_output_defs = [
        (grad_ifmap_training_uid, (1, H, W, C)),
        (grad_weight_training_uid, (C,)),
        (grad_bias_training_uid, (C,)),
    ]

    return prec, model_kwargs, simulation_output_defs, golden_model_backward_training


def main():
    errors_filepath.mkdir(parents=True, exist_ok=True)
    for f in errors_filepath.glob("*.csv"):
        f.unlink()
    # # Run simulation and get outputs
    args = verification.parse_args()

    print("Reading ELF to determine operation mode. ")
    # Extract input operands from ELF file
    if args.symbols_bin:
        elf = Elf(args.symbols_bin)
    else:
        elf = Elf(args.snitch_bin)

    is_forward = bytes_to_int(elf.get_symbol_contents("is_forward"))[0]
    is_training = bytes_to_int(elf.get_symbol_contents("is_training"))[0]

    if is_forward and is_training:
        (
            prec,
            golden_model_inputs,
            simulation_output_defs,
            model_fn,
        ) = verify_forward_training(elf)
    elif is_forward and not is_training:
        (
            prec,
            golden_model_inputs,
            simulation_output_defs,
            model_fn,
        ) = verify_forward_eval(elf)
    elif not is_forward and is_training:
        (
            prec,
            golden_model_inputs,
            simulation_output_defs,
            model_fn,
        ) = verify_backward_training(elf)
    elif not is_forward and not is_training:
        (
            prec,
            golden_model_inputs,
            simulation_output_defs,
            model_fn,
        ) = verify_backward_eval(elf)

    print("Beginning Simulation.")
    raw_results = verification.simulate(
        sim_bin=args.sim_bin,
        snitch_bin=args.snitch_bin,
        symbols_bin=args.symbols_bin,
        log=args.log,
        output_uids=[output_def[0] for output_def in simulation_output_defs] + ["temp"],
    )

    simulation_results = [
        extract_torch_tensor_from_simulation(raw_results, uid, prec, shape)
        for uid, shape in simulation_output_defs
    ]
    
    print("All data extracted from simulation and binary. Comparing to golden model. ")
    channels_last_model = channels_last_to_contiguous_wrapper(model_fn)
    golden_results = channels_last_model(**golden_model_inputs)

    return sum(
        check_correctness(
            simulation_output_defs[i][0], golden_results[i], simulation_results[i], prec
        )
        for i in range(len(simulation_output_defs))
    )


if __name__ == "__main__":
    sys.exit(main())
