# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Author: Luca Colagrande <colluca@iis.ee.ethz.ch>

import struct
from datetime import datetime
import torch
import numpy as np


def emit_license():
    s = (f"// Copyright {datetime.now().year} ETH Zurich and University of Bologna.\n"
         f"// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n"
         f"// SPDX-License-Identifier: Apache-2.0\n")
    return s


def floating_point_torch_type(precision):
    prec_to_torch_type_map = {
        '64': torch.float64,
        '32': torch.float32,
        '16': torch.float16,
        '8': None
    }
    return prec_to_torch_type_map[precision]


def floating_point_numpy_type(precision: str) -> np.dtype:
    prec_to_numpy_dtype_map = {
        '64': np.float64,
        '32': np.float32,
        '16': np.float16,
        '8': None
    }
    return prec_to_numpy_dtype_map[precision]


# Returns the C type representing a floating-point value of the specified precision
def floating_point_ctype(precision):
    prec_to_fp_type_map = {
        '64': 'double',
        '32': 'float',
        '16': '__fp16',
        '8': '__fp8'
    }
    return prec_to_fp_type_map[precision]


def flatten(array):
    if isinstance(array, np.ndarray):
        return array.flatten()
    if isinstance(array, torch.Tensor):
        return array.numpy().flatten()


def variable_attributes(alignment=None, section=None):
    attributes = ''
    if alignment:
        attributes = f'__attribute__ ((aligned ({alignment})))'
    if section:
        attributes += f' __attribute__ ((section ("{section}")))'
    return attributes


def alias_dtype(dtype):
    if dtype == '__fp8':
        return 'char'
    else:
        return dtype


def format_array_declaration(dtype, uid, shape, alignment=None, section=None):
    attributes = variable_attributes(alignment, section)
    s = f'{alias_dtype(dtype)} {uid}'
    for dim in shape:
        s += f'[{dim}]'
    if attributes:
        s += f' {attributes};'
    else:
        s += ';'
    return s


# In the case of dtype __fp8, array field expects a dictionary of
# sign, exponent and mantissa arrays
def format_array_definition(dtype, uid, array, alignment=None, section=None):
    # Definition starts with the declaration stripped off of the terminating semicolon
    s = format_array_declaration(dtype, uid, array.shape, alignment, section)[:-1]
    s += ' = {\n'
    # Flatten array
    if dtype == '__fp8':
        array = zip(flatten(array['sign']),
                    flatten(array['exponent']),
                    flatten(array['mantissa']))
    else:
        array = flatten(array)
    # Format array elements
    for el in array:
        if dtype == '__fp8':
            sign, exp, mant = el
            el = sign * 2**7 + exp * 2**2 + mant
            el_str = f'0x{el:02x}'
        else:
            el_str = f'{el}'
        s += f'\t{el_str},\n'
    s += '};'
    return s


def format_scalar_definition(dtype, uid, scalar):
    s = f'{alias_dtype(dtype)} {uid} = {scalar};'
    return s


def format_struct_definition(dtype, uid, map):
    s = f'{alias_dtype(dtype)} {uid} = {{\n'
    s += ',\n'.join([f'\t.{key} = {value}' for (key, value) in map.items()])
    s += '\n};'
    return s


def format_ifdef_wrapper(macro, body):
    s = f'#ifdef {macro}\n'
    s += f'{body}\n'
    s += f'#endif // {macro}\n'
    return s


# bytearray assumed little-endian
def bytes_to_struct(byte_array, struct_map):
    struct_fields = struct_map.keys()
    fmt_specifiers = struct_map.values()
    fmt_string = ''.join(fmt_specifiers)
    field_values = struct.unpack(f'<{fmt_string}', byte_array)
    return dict(zip(struct_fields, field_values))


# bytearray assumed little-endian
def bytes_to_float(byte_array, prec='64') -> np.ndarray:
    numpy_dtype = np.dtype(floating_point_numpy_type(prec)).newbyteorder('<')

    numpy_array = np.frombuffer(byte_array, dtype=numpy_dtype)
    if numpy_array.size == 1:
        return numpy_array[0]
    else:
        return numpy_array.copy()


# bytearray assumed little-endian
def bytes_to_int(byte_array, prec='32', signedness='unsigned'):
    assert prec == '32', "Only 32 bit precision supported so far"
    assert signedness == 'unsigned', "Only unsigned integers supported so far"

    uint32_size = struct.calcsize('I')  # Size of a uint32 in bytes
    num_uints = len(byte_array) // uint32_size

    # Unpack the byte array into a list of uints
    uints = []
    for i in range(num_uints):
        uint32_bytes = byte_array[i * uint32_size:(i + 1) * uint32_size]
        uint = struct.unpack('<I', uint32_bytes)[0]
        uints.append(uint)
    return uints
