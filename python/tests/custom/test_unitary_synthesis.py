# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import numpy as np

import cudaq

## NOTE: The random operations in this file are generated using
#        `scipy.stats.unitary_group.rvs(2)` with `seed=13`. The synthesized
#        kernels are generated by running transformation passes on the original
#        kernels which use the custom operation. These conversions are covered
#        in the `test/Transforms/UnitarySynthesis/random_unitary_*` tests.

## TODO: Set up this test suite such that the synthesized kernel is automatically
#        generated.


def check_state(matrix, state):
    # state must match the first column of the custom unitary matrix
    assert np.isclose(matrix[0][0], state[0])
    assert np.isclose(matrix[1][0], state[1])


def test_random_unitary_1():
    matrix1 = np.array([[-0.35004537 + 0.6609388j, 0.52346031 - 0.40818801j],
                        [-0.02186735 + 0.66343799j, -0.32826912 + 0.67202027j]])
    cudaq.register_operation("op1", matrix1)

    @cudaq.kernel
    def kernel1():
        q = cudaq.qubit()
        op1(q)

    check_state(matrix1, cudaq.get_state(kernel1))

    @cudaq.kernel
    def synth_kernel1():
        q = cudaq.qubit()
        rz(0.42144127523482622, q)
        ry(1.451771775281852, q)
        rz(5.8290733884896948, q)
        r1(-2.200142008639606, q)
        rz(2.200142008639606, q)

    check_state(matrix1, cudaq.get_state(synth_kernel1))


def test_random_unitary_2():
    matrix2 = np.array([[0.74299871 + 0.28281495j, -0.46740645 - 0.38665209j],
                        [-0.39644517 + 0.45912945j, -0.68548676 + 0.40266523j]])
    cudaq.register_operation("op2", matrix2)

    @cudaq.kernel
    def kernel2():
        q = cudaq.qubit()
        op2(q)

    check_state(matrix2, cudaq.get_state(kernel2))

    @cudaq.kernel
    def synth_kernel2():
        q = cudaq.qubit()
        rz(0.32741874404908811, q)
        ry(1.3035642670958536, q)
        rz(1.9193534130687422, q)
        r1(2.9741842417155158, q)
        rz(-2.9741842417155158, q)

    check_state(matrix2, cudaq.get_state(synth_kernel2))


def test_random_unitary_3():
    matrix3 = np.array([[0.08467619 - 0.65461771j, 0.74164005 + 0.1194807j],
                        [0.40327485 + 0.63377835j, 0.56763924 - 0.33686808j]])
    cudaq.register_operation("op3", matrix3)

    @cudaq.kernel
    def kernel3():
        q = cudaq.qubit()
        op3(q)

    check_state(matrix3, cudaq.get_state(kernel3))

    @cudaq.kernel
    def synth_kernel3():
        q = cudaq.qubit()
        rz(-1.5397032671761617, q)
        ry(1.6997647512980398, q)
        rz(2.4462690029098386, q)
        r1(-1.9777512475989565, q)
        rz(1.9777512475989565, q)

    check_state(matrix3, cudaq.get_state(synth_kernel3))


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
