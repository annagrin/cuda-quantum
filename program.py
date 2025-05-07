# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import numpy as np

def test_control():
    @cudaq.kernel
    def kernel(a: float, b: float) -> float:
        return a * b

    ret = kernel(2, 4)
    assert np.isclose(ret, 8., atol=1e-12)

test_control()

