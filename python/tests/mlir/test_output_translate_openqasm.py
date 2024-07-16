# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import pytest
import numpy as np
import cudaq


def test_synth_and_openqasm():

    @cudaq.kernel
    def ghz(numQubits: int):
        qubits = cudaq.qvector(numQubits)
        h(qubits.front())
        for i in range(numQubits - 1):
            x.ctrl(qubits[i], qubits[i + 1])

    synth = cudaq.synthesize(ghz, 5)
    print(cudaq.translate(synth, format='openqasm2'))


# CHECK: // Code generated by NVIDIA's nvq++ compiler
# CHECK: OPENQASM 2.0;

# CHECK: include "qelib1.inc";

# CHECK: qreg var0[5];
# CHECK: h var0[0];
# CHECK: cx var0[0], var0[1];
# CHECK: cx var0[1], var0[2];
# CHECK: cx var0[2], var0[3];
# CHECK: cx var0[3], var0[4];
