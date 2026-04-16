# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np
import pytest
from gt4py.next.program_processors.program_setup_utils import setup_program
from gt4py.next.program_processors.runners.gtfn import run_gtfn_cached as gtfn_cpu
from gt4py.next.modules.translator import pack_cell_field, unpack_cell_field

from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_to_w_in_upper_damping_layer import (
    apply_nabla2_to_w_in_upper_damping_layer,
    apply_nabla2_to_w_in_upper_damping_layer_cart,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field
from icon4py.model.testing.stencil_tests import StencilTest


def apply_nabla2_to_w_in_upper_damping_layer_numpy(
    w: np.ndarray,
    diff_multfac_n2w: np.ndarray,
    cell_area: np.ndarray,
    z_nabla2_c: np.ndarray,
) -> np.ndarray:
    cell_area = np.expand_dims(cell_area, axis=-1)
    w = w + diff_multfac_n2w * cell_area * z_nabla2_c
    return w


class TestApplyNabla2ToWInUpperDampingLayer(StencilTest):
    PROGRAM = apply_nabla2_to_w_in_upper_damping_layer
    OUTPUTS = ("w",)

    @pytest.fixture
    def input_data(self, grid: base.Grid):
        w = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        diff_multfac_n2w = random_field(grid, dims.KDim, dtype=wpfloat)
        cell_area = random_field(grid, dims.CellDim, dtype=wpfloat)
        z_nabla2_c = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            w=w,
            diff_multfac_n2w=diff_multfac_n2w,
            cell_area=cell_area,
            z_nabla2_c=z_nabla2_c,
            horizontal_start=0,
            horizontal_end=int(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        w: np.ndarray,
        diff_multfac_n2w: np.ndarray,
        cell_area: np.ndarray,
        z_nabla2_c: np.ndarray,
        **kwargs,
    ) -> dict:
        w = apply_nabla2_to_w_in_upper_damping_layer_numpy(
            w, diff_multfac_n2w, cell_area, z_nabla2_c
        )
        return dict(w=w)


# def test_apply_nabla2_to_w_in_upper_damping_layer_cartesian(backend="gtfn_cpu"):
#     ni, nj, num_levels = 10, 10, 10
#     n_cells = ni * nj * 2

#     np.random.seed(42)
#     w_np = np.random.rand(n_cells, num_levels)
#     diff_multfac_n2w_np = np.random.rand(num_levels)
#     cell_area_np = np.random.rand(n_cells)
#     z_nabla2_c_np = np.random.rand(n_cells, num_levels)

#     expected_output = TestApplyNabla2ToWInUpperDampingLayer.reference(
#         connectivities={}, w=w_np.copy(), diff_multfac_n2w=diff_multfac_n2w_np,
#         cell_area=cell_area_np, z_nabla2_c=z_nabla2_c_np,
#         horizontal_start=0, horizontal_end=n_cells, vertical_start=0, vertical_end=num_levels,
#     )

#     Kolor = getattr(dims, "Kolor", getattr(dims, "KolorDim", gtx.Dimension("Kolor")))
#     w_f = gtx.as_field([dims.IDim, dims.JDim, Kolor, dims.KDim], pack_cell_field(w_np, ni, nj))
#     diff_multfac_n2w_f = gtx.as_field([dims.KDim], diff_multfac_n2w_np)
#     cell_area_f = gtx.as_field([dims.IDim, dims.JDim, Kolor], pack_cell_field(cell_area_np, ni, nj))
#     z_nabla2_c_f = gtx.as_field([dims.IDim, dims.JDim, Kolor, dims.KDim], pack_cell_field(z_nabla2_c_np, ni, nj))

#     selected_backend = gtfn_cpu
#     prog = setup_program(
#         apply_nabla2_to_w_in_upper_damping_layer_cart, backend=selected_backend,
#         horizontal_sizes={"domain_min_i": gtx.int32(0), "domain_max_i": gtx.int32(ni), "domain_min_j": gtx.int32(0), "domain_max_j": gtx.int32(nj), "domain_max_kolor": gtx.int32(2)},
#     )

#     prog(w=w_f, diff_multfac_n2w=diff_multfac_n2w_f, cell_area=cell_area_f, z_nabla2_c=z_nabla2_c_f,
#          domain_min_i=gtx.int32(0), domain_max_i=gtx.int32(ni), domain_min_j=gtx.int32(0), domain_max_j=gtx.int32(nj), domain_max_kolor=gtx.int32(2), vertical_start=gtx.int32(0), vertical_end=gtx.int32(num_levels), offset_provider={})

#     actual_w_np = unpack_cell_field(w_f.asnumpy(), n_cells, ni, nj)
#     np.testing.assert_allclose(actual_w_np, expected_output["w"], rtol=1e-12, atol=0)