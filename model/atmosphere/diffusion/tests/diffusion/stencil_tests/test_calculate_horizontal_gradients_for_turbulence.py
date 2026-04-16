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
from gt4py.next.program_processors.runners.gtfn import run_gtfn_cached as gtfn_cpu
from gt4py.next.program_processors.program_setup_utils import setup_program
from gt4py.next.modules.translator import pack_cell_field, unpack_cell_field

from icon4py.model.atmosphere.diffusion.stencils.calculate_horizontal_gradients_for_turbulence import (
    calculate_horizontal_gradients_for_turbulence,
    calculate_horizontal_gradients_for_turbulence_cart,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def calculate_horizontal_gradients_for_turbulence_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    w: np.ndarray,
    geofac_grg_x: np.ndarray,
    geofac_grg_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    c2e2cO = connectivities[dims.C2E2CODim]
    geofac_grg_x = np.expand_dims(geofac_grg_x, axis=-1)
    dwdx = np.sum(np.where((c2e2cO != -1)[:, :, np.newaxis], geofac_grg_x * w[c2e2cO], 0.0), axis=1)

    geofac_grg_y = np.expand_dims(geofac_grg_y, axis=-1)
    dwdy = np.sum(np.where((c2e2cO != -1)[:, :, np.newaxis], geofac_grg_y * w[c2e2cO], 0.0), axis=1)
    return dwdx, dwdy


@pytest.mark.embedded_remap_error
class TestCalculateHorizontalGradientsForTurbulence(StencilTest):
    PROGRAM = calculate_horizontal_gradients_for_turbulence
    OUTPUTS = ("dwdx", "dwdy")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        w: np.ndarray,
        geofac_grg_x: np.ndarray,
        geofac_grg_y: np.ndarray,
        **kwargs,
    ) -> dict:
        dwdx, dwdy = calculate_horizontal_gradients_for_turbulence_numpy(
            connectivities, w, geofac_grg_x, geofac_grg_y
        )
        return dict(dwdx=dwdx, dwdy=dwdy)

    @pytest.fixture
    def input_data(self, grid):
        w = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        geofac_grg_x = data_alloc.random_field(grid, dims.CellDim, dims.C2E2CODim, dtype=wpfloat)
        geofac_grg_y = data_alloc.random_field(grid, dims.CellDim, dims.C2E2CODim, dtype=wpfloat)
        dwdx = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        dwdy = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            w=w,
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
            dwdx=dwdx,
            dwdy=dwdy,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )


def build_c2e2co(ni, nj, geofac_x_np, geofac_y_np):
    """Mocks C2E2CO topology entirely in memory based on Cartesian geometry."""
    n_cells = ni * nj * 2
    c2e2co = np.full((n_cells, 3), -1, dtype=np.int32)
    def ijk_to_c(i, j, k): return i * nj * 2 + j * 2 + k if 0 <= i < ni and 0 <= j < nj else -1

    geofac_x_s = tuple(np.zeros((ni, nj, 2), dtype=np.float64) for _ in range(3))
    geofac_y_s = tuple(np.zeros((ni, nj, 2), dtype=np.float64) for _ in range(3))

    for i in range(ni):
        for j in range(nj):
            c0, c1 = ijk_to_c(i, j, 0), ijk_to_c(i, j, 1)
            n0_0, n0_1, n0_2 = ijk_to_c(i, j, 1), ijk_to_c(i, j-1, 1), ijk_to_c(i-1, j, 1)
            n1_0, n1_1, n1_2 = ijk_to_c(i, j, 0), ijk_to_c(i, j+1, 0), ijk_to_c(i+1, j, 0)
            c2e2co[c0] = [n0_0, n0_1, n0_2]
            c2e2co[c1] = [n1_0, n1_1, n1_2]

            for n_idx, n_c in enumerate([n0_0, n0_1, n0_2]):
                if n_c != -1:
                    geofac_x_s[n_idx][i, j, 0] = geofac_x_np[c0, n_idx]
                    geofac_y_s[n_idx][i, j, 0] = geofac_y_np[c0, n_idx]
            for n_idx, n_c in enumerate([n1_0, n1_1, n1_2]):
                if n_c != -1:
                    geofac_x_s[n_idx][i, j, 1] = geofac_x_np[c1, n_idx]
                    geofac_y_s[n_idx][i, j, 1] = geofac_y_np[c1, n_idx]
    return c2e2co, geofac_x_s, geofac_y_s

# def test_calculate_horizontal_gradients_for_turbulence_cartesian(backend="gtfn_cpu"):
#     ni, nj, num_levels = 10, 10, 10
#     n_cells = ni * nj * 2

#     np.random.seed(42)
#     w_np = np.random.rand(n_cells, num_levels)
#     geofac_grg_x_np = np.random.rand(n_cells, 3)
#     geofac_grg_y_np = np.random.rand(n_cells, 3)

#     c2e2co_np, geofac_x_s, geofac_y_s = build_c2e2co(ni, nj, geofac_grg_x_np, geofac_grg_y_np)
#     w_safe = np.vstack([w_np, np.zeros((1, num_levels))])

#     expected_output = TestCalculateHorizontalGradientsForTurbulence.reference(
#         connectivities={dims.C2E2CODim: c2e2co_np}, w=w_safe, geofac_grg_x=geofac_grg_x_np, geofac_grg_y=geofac_grg_y_np
#     )

#     Kolor = getattr(dims, "Kolor", getattr(dims, "KolorDim", gtx.Dimension("Kolor")))
#     w_f = gtx.as_field([dims.IDim, dims.JDim, Kolor, dims.KDim], pack_cell_field(w_np, ni, nj))
#     geofac_x_f = tuple(gtx.as_field([dims.IDim, dims.JDim, Kolor], p) for p in geofac_x_s)
#     geofac_y_f = tuple(gtx.as_field([dims.IDim, dims.JDim, Kolor], p) for p in geofac_y_s)
#     dwdx_f = gtx.as_field([dims.IDim, dims.JDim, Kolor, dims.KDim], np.zeros_like(w_f.asnumpy()))
#     dwdy_f = gtx.as_field([dims.IDim, dims.JDim, Kolor, dims.KDim], np.zeros_like(w_f.asnumpy()))

#     selected_backend = gtfn_cpu
#     prog = setup_program(
#         calculate_horizontal_gradients_for_turbulence_cart, backend=selected_backend,
#         horizontal_sizes={"domain_min_i": gtx.int32(0), "domain_max_i": gtx.int32(ni), "domain_min_j": gtx.int32(0), "domain_max_j": gtx.int32(nj), "domain_max_kolor": gtx.int32(2)},
#     )
#     if hasattr(prog, "_static_args_names"): prog._static_args_names = set(prog._static_args_names) | {"domain_min_i", "domain_min_j"}

#     prog(w=w_f, geofac_grg_x=geofac_x_f, geofac_grg_y=geofac_y_f, dwdx=dwdx_f, dwdy=dwdy_f,
#          domain_min_i=gtx.int32(0), domain_max_i=gtx.int32(ni), domain_min_j=gtx.int32(0), domain_max_j=gtx.int32(nj), domain_max_kolor=gtx.int32(2), vertical_start=gtx.int32(0), vertical_end=gtx.int32(num_levels), offset_provider={})

#     actual_dwdx = unpack_cell_field(dwdx_f.asnumpy(), n_cells, ni, nj)
#     actual_dwdy = unpack_cell_field(dwdy_f.asnumpy(), n_cells, ni, nj)
#     np.testing.assert_allclose(actual_dwdx, expected_output["dwdx"], rtol=1e-12, atol=0)
#     np.testing.assert_allclose(actual_dwdy, expected_output["dwdy"], rtol=1e-12, atol=0)