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

from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_to_w import apply_nabla2_to_w#, apply_nabla2_to_w_cart
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field
from icon4py.model.testing.stencil_tests import StencilTest


def apply_nabla2_to_w_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    area: np.ndarray,
    z_nabla2_c: np.ndarray,
    geofac_n2s: np.ndarray,
    w: np.ndarray,
    diff_multfac_w: float,
) -> np.ndarray:
    c2e2cO = connectivities[dims.C2E2CODim]
    geofac_n2s = np.expand_dims(geofac_n2s, axis=-1)
    area = np.expand_dims(area, axis=-1)
    w = w - diff_multfac_w * area * area * np.sum(
        np.where((c2e2cO != -1)[:, :, np.newaxis], z_nabla2_c[c2e2cO] * geofac_n2s, 0.0), axis=1
    )
    return w


@pytest.mark.embedded_remap_error
class TestMoApplyNabla2ToW(StencilTest):
    PROGRAM = apply_nabla2_to_w
    OUTPUTS = ("w",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        area: np.ndarray,
        z_nabla2_c: np.ndarray,
        geofac_n2s: np.ndarray,
        w: np.ndarray,
        diff_multfac_w: float,
        **kwargs,
    ) -> dict:
        w = apply_nabla2_to_w_numpy(connectivities, area, z_nabla2_c, geofac_n2s, w, diff_multfac_w)
        return dict(w=w)

    @pytest.fixture
    def input_data(self, grid):
        area = random_field(grid, dims.CellDim, dtype=wpfloat)
        z_nabla2_c = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        geofac_n2s = random_field(grid, dims.CellDim, dims.C2E2CODim, dtype=wpfloat)
        w = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        return dict(
            area=area,
            z_nabla2_c=z_nabla2_c,
            geofac_n2s=geofac_n2s,
            w=w,
            diff_multfac_w=wpfloat("5.0"),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )


# def build_c2e2co_for_w(ni, nj, geofac_n2s_np):
#     n_cells = ni * nj * 2
#     c2e2co = np.full((n_cells, 3), -1, dtype=np.int32)
#     def ijk_to_c(i, j, k): return i * nj * 2 + j * 2 + k if 0 <= i < ni and 0 <= j < nj else -1

#     geofac_n2s_s = tuple(np.zeros((ni, nj, 2), dtype=np.float64) for _ in range(3))

#     for i in range(ni):
#         for j in range(nj):
#             c0, c1 = ijk_to_c(i, j, 0), ijk_to_c(i, j, 1)
#             n0_0, n0_1, n0_2 = ijk_to_c(i, j, 1), ijk_to_c(i, j-1, 1), ijk_to_c(i-1, j, 1)
#             n1_0, n1_1, n1_2 = ijk_to_c(i, j, 0), ijk_to_c(i, j+1, 0), ijk_to_c(i+1, j, 0)
#             c2e2co[c0] = [n0_0, n0_1, n0_2]
#             c2e2co[c1] = [n1_0, n1_1, n1_2]

#             for n_idx, n_c in enumerate([n0_0, n0_1, n0_2]):
#                 if n_c != -1: geofac_n2s_s[n_idx][i, j, 0] = geofac_n2s_np[c0, n_idx]
#             for n_idx, n_c in enumerate([n1_0, n1_1, n1_2]):
#                 if n_c != -1: geofac_n2s_s[n_idx][i, j, 1] = geofac_n2s_np[c1, n_idx]
#     return c2e2co, geofac_n2s_s

# def test_apply_nabla2_to_w_cartesian(backend="gtfn_cpu"):
#     ni, nj, num_levels = 10, 10, 10
#     n_cells = ni * nj * 2

#     np.random.seed(42)
#     area_np = np.random.rand(n_cells)
#     z_nabla2_c_np = np.random.rand(n_cells, num_levels)
#     geofac_n2s_np = np.random.rand(n_cells, 3)
#     w_np = np.random.rand(n_cells, num_levels)
#     diff_multfac_w = 5.0

#     c2e2co_np, geofac_n2s_s = build_c2e2co_for_w(ni, nj, geofac_n2s_np)
#     z_nabla2_c_safe = np.vstack([z_nabla2_c_np, np.zeros((1, num_levels))])

#     expected_output = TestMoApplyNabla2ToW.reference(
#         connectivities={dims.C2E2CODim: c2e2co_np}, area=area_np, z_nabla2_c=z_nabla2_c_safe,
#         geofac_n2s=geofac_n2s_np, w=w_np.copy(), diff_multfac_w=diff_multfac_w,
#     )

#     Kolor = getattr(dims, "Kolor", getattr(dims, "KolorDim", gtx.Dimension("Kolor")))
#     area_f = gtx.as_field([dims.IDim, dims.JDim, Kolor], pack_cell_field(area_np, ni, nj))
#     z_nabla2_c_f = gtx.as_field([dims.IDim, dims.JDim, Kolor, dims.KDim], pack_cell_field(z_nabla2_c_np, ni, nj))
#     geofac_n2s_f = tuple(gtx.as_field([dims.IDim, dims.JDim, Kolor], p) for p in geofac_n2s_s)
#     w_f = gtx.as_field([dims.IDim, dims.JDim, Kolor, dims.KDim], pack_cell_field(w_np, ni, nj))

#     selected_backend = gtfn_cpu
#     prog = setup_program(
#         apply_nabla2_to_w_cart, backend=selected_backend,
#         horizontal_sizes={"domain_min_i": gtx.int32(0), "domain_max_i": gtx.int32(ni), "domain_min_j": gtx.int32(0), "domain_max_j": gtx.int32(nj), "domain_max_kolor": gtx.int32(2)},
#     )
#     if hasattr(prog, "_static_args_names"): prog._static_args_names = set(prog._static_args_names) | {"domain_min_i", "domain_min_j"}

#     prog(area=area_f, z_nabla2_c=z_nabla2_c_f, geofac_n2s=geofac_n2s_f, w=w_f, diff_multfac_w=diff_multfac_w,
#          domain_min_i=gtx.int32(0), domain_max_i=gtx.int32(ni), domain_min_j=gtx.int32(0), domain_max_j=gtx.int32(nj), domain_max_kolor=gtx.int32(2), vertical_start=gtx.int32(0), vertical_end=gtx.int32(num_levels), offset_provider={})

#     actual_w = unpack_cell_field(w_f.asnumpy(), n_cells, ni, nj)
#     np.testing.assert_allclose(actual_w, expected_output["w"], rtol=1e-12, atol=0)