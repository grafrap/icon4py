# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.diffusion.stencils.rbf_nabla4_direct import rbf_nabla4_direct
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StandardStaticVariants, StencilTest


def _calculate_nabla4_full_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    u_vert: np.ndarray,
    v_vert: np.ndarray,
    primal_normal_vert_v1: np.ndarray,
    primal_normal_vert_v2: np.ndarray,
    z_nabla2_e: np.ndarray,
    inv_vert_vert_length: np.ndarray,
    inv_primal_edge_length: np.ndarray,
) -> np.ndarray:
    """Compute nabla4 over the full edge array (no horizontal clipping).
    Used as an intermediate step before the vertex-level clip."""
    e2c2v = connectivities[dims.E2C2VDim]
    u_vert_e2c2v = u_vert[e2c2v]
    v_vert_e2c2v = v_vert[e2c2v]

    pnv1 = np.expand_dims(primal_normal_vert_v1, axis=-1)
    pnv2 = np.expand_dims(primal_normal_vert_v2, axis=-1)
    ivvl = np.expand_dims(inv_vert_vert_length, axis=-1)
    ipel = np.expand_dims(inv_primal_edge_length, axis=-1)

    nabv_tang = (
        u_vert_e2c2v[:, 0] * pnv1[:, 0] + v_vert_e2c2v[:, 0] * pnv2[:, 0]
    ) + (u_vert_e2c2v[:, 1] * pnv1[:, 1] + v_vert_e2c2v[:, 1] * pnv2[:, 1])
    nabv_norm = (
        u_vert_e2c2v[:, 2] * pnv1[:, 2] + v_vert_e2c2v[:, 2] * pnv2[:, 2]
    ) + (u_vert_e2c2v[:, 3] * pnv1[:, 3] + v_vert_e2c2v[:, 3] * pnv2[:, 3])
    return 4.0 * (
        (nabv_norm - 2.0 * z_nabla2_e) * ivvl**2
        + (nabv_tang - 2.0 * z_nabla2_e) * ipel**2
    )


def _mo_intp_rbf_vec_interpol_vertex_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    p_e_in: np.ndarray,
    ptr_coeff_1: np.ndarray,
    ptr_coeff_2: np.ndarray,
    horizontal_start: int,
    horizontal_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    v2e = connectivities[dims.V2EDim]
    ptr_coeff_1 = np.expand_dims(ptr_coeff_1, axis=-1)
    p_u_out = np.sum(
        np.where(np.expand_dims(v2e, axis=-1) >= 0, p_e_in[v2e] * ptr_coeff_1, 0.0), axis=1
    )
    ptr_coeff_2 = np.expand_dims(ptr_coeff_2, axis=-1)
    p_v_out = np.sum(
        np.where(np.expand_dims(v2e, axis=-1) >= 0, p_e_in[v2e] * ptr_coeff_2, 0.0), axis=1
    )
    p_u_final_out = np.zeros_like(p_u_out)
    p_v_final_out = np.zeros_like(p_v_out)
    p_u_final_out[horizontal_start:horizontal_end, :] = p_u_out[horizontal_start:horizontal_end, :]
    p_v_final_out[horizontal_start:horizontal_end, :] = p_v_out[horizontal_start:horizontal_end, :]
    return p_u_final_out, p_v_final_out


def _build_v2e2c2v(
    v2e: np.ndarray,
    e2c2v: np.ndarray,
) -> np.ndarray:
    # (V2E slot, E2C2V slot) -> V2E2C2V slot (7 unique vertices per vertex)
    lookup = [
        [0, 1, 2, 3],  # V2E[0]: center, N, NW, E
        [0, 3, 1, 4],  # V2E[1]: center, E, N, SE
        [0, 4, 3, 5],  # V2E[2]: center, SE, E, S
        [5, 0, 6, 4],  # V2E[3]: S, center, W, SE
        [6, 0, 2, 5],  # V2E[4]: W, center, NW, S
        [2, 0, 1, 6],  # V2E[5]: NW, center, N, W
    ]
    n_vertices = v2e.shape[0]
    v2e2c2v = np.full((n_vertices, 7), fill_value=-1, dtype=np.int32)
    for v in range(n_vertices):
        for se in range(6):
            e = v2e[v, se]
            if e < 0:
                continue
            for sv in range(4):
                slot = lookup[se][sv]
                v2e2c2v[v, slot] = e2c2v[e, sv]
    return v2e2c2v


@pytest.mark.uses_concat_where
@pytest.mark.continuous_benchmarking
class TestRBFNABLA4Direct(StencilTest):
    PROGRAM = rbf_nabla4_direct
    OUTPUTS = ("u_vert_out", "v_vert_out")
    STATIC_PARAMS = {
        StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "horizontal_start",
            "horizontal_end",
            "vertical_start",
            "vertical_end",
        ),
    }

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        u_vert: np.ndarray,
        v_vert: np.ndarray,
        primal_normal_vert_v1: np.ndarray,
        primal_normal_vert_v2: np.ndarray,
        z_nabla2_e: np.ndarray,
        inv_vert_vert_length: np.ndarray,
        inv_primal_edge_length: np.ndarray,
        ptr_coeff_1: np.ndarray,
        ptr_coeff_2: np.ndarray,
        horizontal_start: int,
        horizontal_end: int,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        z_nabla4_e2 = _calculate_nabla4_full_numpy(
            connectivities,
            u_vert,
            v_vert,
            primal_normal_vert_v1,
            primal_normal_vert_v2,
            z_nabla2_e,
            inv_vert_vert_length,
            inv_primal_edge_length,
        )
        u_vert_out, v_vert_out = _mo_intp_rbf_vec_interpol_vertex_numpy(
            connectivities,
            z_nabla4_e2,
            ptr_coeff_1,
            ptr_coeff_2,
            horizontal_start,
            horizontal_end,
        )
        return dict(u_vert_out=u_vert_out, v_vert_out=v_vert_out)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        u_vert = data_alloc.random_field(grid, dims.VertexDim, dims.KDim)
        v_vert = data_alloc.random_field(grid, dims.VertexDim, dims.KDim)

        primal_normal_vert_v1 = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2VDim)
        primal_normal_vert_v2 = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2VDim)

        inv_vert_vert_length = data_alloc.random_field(grid, dims.EdgeDim)
        inv_primal_edge_length = data_alloc.random_field(grid, dims.EdgeDim)

        z_nabla2_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)

        ptr_coeff_1 = data_alloc.random_field(grid, dims.VertexDim, dims.V2EDim, dtype=wpfloat)
        ptr_coeff_2 = data_alloc.random_field(grid, dims.VertexDim, dims.V2EDim, dtype=wpfloat)

        u_vert_out = data_alloc.zero_field(grid, dims.VertexDim, dims.KDim, dtype=wpfloat)
        v_vert_out = data_alloc.zero_field(grid, dims.VertexDim, dims.KDim, dtype=wpfloat)

        # Build V2E2C2V connectivity (7 unique vertex-to-vertex paths) and inject into grid.
        v2e_table = grid.connectivities["V2E"].asnumpy()
        e2c2v_table = grid.connectivities["E2C2V"].asnumpy()
        v2e2c2v_table = _build_v2e2c2v(v2e_table, e2c2v_table)
        grid.connectivities["V2E2C2V"] = gtx.as_connectivity(
            [dims.VertexDim, dims.V2E2C2VDim],
            dims.VertexDim,
            data=v2e2c2v_table,
            dtype=gtx.int32,
            skip_value=-1,
        )

        vertex_domain = h_grid.domain(dims.VertexDim)
        horizontal_start = grid.start_index(vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3))
        horizontal_end = grid.end_index(vertex_domain(h_grid.Zone.LOCAL))

        return dict(
            u_vert=u_vert,
            v_vert=v_vert,
            primal_normal_vert_v1=primal_normal_vert_v1,
            primal_normal_vert_v2=primal_normal_vert_v2,
            z_nabla2_e=z_nabla2_e,
            inv_vert_vert_length=inv_vert_vert_length,
            inv_primal_edge_length=inv_primal_edge_length,
            ptr_coeff_1=ptr_coeff_1,
            ptr_coeff_2=ptr_coeff_2,
            u_vert_out=u_vert_out,
            v_vert_out=v_vert_out,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
