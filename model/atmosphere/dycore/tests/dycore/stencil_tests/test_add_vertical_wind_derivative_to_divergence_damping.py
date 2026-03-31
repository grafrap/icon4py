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

from icon4py.model.atmosphere.dycore.stencils.add_vertical_wind_derivative_to_divergence_damping import (
    add_vertical_wind_derivative_to_divergence_damping,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests
from gt4py.next.modules.translator import transform_to_unstructured


def add_vertical_wind_derivative_to_divergence_damping_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    hmask_dd3d: np.ndarray,
    scalfac_dd3d: np.ndarray,
    inv_dual_edge_length: np.ndarray,
    z_dwdz_dd: np.ndarray,
    z_graddiv_vn: np.ndarray,
) -> np.ndarray:
    # Work on local copies only: this reference is called before the stencil and
    # must not mutate shared test inputs/connectivities.
    z_dwdz_dd = np.array(z_dwdz_dd, copy=True)
    e2c = np.array(connectivities[dims.E2CDim], copy=True)

    # transform to unstructured layout:
    mapping = transform_to_unstructured(hmask_dd3d, 2)
    backtransform = mapping[1]
    transform = mapping[0]

    cell_mapping = transform_to_unstructured(z_dwdz_dd[:,0], 2, "Cell")
    cell_backtransform = cell_mapping[1]
    print(cell_backtransform[:30])
    for i in range(e2c.shape[0]):
        v1, v2 = e2c[i, 0], e2c[i, 1]
        print(f"edge {i}: v1={v1}, v2={v2}")
        if v1 != -1:
            e2c[i, 0] = cell_backtransform[v1]
        if v2 != -1:
            e2c[i, 1] = cell_backtransform[v2]
        print(f"after transformation edge {i}: v1={e2c[i, 0]}, v2={e2c[i, 1]}")


    # Build unstructured cell field so remapped E2C cell indices gather correctly.
    z_dwdz_dd_unstructured = np.zeros_like(z_dwdz_dd)
    z_dwdz_dd_unstructured[cell_backtransform[:]] = z_dwdz_dd[:]
    z_dwdz_dd = z_dwdz_dd_unstructured


    z_dwdz_dd_e2c = z_dwdz_dd[e2c]
    
    # z_dwdz_dd_e2c has shape (n_edges, 2, n_k) when z_dwdz_dd carries a K-dim.
    # Build per-edge arrays for both neighbouring cells for all vertical levels.
    if z_dwdz_dd_e2c.ndim == 3:
        n_edges, _, n_k = z_dwdz_dd_e2c.shape
        e2c0 = np.zeros((n_edges, n_k), dtype=z_dwdz_dd_e2c.dtype)
        e2c1 = np.zeros((n_edges, n_k), dtype=z_dwdz_dd_e2c.dtype)

        mask0 = e2c[:, 0] != -1
        mask1 = e2c[:, 1] != -1

        if np.any(mask0):
            e2c0[mask0, :] = z_dwdz_dd_e2c[mask0, 0, :]
        if np.any(mask1):
            e2c1[mask1, :] = z_dwdz_dd_e2c[mask1, 1, :]

        # weighted difference per-edge, per-level
        z_dwdz_dd_weighted = e2c1 - e2c0
    else:
        # fall back to scalar behaviour
        n_edges = z_dwdz_dd_e2c.shape[0]
        e2c0 = np.zeros((n_edges,), dtype=z_dwdz_dd_e2c.dtype)
        e2c1 = np.zeros((n_edges,), dtype=z_dwdz_dd_e2c.dtype)
        for i in range(n_edges):
            e2c0[i] = z_dwdz_dd_e2c[i, 0] if e2c[i, 0] != -1 else 0.0
            e2c1[i] = z_dwdz_dd_e2c[i, 1] if e2c[i, 1] != -1 else 0.0
        z_dwdz_dd_weighted = e2c1 - e2c0

    # Keep the original stencil-style full update here for debugging/comparison.
    # scalfac_dd3d_2d = np.expand_dims(scalfac_dd3d, axis=0)
    # hmask_dd3d_2d = np.expand_dims(hmask_dd3d, axis=-1)
    # inv_dual_edge_length_2d = np.expand_dims(inv_dual_edge_length, axis=-1)
    #
    # # If needed, map edge-defined fields consistently with the edge mapping.
    # # hmask_dd3d_2d = hmask_dd3d_2d[backtransform[:], :]
    # # inv_dual_edge_length_2d = inv_dual_edge_length_2d[backtransform[:], :]
    #
    # z_graddiv_vn_full = z_graddiv_vn + (
    #     hmask_dd3d_2d
    #     * scalfac_dd3d_2d
    #     * inv_dual_edge_length_2d
    #     * z_dwdz_dd_weighted
    # )
    #
    # # Optional final remap back to structured order (if intermediate values are
    # # explicitly computed in unstructured edge order):
    # # z_graddiv_vn_struct = np.zeros_like(z_graddiv_vn_full)
    # # z_graddiv_vn_struct[transform[:], :] = z_graddiv_vn_full[:, :]


    return z_dwdz_dd_weighted


# @pytest.mark.skip_value_error
class TestAddVerticalWindDerivativeToDivergenceDamping(stencil_tests.StencilTest):
    PROGRAM = add_vertical_wind_derivative_to_divergence_damping
    OUTPUTS = ("z_graddiv_vn",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        hmask_dd3d: np.ndarray,
        scalfac_dd3d: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        z_dwdz_dd: np.ndarray,
        z_graddiv_vn: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        z_graddiv_vn = add_vertical_wind_derivative_to_divergence_damping_numpy(
            connectivities,
            hmask_dd3d,
            scalfac_dd3d,
            inv_dual_edge_length,
            z_dwdz_dd,
            z_graddiv_vn,
        )
        return dict(z_graddiv_vn=z_graddiv_vn)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        hmask_dd3d = data_alloc.random_field(grid, dims.EdgeDim, dtype=ta.wpfloat)
        scalfac_dd3d = data_alloc.random_field(grid, dims.KDim, dtype=ta.wpfloat)
        inv_dual_edge_length = data_alloc.random_field(grid, dims.EdgeDim, dtype=ta.wpfloat)
        z_dwdz_dd = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        z_graddiv_vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)
        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3))
        print(f"horizonta_start: ",horizontal_start)


        return dict(
            hmask_dd3d=hmask_dd3d,
            scalfac_dd3d=scalfac_dd3d,
            inv_dual_edge_length=inv_dual_edge_length,
            z_dwdz_dd=z_dwdz_dd,
            z_graddiv_vn=z_graddiv_vn,
            horizontal_start=horizontal_start,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
