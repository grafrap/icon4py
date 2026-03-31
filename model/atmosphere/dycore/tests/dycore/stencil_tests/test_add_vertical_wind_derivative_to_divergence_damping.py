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

    # get the unstructured mapping for edges and cells
    _, edge_backtransform, edge_unstructured_mask, horizontal_start = transform_to_unstructured(
        hmask_dd3d, 13, "Edge", 3
    )

    _, cell_backtransform, _, _ = transform_to_unstructured(
        z_dwdz_dd[:, 0], 13, "Cell", 2
    )

    # Define helper functions for mappings
    def to_unstructured(field: np.ndarray, backtransform: np.ndarray) -> np.ndarray:
        out = np.zeros_like(field)
        out[backtransform] = field
        return out

    def to_structured(field: np.ndarray, backtransform: np.ndarray) -> np.ndarray:
        return field[backtransform]

    # Convert fields to the unstructured ICON-like ordering.
    hmask_dd3d_u = edge_unstructured_mask
    inv_dual_edge_length_u = to_unstructured(inv_dual_edge_length, edge_backtransform)
    z_graddiv_vn_u = to_unstructured(z_graddiv_vn, edge_backtransform)
    z_dwdz_dd_u = to_unstructured(z_dwdz_dd, cell_backtransform)

    # Remap E2C to unstructured indexing on both axes:
    # 1) reorder rows by unstructured edge ids
    # 2) map structured cell ids to unstructured cell ids
    e2c_u = to_unstructured(e2c, edge_backtransform)
    e2c_u[:, 0] = cell_backtransform[e2c_u[:, 0]]
    e2c_u[:, 1] = cell_backtransform[e2c_u[:, 1]]

    z_dwdz_dd_e2c = z_dwdz_dd_u[e2c_u]

    # Compute the e2c difference
    z_dwdz_dd_weighted = z_dwdz_dd_e2c[:, 1, ...] - z_dwdz_dd_e2c[:, 0, ...]

    scalfac_dd3d_2d = np.expand_dims(scalfac_dd3d, axis=0)
    hmask_dd3d_2d = np.expand_dims(hmask_dd3d_u, axis=-1)
    inv_dual_edge_length_2d = np.expand_dims(inv_dual_edge_length_u, axis=-1)

    # write stencil result into a new array to avoid mutating the input z_graddiv_vn
    z_graddiv_vn_final_u = np.array(z_graddiv_vn_u, copy=True)
    z_graddiv_vn_final_u[horizontal_start:, :] = z_graddiv_vn_u[horizontal_start:, :] + (
        hmask_dd3d_2d[horizontal_start:, :]
        * scalfac_dd3d_2d
        * inv_dual_edge_length_2d[horizontal_start:, :]
        * z_dwdz_dd_weighted[horizontal_start:, :]
    )
    # Return in the original (structured) order expected by the test harness.
    return to_structured(z_graddiv_vn_final_u, edge_backtransform)


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
