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

from icon4py.model.atmosphere.dycore.stencils.compute_graddiv2_of_vn import compute_graddiv2_of_vn
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


def compute_graddiv2_of_vn_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    geofac_grdiv: np.ndarray,
    z_graddiv_vn: np.ndarray,
    z_graddiv2_vn: np.ndarray,
    horizontal_start: int,
    horizontal_end: int,
    **kwargs,
) -> np.ndarray:
    e2c2eO = connectivities[dims.E2C2EODim]
    geofac_grdiv = np.expand_dims(geofac_grdiv, axis=-1)
    z_graddiv2_vn[horizontal_start:horizontal_end, :] = np.sum(
        np.where(
            (e2c2eO != -1)[horizontal_start:horizontal_end, :, np.newaxis],
            z_graddiv_vn[e2c2eO[horizontal_start:horizontal_end]] * geofac_grdiv[horizontal_start:horizontal_end],
            0,
        ),
        axis=1,
    )
    return z_graddiv2_vn


@pytest.mark.embedded_remap_error
class TestComputeGraddiv2OfVn(StencilTest):
    PROGRAM = compute_graddiv2_of_vn
    OUTPUTS = ("z_graddiv2_vn",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        geofac_grdiv: np.ndarray,
        z_graddiv_vn: np.ndarray,
        z_graddiv2_vn: np.ndarray,
        horizontal_start: int,
        horizontal_end: int,
        **kwargs: Any,
    ) -> dict:
        z_graddiv2_vn = compute_graddiv2_of_vn_numpy(
            connectivities, geofac_grdiv, z_graddiv_vn,
            z_graddiv2_vn=z_graddiv2_vn,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )
        return dict(z_graddiv2_vn=z_graddiv2_vn)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        z_graddiv_vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        geofac_grdiv = random_field(grid, dims.EdgeDim, dims.E2C2EODim, dtype=wpfloat)
        z_graddiv2_vn = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)

        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
        horizontal_end = grid.end_index(edge_domain(h_grid.Zone.LOCAL))

        return dict(
            geofac_grdiv=geofac_grdiv,
            z_graddiv_vn=z_graddiv_vn,
            z_graddiv2_vn=z_graddiv2_vn,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
