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
from gt4py.next.modules.translator import pack_edge_field, unpack_edge_field, build_index_map_from_lonlat_e2v
from gt4py.next.program_processors.runners.gtfn import run_gtfn_cached as gtfn_cpu
from gt4py.next.program_processors.program_setup_utils import setup_program
import xarray as xr
import os

from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_to_vn_in_lateral_boundary import (
    apply_nabla2_to_vn_in_lateral_boundary,
    apply_nabla2_to_vn_in_lateral_boundary_cart,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils.data_allocation import random_field
from icon4py.model.testing.stencil_tests import StencilTest

def apply_nabla2_to_vn_in_lateral_boundary_numpy(
    z_nabla2_e: np.array, area_edge: np.array, vn: np.array, fac_bdydiff_v
) -> np.array:
    area_edge = np.expand_dims(area_edge, axis=-1)
    vn = vn + (z_nabla2_e * area_edge * fac_bdydiff_v)
    return vn


class TestApplyNabla2ToVnInLateralBoundary(StencilTest):
    PROGRAM = apply_nabla2_to_vn_in_lateral_boundary
    OUTPUTS = ("vn",)

    @pytest.fixture
    def input_data(self, grid: base.Grid):
        fac_bdydiff_v = wpfloat("5.0")
        z_nabla2_e = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        area_edge = random_field(grid, dims.EdgeDim, dtype=wpfloat)
        vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        return dict(
            fac_bdydiff_v=fac_bdydiff_v,
            z_nabla2_e=z_nabla2_e,
            area_edge=area_edge,
            vn=vn,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        z_nabla2_e: np.ndarray,
        area_edge: np.ndarray,
        vn: np.ndarray,
        fac_bdydiff_v: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        vn = apply_nabla2_to_vn_in_lateral_boundary_numpy(z_nabla2_e, area_edge, vn, fac_bdydiff_v)
        return dict(vn=vn)


def test_apply_nabla2_to_vn_in_lateral_boundary_cartesian(backend="gtfn_cpu"):
    mesh_nc = os.environ.get(
        "GT4PY_TRANSLATOR_MESH", 
        "/home/raphael/Documents/Studium/Msc_thesis/grid-generator/parallelogram_grid.nc"
    )
    if not os.path.exists(mesh_nc):
        pytest.skip(f"Mesh file {mesh_nc} not found.")
        
    ds = xr.open_dataset(mesh_nc)
    e2v = np.where(ds["edge_vertices"].transpose("edge", "nc").values.astype(np.int32) > 0, 
                   ds["edge_vertices"].transpose("edge", "nc").values.astype(np.int32) - 1, -1)
    lonlat = np.stack([ds["longitude_vertices"].values, ds["latitude_vertices"].values], axis=1).astype(np.float64)
    
    nodes_size = ds.sizes["vertex"]
    n_edges = ds.sizes["edge"]
    num_levels = 10
    
    index_map = build_index_map_from_lonlat_e2v(lonlat, e2v, nodes_size=nodes_size)
    
    np.random.seed(42)
    fac_bdydiff_v = wpfloat("5.0")
    z_nabla2_e_np = np.random.rand(n_edges, num_levels)
    area_edge_np = np.random.rand(n_edges)
    vn_np = np.random.rand(n_edges, num_levels)
    
    # 3. GET GROUND TRUTH: Official icon4py reference method
    expected_output = TestApplyNabla2ToVnInLateralBoundary.reference(
        connectivities={},
        z_nabla2_e=z_nabla2_e_np,
        area_edge=area_edge_np,
        vn=vn_np.copy(),
        fac_bdydiff_v=fac_bdydiff_v,
        horizontal_start=0,
        horizontal_end=n_edges,
        vertical_start=0,
        vertical_end=num_levels,
    )
    expected_vn = expected_output["vn"]
    
    # 4. Translate to structured fields
    Kolor = getattr(dims, "Kolor", getattr(dims, "KolorDim", gtx.Dimension("Kolor")))
    z_nabla2_e_f = gtx.as_field([dims.IDim, dims.JDim, Kolor, dims.KDim], pack_edge_field(z_nabla2_e_np, index_map))
    area_edge_f = gtx.as_field([dims.IDim, dims.JDim, Kolor], pack_edge_field(area_edge_np, index_map))
    vn_f = gtx.as_field([dims.IDim, dims.JDim, Kolor, dims.KDim], pack_edge_field(vn_np, index_map))

    ni, nj = index_map.ij_to_vertex.shape
    
    selected_backend = gtfn_cpu

    prog = setup_program(
        apply_nabla2_to_vn_in_lateral_boundary_cart,
        backend=selected_backend,
        horizontal_sizes={
            "domain_min_i": gtx.int32(0), "domain_max_i": gtx.int32(ni),
            "domain_min_j": gtx.int32(0), "domain_max_j": gtx.int32(nj),
            "domain_max_kolor": gtx.int32(3),
        },
    )

    if hasattr(prog, "_static_args_names"):
        prog._static_args_names = set(prog._static_args_names) | {"domain_min_i", "domain_min_j"}

    prog(
        z_nabla2_e=z_nabla2_e_f, area_edge=area_edge_f, vn=vn_f, fac_bdydiff_v=fac_bdydiff_v,
        domain_min_i=gtx.int32(0), domain_max_i=gtx.int32(ni),
        domain_min_j=gtx.int32(0), domain_max_j=gtx.int32(nj),
        domain_max_kolor=gtx.int32(3),
        vertical_start=gtx.int32(0), vertical_end=gtx.int32(num_levels),
        offset_provider={}
    )

    actual_vn_np = unpack_edge_field(vn_f.asnumpy(), index_map, n_edges)
    np.testing.assert_allclose(actual_vn_np, expected_vn, rtol=1e-12, atol=1e-14)