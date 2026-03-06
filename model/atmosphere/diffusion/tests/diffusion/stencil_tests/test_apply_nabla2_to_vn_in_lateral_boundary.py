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
    # 1. Load the Parallelogram Grid
    mesh_nc = os.environ.get(
        "GT4PY_TRANSLATOR_MESH", 
        "/home/raphael/Documents/Studium/Msc_thesis/grid-generator/parallelogram_grid.nc"
    )
    if not os.path.exists(mesh_nc):
        pytest.skip(f"Mesh file {mesh_nc} not found.")
        
    ds = xr.open_dataset(mesh_nc)
    
    # Extract E2V and LonLat to build the index map
    e2v = ds["edge_vertices"].transpose("edge", "nc").values.astype(np.int32)
    e2v = np.where(e2v > 0, e2v - 1, -1)
    
    lon = ds["longitude_vertices"].values.astype(np.float64)
    lat = ds["latitude_vertices"].values.astype(np.float64)
    lonlat = np.stack([lon, lat], axis=1)
    
    nodes_size = ds.sizes["vertex"]
    n_edges = ds.sizes["edge"]
    num_levels = 10
    
    # Build the real, topologically correct IndexMap
    index_map = build_index_map_from_lonlat_e2v(lonlat, e2v, nodes_size=nodes_size)
    
    # 2. Generate random numpy arrays for the unstructured inputs based on the REAL grid size
    np.random.seed(42)
    fac_bdydiff_v = 5.0
    z_nabla2_e_np = np.random.rand(n_edges, num_levels)
    area_edge_np = np.random.rand(n_edges)
    vn_np = np.random.rand(n_edges, num_levels)
    
    # 3. Get expected Reference output from numpy
    expected_vn_np = apply_nabla2_to_vn_in_lateral_boundary_numpy(
        z_nabla2_e_np, area_edge_np, vn_np, fac_bdydiff_v
    )
    
    # 4. Translate unstructured edge fields to structured fields using the map
    z_nabla2_e_s = pack_edge_field(z_nabla2_e_np, index_map)
    area_edge_s = pack_edge_field(area_edge_np, index_map)
    vn_s = pack_edge_field(vn_np, index_map)
    
    # 5. Convert to gt4py Fields
    z_nabla2_e_f = gtx.as_field([dims.IDim, dims.JDim, dims.Kolor, dims.KDim], z_nabla2_e_s)
    area_edge_f = gtx.as_field([dims.IDim, dims.JDim, dims.Kolor], area_edge_s)
    vn_f = gtx.as_field([dims.IDim, dims.JDim, dims.Kolor, dims.KDim], vn_s)

    ni, nj = index_map.ij_to_vertex.shape
    
    # --- Resolve Backend ---
    if backend == "gtfn_cpu":
        selected_backend = gtfn_cpu
    else:
        raise ValueError(f"Backend {backend} not supported in this test.")

    # 6. Execute Structured Stencil
    prog = setup_program(
        apply_nabla2_to_vn_in_lateral_boundary_cart,
        backend=selected_backend,
        horizontal_sizes={
            "domain_max_i": gtx.int32(ni),
            "domain_max_j": gtx.int32(nj),
            "domain_max_kolor": gtx.int32(3),
        },
    )

    prog(
        z_nabla2_e=z_nabla2_e_f,
        area_edge=area_edge_f,
        vn=vn_f,  # Input AND Output
        fac_bdydiff_v=fac_bdydiff_v,
        domain_min_i=gtx.int32(0),
        domain_max_i=gtx.int32(ni),
        domain_min_j=gtx.int32(0),
        domain_max_j=gtx.int32(nj),
        domain_max_kolor=gtx.int32(3),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(num_levels),
        offset_provider={}
    )

    # 7. Unpack structured back to unstructured and Verify
    actual_vn_np = unpack_edge_field(vn_f.asnumpy(), index_map, n_edges)
    
    np.testing.assert_allclose(actual_vn_np, expected_vn_np, rtol=1e-12, atol=0)