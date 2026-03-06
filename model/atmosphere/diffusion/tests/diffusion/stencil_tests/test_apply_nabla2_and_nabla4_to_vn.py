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

from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_and_nabla4_to_vn import (
    apply_nabla2_and_nabla4_to_vn,
    apply_nabla2_and_nabla4_to_vn_cart,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field
from icon4py.model.testing.stencil_tests import StencilTest

from gt4py.next import as_field
from gt4py.next.modules.translator import pack_edge_field, unpack_edge_field, build_index_map_from_lonlat_e2v
from gt4py.next.program_processors.program_setup_utils import setup_program
from gt4py.next.program_processors.runners.gtfn import run_gtfn_cached as gtfn_cpu
import xarray as xr
import os


def apply_nabla2_and_nabla4_to_vn_numpy(
    area_edge,
    kh_smag_e,
    z_nabla2_e,
    z_nabla4_e2,
    diff_multfac_vn,
    nudgecoeff_e,
    vn,
    nudgezone_diff,
):
    area_edge = np.expand_dims(area_edge, axis=-1)
    diff_multfac_vn = np.expand_dims(diff_multfac_vn, axis=0)
    nudgecoeff_e = np.expand_dims(nudgecoeff_e, axis=-1)
    vn = vn + area_edge * (
        np.maximum(nudgezone_diff * nudgecoeff_e, kh_smag_e) * z_nabla2_e
        - diff_multfac_vn * z_nabla4_e2 * area_edge
    )
    return vn


class TestApplyNabla2AndNabla4ToVn(StencilTest):
    PROGRAM = apply_nabla2_and_nabla4_to_vn
    OUTPUTS = ("vn",)

    @pytest.fixture
    def input_data(self, grid):
        area_edge = random_field(grid, dims.EdgeDim, dtype=wpfloat)
        kh_smag_e = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        z_nabla2_e = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        z_nabla4_e2 = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        diff_multfac_vn = random_field(grid, dims.KDim, dtype=wpfloat)
        nudgecoeff_e = random_field(grid, dims.EdgeDim, dtype=wpfloat)
        vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        nudgezone_diff = vpfloat("9.0")

        return dict(
            area_edge=area_edge,
            kh_smag_e=kh_smag_e,
            z_nabla2_e=z_nabla2_e,
            z_nabla4_e2=z_nabla4_e2,
            diff_multfac_vn=diff_multfac_vn,
            nudgecoeff_e=nudgecoeff_e,
            vn=vn,
            nudgezone_diff=nudgezone_diff,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        area_edge: np.ndarray,
        kh_smag_e: np.ndarray,
        z_nabla2_e: np.ndarray,
        z_nabla4_e2: np.ndarray,
        diff_multfac_vn: np.ndarray,
        nudgecoeff_e: np.ndarray,
        vn: np.ndarray,
        nudgezone_diff: np.ndarray,
        **kwargs,
    ) -> dict:
        vn = apply_nabla2_and_nabla4_to_vn_numpy(
            area_edge,
            kh_smag_e,
            z_nabla2_e,
            z_nabla4_e2,
            diff_multfac_vn,
            nudgecoeff_e,
            vn,
            nudgezone_diff,
        )
        return dict(vn=vn)



def test_apply_nabla2_and_nabla4_to_vn_cartesian(backend="gtfn_cpu"):
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
    num_levels = 10 # Since the netCDF is 2D, we arbitrarily pick 10 vertical levels for the test
    
    # Build the real, topologically correct IndexMap
    index_map = build_index_map_from_lonlat_e2v(lonlat, e2v, nodes_size=nodes_size)
    
    # 2. Generate random numpy arrays for the unstructured inputs based on the REAL grid size
    np.random.seed(42)
    area_edge_np = np.random.rand(n_edges)
    kh_smag_e_np = np.random.rand(n_edges, num_levels)
    z_nabla2_e_np = np.random.rand(n_edges, num_levels)
    z_nabla4_e2_np = np.random.rand(n_edges, num_levels)
    diff_multfac_vn_np = np.random.rand(num_levels)
    nudgecoeff_e_np = np.random.rand(n_edges)
    vn_np = np.random.rand(n_edges, num_levels)
    nudgezone_diff = 9.0
    
    # 3. Get expected Reference output from numpy
    expected_vn_np = apply_nabla2_and_nabla4_to_vn_numpy(
        area_edge_np, kh_smag_e_np, z_nabla2_e_np, z_nabla4_e2_np, 
        diff_multfac_vn_np, nudgecoeff_e_np, vn_np, nudgezone_diff
    )
    
    # 4. Translate unstructured edge fields to structured fields using the real map
    area_edge_s = pack_edge_field(area_edge_np, index_map)
    kh_smag_e_s = pack_edge_field(kh_smag_e_np, index_map)
    z_nabla2_e_s = pack_edge_field(z_nabla2_e_np, index_map)
    z_nabla4_e2_s = pack_edge_field(z_nabla4_e2_np, index_map)
    nudgecoeff_e_s = pack_edge_field(nudgecoeff_e_np, index_map)
    vn_s = pack_edge_field(vn_np, index_map)
    
    # 5. Convert to gt4py Fields
    area_edge_f = gtx.as_field([dims.IDim, dims.JDim, dims.Kolor], area_edge_s)
    kh_smag_e_f = gtx.as_field([dims.IDim, dims.JDim, dims.Kolor, dims.KDim], kh_smag_e_s)
    z_nabla2_e_f = gtx.as_field([dims.IDim, dims.JDim, dims.Kolor, dims.KDim], z_nabla2_e_s)
    z_nabla4_e2_f = gtx.as_field([dims.IDim, dims.JDim, dims.Kolor, dims.KDim], z_nabla4_e2_s)
    diff_multfac_vn_f = gtx.as_field([dims.KDim], diff_multfac_vn_np)
    nudgecoeff_e_f = gtx.as_field([dims.IDim, dims.JDim, dims.Kolor], nudgecoeff_e_s)
    vn_f = gtx.as_field([dims.IDim, dims.JDim, dims.Kolor, dims.KDim], vn_s)


    ni, nj = index_map.ij_to_vertex.shape
    
    # 6. Execute Structured Stencil
    prog = setup_program(
        apply_nabla2_and_nabla4_to_vn_cart,
        backend=gtfn_cpu,
        horizontal_sizes={
            "domain_max_i": gtx.int32(ni),
            "domain_max_j": gtx.int32(nj),
            "domain_max_kolor": gtx.int32(3),
        },
    )

    prog(
        area_edge=area_edge_f,
        kh_smag_e=kh_smag_e_f,
        z_nabla2_e=z_nabla2_e_f,
        z_nabla4_e2=z_nabla4_e2_f,
        diff_multfac_vn=diff_multfac_vn_f,
        nudgecoeff_e=nudgecoeff_e_f,
        vn=vn_f,
        nudgezone_diff=nudgezone_diff,
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