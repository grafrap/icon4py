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
from gt4py.next.modules.translator import pack_cell_field, unpack_cell_field, build_index_map_from_lonlat_e2v, build_cell_to_ijk

from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_for_w import (
    calculate_nabla2_for_w,
    calculate_nabla2_for_w_cart,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.utils.data_allocation import constant_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


def calculate_nabla2_for_w_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray], w: np.ndarray, geofac_n2s: np.ndarray
) -> np.ndarray:
    c2e2cO = connectivities[dims.C2E2CODim]
    geofac_n2s = np.expand_dims(geofac_n2s, axis=-1)
    z_nabla2_c = np.sum(
        np.where((c2e2cO != -1)[:, :, np.newaxis], w[c2e2cO] * geofac_n2s, 0), axis=1
    )
    return z_nabla2_c


@pytest.mark.embedded_remap_error
class TestCalculateNabla2ForW(StencilTest):
    PROGRAM = calculate_nabla2_for_w
    OUTPUTS = ("z_nabla2_c",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        w: np.ndarray,
        geofac_n2s: np.ndarray,
        **kwargs,
    ) -> dict:
        z_nabla2_c = calculate_nabla2_for_w_numpy(connectivities, w, geofac_n2s)
        return dict(z_nabla2_c=z_nabla2_c)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        w = constant_field(grid, 1.0, dims.CellDim, dims.KDim)
        geofac_n2s = constant_field(grid, 2.0, dims.CellDim, dims.C2E2CODim)
        z_nabla2_c = zero_field(grid, dims.CellDim, dims.KDim)

        return dict(
            w=w,
            geofac_n2s=geofac_n2s,
            z_nabla2_c=z_nabla2_c,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )


def build_c2e2co_and_geofac(ijk_to_cell, geofac_n2s_np):
    ni, nj, _ = ijk_to_cell.shape
    n_cells = geofac_n2s_np.shape[0]
    
    # We construct the unstructured connectivity table explicitly matching Cartesian shifts
    c2e2co = np.full((n_cells, 3), -1, dtype=np.int32)
    geofac_s = tuple(np.zeros((ni, nj, 2), dtype=np.float64) for _ in range(3))
    
    for i in range(ni):
        for j in range(nj):
            c0 = ijk_to_cell[i, j, 0]
            c1 = ijk_to_cell[i, j, 1]
            
            if c0 >= 0:
                n0_0 = ijk_to_cell[i, j, 1] if j>=0 else -1
                n0_1 = ijk_to_cell[i, j-1, 1] if j-1>=0 else -1
                n0_2 = ijk_to_cell[i-1, j, 1] if i-1>=0 else -1
                c2e2co[c0] = [n0_0, n0_1, n0_2]
                for idx, n_c in enumerate([n0_0, n0_1, n0_2]):
                    if n_c != -1: geofac_s[idx][i, j, 0] = geofac_n2s_np[c0, idx]
                    
            if c1 >= 0:
                n1_0 = ijk_to_cell[i, j, 0] if j>=0 else -1
                n1_1 = ijk_to_cell[i, j+1, 0] if j+1<nj else -1
                n1_2 = ijk_to_cell[i+1, j, 0] if i+1<ni else -1
                c2e2co[c1] = [n1_0, n1_1, n1_2]
                for idx, n_c in enumerate([n1_0, n1_1, n1_2]):
                    if n_c != -1: geofac_s[idx][i, j, 1] = geofac_n2s_np[c1, idx]
                    
    return c2e2co, geofac_s

# # ----------------- Test Execution -----------------
# import os
# import xarray as xr
# def test_calculate_nabla2_for_w_cartesian(backend="gtfn_cpu"):
#     mesh_nc = os.environ.get(
#         "GT4PY_TRANSLATOR_MESH", 
#         "/home/raphael/Documents/Studium/Msc_thesis/grid-generator/parallelogram_grid.nc"
#     )
#     if not os.path.exists(mesh_nc):
#         pytest.skip(f"Mesh file {mesh_nc} not found.")

#     ds = xr.open_dataset(mesh_nc)
#     e2v = np.where(
#         ds["edge_vertices"].transpose("edge", "nc").values.astype(np.int32) > 0, 
#         ds["edge_vertices"].transpose("edge", "nc").values.astype(np.int32) - 1, -1
#     )
#     lonlat = np.stack([ds["longitude_vertices"].values, ds["latitude_vertices"].values], axis=1).astype(np.float64)

#     nodes_size = ds.sizes["vertex"]
#     n_cells = ds.sizes["cell"]
#     num_levels = 10

#     index_map = build_index_map_from_lonlat_e2v(lonlat, e2v, nodes_size=nodes_size)
#     ni, nj = index_map.ij_to_vertex.shape

#     # Construct the Cell topology directly from the netCDF cell definitions
#     ijk_to_cell = build_cell_to_ijk(index_map, ds)

#     np.random.seed(42)
#     w_np = np.random.rand(n_cells, num_levels)
#     geofac_n2s_np = np.random.rand(n_cells, 3)

#     # Convert the cartesian topology bounds into an exact unstructured connectivity table
#     c2e2co_np, geofac_n2s_s = build_c2e2co_and_geofac(ijk_to_cell, geofac_n2s_np)

#     # GET GROUND TRUTH: Official icon4py reference method (No vstack needed for cell arrays)
#     expected_output = TestCalculateNabla2ForW.reference(
#         connectivities={dims.C2E2CODim: c2e2co_np},
#         w=w_np.copy(),
#         geofac_n2s=geofac_n2s_np
#     )

#     # Cast to Cartesian Fields
#     Kolor = getattr(dims, "Kolor", getattr(dims, "KolorDim", gtx.Dimension("Kolor")))
#     w_f = gtx.as_field([dims.IDim, dims.JDim, Kolor, dims.KDim], pack_cell_field(w_np, ijk_to_cell))
#     geofac_n2s_f = tuple(gtx.as_field([dims.IDim, dims.JDim, Kolor], p) for p in geofac_n2s_s)
#     z_nabla2_c_f = gtx.as_field([dims.IDim, dims.JDim, Kolor, dims.KDim], np.zeros_like(w_f.asnumpy()))

#     selected_backend = gtfn_cpu
#     prog = setup_program(
#         calculate_nabla2_for_w_cart, backend=selected_backend,
#         horizontal_sizes={
#             "domain_min_i": gtx.int32(0), "domain_max_i": gtx.int32(ni), 
#             "domain_min_j": gtx.int32(0), "domain_max_j": gtx.int32(nj), 
#             "domain_max_kolor": gtx.int32(2)
#         },
#     )
    
#     if hasattr(prog, "_static_args_names"):
#         prog._static_args_names = set(prog._static_args_names) | {"domain_min_i", "domain_min_j"}

#     prog(
#         w=w_f, geofac_n2s=geofac_n2s_f, z_nabla2_c=z_nabla2_c_f,
#         domain_min_i=gtx.int32(0), domain_max_i=gtx.int32(ni), 
#         domain_min_j=gtx.int32(0), domain_max_j=gtx.int32(nj), 
#         domain_max_kolor=gtx.int32(2), 
#         vertical_start=gtx.int32(0), vertical_end=gtx.int32(num_levels), 
#         offset_provider={}
#     )

#     actual_z_nabla2_c = unpack_cell_field(z_nabla2_c_f.asnumpy(), ijk_to_cell, n_cells)
#     np.testing.assert_allclose(actual_z_nabla2_c, expected_output["z_nabla2_c"], rtol=1e-12, atol=0)