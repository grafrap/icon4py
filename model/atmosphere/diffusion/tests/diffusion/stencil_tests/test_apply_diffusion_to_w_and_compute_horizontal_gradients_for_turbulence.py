# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import os
import xarray as xr
import gt4py.next as gtx
import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32
from gt4py.next.program_processors.program_setup_utils import setup_program
from gt4py.next.program_processors.runners.gtfn import run_gtfn_cached as gtfn_cpu

from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence import (
    apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence,
    # apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence_cart,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing import definitions
from icon4py.model.testing.stencil_tests import StandardStaticVariants, StencilTest

from .test_apply_nabla2_to_w import apply_nabla2_to_w_numpy
from .test_apply_nabla2_to_w_in_upper_damping_layer import (
    apply_nabla2_to_w_in_upper_damping_layer_numpy,
)
from .test_calculate_horizontal_gradients_for_turbulence import (
    calculate_horizontal_gradients_for_turbulence_numpy,
)
from .test_calculate_nabla2_for_w import calculate_nabla2_for_w_numpy
from gt4py.next.modules.translator import (
    build_index_map_from_lonlat_e2v,
    build_cell_to_ijk,
    pack_cell_field,
    unpack_cell_field,
    build_c2e2co_unstructured,
    pack_c2e2co_field
)


@pytest.mark.embedded_remap_error
@pytest.mark.continuous_benchmarking
class TestApplyDiffusionToWAndComputeHorizontalGradientsForTurbulence(StencilTest):
    PROGRAM = apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence
    OUTPUTS = ("w", "dwdx", "dwdy")
    STATIC_PARAMS = {
        StandardStaticVariants.NONE: (),
        StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "horizontal_start",
            "horizontal_end",
            "halo_idx",
            "interior_idx",
            "vertical_start",
            "vertical_end",
            "nrdmax",
            "type_shear",
        ),
        StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "vertical_start",
            "vertical_end",
            "nrdmax",
            "type_shear",
        ),
    }

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        area,
        geofac_n2s,
        geofac_grg_x,
        geofac_grg_y,
        w_old,
        type_shear,
        dwdx,
        dwdy,
        diff_multfac_w,
        diff_multfac_n2w,
        nrdmax,
        interior_idx,
        halo_idx,
        horizontal_start,
        horizontal_end,
        vertical_start,
        vertical_end,
        **kwargs,
    ) -> dict:
        k = np.arange(w_old.shape[1])
        cell = np.arange(w_old.shape[0])
        reshaped_k = k[np.newaxis, :]
        reshaped_cell = cell[:, np.newaxis]
        out_w, out_dwdx, out_dwdy = (
            np.zeros_like(w_old),
            dwdx.copy(),
            dwdy.copy(),
        )  # create output arrays to update only the necessary slices
        if type_shear == 2:
            dwdx, dwdy = np.where(
                reshaped_k > 0,
                calculate_horizontal_gradients_for_turbulence_numpy(
                    connectivities, w_old, geofac_grg_x, geofac_grg_y
                ),
                (dwdx, dwdy),
            )

        z_nabla2_c = calculate_nabla2_for_w_numpy(connectivities, w_old, geofac_n2s)

        w = np.where(
            (interior_idx <= reshaped_cell) & (reshaped_cell < halo_idx),
            apply_nabla2_to_w_numpy(
                connectivities, area, z_nabla2_c, geofac_n2s, w_old, diff_multfac_w
            ),
            w_old,
        )

        w = np.where(
            (reshaped_k > 0)
            & (reshaped_k < nrdmax)
            & (interior_idx <= reshaped_cell)
            & (reshaped_cell < halo_idx),
            apply_nabla2_to_w_in_upper_damping_layer_numpy(w, diff_multfac_n2w, area, z_nabla2_c),
            w,
        )
        subset = (slice(horizontal_start, horizontal_end), slice(vertical_start, vertical_end))
        out_w[subset] = w[subset]
        out_dwdx[subset] = dwdx[subset]
        out_dwdy[subset] = dwdy[subset]
        return dict(w=out_w, dwdx=out_dwdx, dwdy=out_dwdy)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        nrdmax = 13
        cell_domain = h_grid.domain(dims.CellDim)
        interior_idx = grid.start_index(cell_domain(h_grid.Zone.INTERIOR))  # 0 for simple grid
        halo_idx = grid.end_index(
            cell_domain(h_grid.Zone.LOCAL)
        )  # same as horizontal_end for simple grid
        type_shear = 2

        def _get_start_index_for_w_diffusion() -> int32:
            return (
                grid.start_index(cell_domain(h_grid.Zone.NUDGING))
                if grid.limited_area
                else grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4))
            )

        horizontal_start = _get_start_index_for_w_diffusion()
        horizontal_end = grid.end_index(cell_domain(h_grid.Zone.HALO))

        geofac_grg_x = random_field(grid, dims.CellDim, dims.C2E2CODim)
        geofac_grg_y = random_field(grid, dims.CellDim, dims.C2E2CODim)
        diff_multfac_n2w = random_field(grid, dims.KDim)
        area = random_field(grid, dims.CellDim)
        geofac_n2s = random_field(grid, dims.CellDim, dims.C2E2CODim)
        w_old = random_field(grid, dims.CellDim, dims.KDim)
        diff_multfac_w = 5.0

        w = zero_field(grid, dims.CellDim, dims.KDim)
        dwdx = random_field(grid, dims.CellDim, dims.KDim)
        dwdy = random_field(grid, dims.CellDim, dims.KDim)

        return dict(
            area=area,
            geofac_n2s=geofac_n2s,
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
            w_old=w_old,
            type_shear=type_shear,
            diff_multfac_w=diff_multfac_w,
            diff_multfac_n2w=diff_multfac_n2w,
            nrdmax=nrdmax,
            interior_idx=interior_idx,
            halo_idx=halo_idx,
            w=w,
            dwdx=dwdx,
            dwdy=dwdy,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=grid.num_levels,
        )

# def test_apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence_cartesian(backend="gtfn_cpu"):
#     mesh_nc = os.environ.get(
#         "GT4PY_TRANSLATOR_MESH", 
#         "/home/raphael/Documents/Studium/Msc_thesis/grid-generator/parallelogram_grid.nc"
#     )
#     if not os.path.exists(mesh_nc):
#         pytest.skip(f"Mesh file {mesh_nc} not found.")

#     ds = xr.open_dataset(mesh_nc)
#     e2v = np.where(ds["edge_vertices"].transpose("edge", "nc").values.astype(np.int32) > 0, 
#                    ds["edge_vertices"].transpose("edge", "nc").values.astype(np.int32) - 1, -1)
#     lonlat = np.stack([ds["longitude_vertices"].values, ds["latitude_vertices"].values], axis=1).astype(np.float64)

#     nodes_size = ds.sizes["vertex"]
#     n_cells = ds.sizes["cell"]
#     num_levels = 10

#     index_map = build_index_map_from_lonlat_e2v(lonlat, e2v, nodes_size=nodes_size)
#     ni, nj = index_map.ij_to_vertex.shape
#     ijk_to_cell = build_cell_to_ijk(index_map, ds)

#     # 1. SORT CELLS: Interior cells MUST be FIRST, Margin (Halo) cells SECOND!
#     is_damping_old = np.zeros(n_cells, dtype=bool)
#     for i in range(ni):
#         for j in range(nj):
#             for k in range(2):
#                 c = ijk_to_cell[i, j, k]
#                 if c >= 0 and not (4 <= i < ni - 4 and 4 <= j < nj - 4):
#                     is_damping_old[c] = True

#     halo_old = np.where(is_damping_old)[0]
#     interior_old = np.where(~is_damping_old)[0]

#     new_to_old = np.concatenate([interior_old, halo_old])
#     old_to_new = np.empty(n_cells, dtype=np.int32)
#     old_to_new[new_to_old] = np.arange(n_cells)

#     interior_idx = 0
#     halo_idx = len(interior_old)

#     # 2. Build sorted unstructured topology for the reference function
#     c2e2co_old = build_c2e2co_unstructured(ijk_to_cell, n_cells)
#     c2e2co_new = np.where(c2e2co_old >= 0, old_to_new[c2e2co_old], -1)
#     c2e2co_sorted = c2e2co_new[new_to_old]

#     # 3. Generate Random Sorted Data
#     np.random.seed(42)
#     area_sorted = np.random.rand(n_cells)
#     w_old_sorted = np.random.rand(n_cells, num_levels)
#     w_sorted = np.random.rand(n_cells, num_levels)
#     geofac_n2s_sorted = np.random.rand(n_cells, 3)
#     geofac_grg_x_sorted = np.random.rand(n_cells, 3)
#     geofac_grg_y_sorted = np.random.rand(n_cells, 3)
    
#     dwdx_sorted = np.zeros((n_cells, num_levels))
#     dwdy_sorted = np.zeros((n_cells, num_levels))
    
#     diff_multfac_n2w = np.random.rand(num_levels)
#     diff_multfac_w = wpfloat("5.0")
#     type_shear = gtx.int32(2)  # MUST BE 2 to compute horizontal gradients!
#     nrdmax = gtx.int32(5)

#     # 4. GET GROUND TRUTH: Official icon4py reference method
#     expected_output = TestApplyDiffusionToWAndComputeHorizontalGradientsForTurbulence.reference(
#         connectivities={dims.C2E2CODim: c2e2co_sorted, dims.C2E2CDim: c2e2co_sorted},
#         area=area_sorted, geofac_n2s=geofac_n2s_sorted,
#         geofac_grg_x=geofac_grg_x_sorted, geofac_grg_y=geofac_grg_y_sorted,
#         w_old=w_old_sorted, w=w_sorted.copy(),
#         type_shear=type_shear, dwdx=dwdx_sorted, dwdy=dwdy_sorted,
#         diff_multfac_w=diff_multfac_w, diff_multfac_n2w=diff_multfac_n2w,
#         nrdmax=nrdmax, interior_idx=interior_idx, halo_idx=halo_idx,
#         horizontal_start=0, horizontal_end=n_cells,
#         vertical_start=0, vertical_end=num_levels
#     )

#     # 5. Reverse sort variables to restore original Cartesian layout order
#     area_old = area_sorted[old_to_new]
#     w_old_old = w_old_sorted[old_to_new]
#     w_old_val = w_sorted[old_to_new]
#     geofac_n2s_old = geofac_n2s_sorted[old_to_new]
#     geofac_grg_x_old = geofac_grg_x_sorted[old_to_new]
#     geofac_grg_y_old = geofac_grg_y_sorted[old_to_new]

#     # 6. Use translator.py to pack all the data flawlessly!
#     Kolor = getattr(dims, "Kolor", getattr(dims, "KolorDim", gtx.Dimension("Kolor")))
#     area_f = gtx.as_field([dims.IDim, dims.JDim, Kolor], pack_cell_field(area_old, ijk_to_cell))
#     w_old_f = gtx.as_field([dims.IDim, dims.JDim, Kolor, dims.KDim], pack_cell_field(w_old_old, ijk_to_cell))
#     w_f = gtx.as_field([dims.IDim, dims.JDim, Kolor, dims.KDim], pack_cell_field(w_old_val, ijk_to_cell))
    
#     geofac_n2s_s = pack_c2e2co_field(geofac_n2s_old, ijk_to_cell)
#     geofac_grg_x_s = pack_c2e2co_field(geofac_grg_x_old, ijk_to_cell)
#     geofac_grg_y_s = pack_c2e2co_field(geofac_grg_y_old, ijk_to_cell)

#     geofac_n2s_f = tuple(gtx.as_field([dims.IDim, dims.JDim, Kolor], p) for p in geofac_n2s_s)
#     geofac_grg_x_f = tuple(gtx.as_field([dims.IDim, dims.JDim, Kolor], p) for p in geofac_grg_x_s)
#     geofac_grg_y_f = tuple(gtx.as_field([dims.IDim, dims.JDim, Kolor], p) for p in geofac_grg_y_s)
    
#     dwdx_f = gtx.as_field([dims.IDim, dims.JDim, Kolor, dims.KDim], np.zeros_like(w_f.asnumpy()))
#     dwdy_f = gtx.as_field([dims.IDim, dims.JDim, Kolor, dims.KDim], np.zeros_like(w_f.asnumpy()))
#     diff_multfac_n2w_f = gtx.as_field([dims.KDim], diff_multfac_n2w)

#     selected_backend = gtfn_cpu
#     prog = setup_program(
#         apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence_cart,
#         backend=selected_backend,
#         horizontal_sizes={"domain_min_i": gtx.int32(0), "domain_max_i": gtx.int32(ni), "domain_min_j": gtx.int32(0), "domain_max_j": gtx.int32(nj), "domain_max_kolor": gtx.int32(2)},
#     )

#     if hasattr(prog, "_static_args_names"):
#         prog._static_args_names = set(prog._static_args_names) | {"domain_min_i", "domain_min_j"}

#     prog(
#         area=area_f, geofac_n2s=geofac_n2s_f, geofac_grg_x=geofac_grg_x_f, geofac_grg_y=geofac_grg_y_f,
#         w_old=w_old_f, w=w_f, type_shear=type_shear, dwdx=dwdx_f, dwdy=dwdy_f,
#         diff_multfac_w=diff_multfac_w, diff_multfac_n2w=diff_multfac_n2w_f, nrdmax=nrdmax,
#         domain_min_i=gtx.int32(0), domain_max_i=gtx.int32(ni),
#         domain_min_j=gtx.int32(0), domain_max_j=gtx.int32(nj),
#         domain_max_kolor=gtx.int32(2),
#         vertical_start=gtx.int32(0), vertical_end=gtx.int32(num_levels),
#         offset_provider={}
#     )

#     # 7. Validate
#     actual_w = unpack_cell_field(w_f.asnumpy(), ijk_to_cell, n_cells)[new_to_old]
#     actual_dwdx = unpack_cell_field(dwdx_f.asnumpy(), ijk_to_cell, n_cells)[new_to_old]
#     actual_dwdy = unpack_cell_field(dwdy_f.asnumpy(), ijk_to_cell, n_cells)[new_to_old]

#     np.testing.assert_allclose(actual_w, expected_output["w"], rtol=1e-12, atol=0)
#     np.testing.assert_allclose(actual_dwdx, expected_output["dwdx"], rtol=1e-12, atol=0)
#     np.testing.assert_allclose(actual_dwdy, expected_output["dwdy"], rtol=1e-12, atol=0)