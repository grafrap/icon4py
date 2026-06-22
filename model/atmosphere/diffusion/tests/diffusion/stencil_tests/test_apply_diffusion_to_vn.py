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
from dataclasses import replace

from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_vn import apply_diffusion_to_vn
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StandardStaticVariants, StencilTest

from icon4py.model.common import type_alias as ta

from .test_apply_nabla2_and_nabla4_global_to_vn import apply_nabla2_and_nabla4_global_to_vn_numpy
from .test_apply_nabla2_and_nabla4_to_vn import apply_nabla2_and_nabla4_to_vn_numpy
from .test_apply_nabla2_to_vn_in_lateral_boundary import (
    apply_nabla2_to_vn_in_lateral_boundary_numpy,
)
from .test_calculate_nabla4 import calculate_nabla4_numpy


@pytest.mark.uses_concat_where
@pytest.mark.continuous_benchmarking
class TestApplyDiffusionToVn(StencilTest):
    PROGRAM = apply_diffusion_to_vn
    OUTPUTS = ("vn",)
    ENABLE_REFERENCE_TRANSLATION_FOR_STRUCTURED_BACKEND = True
    STATIC_PARAMS = {
        StandardStaticVariants.NONE: (),
        StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "horizontal_start",
            "horizontal_end",
            "start_2nd_nudge_line_idx_e",
            "vertical_start",
            "vertical_end",
            "limited_area",
        ),
        StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "vertical_start",
            "vertical_end",
            "limited_area",
        ),
    }

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        u_vert: np.ndarray,
        v_vert: np.ndarray,
        primal_normal_vert_v1: np.ndarray,
        primal_normal_vert_v2: np.ndarray,
        z_nabla2_e: np.ndarray,
        inv_vert_vert_length: np.ndarray,
        inv_primal_edge_length: np.ndarray,
        area_edge: np.ndarray,
        kh_smag_e: np.ndarray,
        diff_multfac_vn: np.ndarray,
        nudgecoeff_e: np.ndarray,
        vn: np.ndarray,
        nudgezone_diff: np.ndarray,
        fac_bdydiff_v: np.ndarray,
        start_2nd_nudge_line_idx_e: np.int32,
        limited_area: bool,
        **kwargs: Any,
    ):
        edge = np.arange(area_edge.shape[0])
        vn_cp = vn.copy()
        z_nabla4_e2 = calculate_nabla4_numpy(
            connectivities,
            u_vert,
            v_vert,
            primal_normal_vert_v1,
            primal_normal_vert_v2,
            z_nabla2_e,
            inv_vert_vert_length,
            inv_primal_edge_length,
            z_nabla4_e2=np.zeros_like(z_nabla2_e),
            horizontal_start=0,
            horizontal_end=z_nabla2_e.shape[0],
        )

        condition = start_2nd_nudge_line_idx_e <= edge[:, np.newaxis]

        if limited_area:
            vn = np.where(
                condition,
                apply_nabla2_and_nabla4_to_vn_numpy(
                    area_edge,
                    kh_smag_e,
                    z_nabla2_e,
                    z_nabla4_e2,
                    diff_multfac_vn,
                    nudgecoeff_e,
                    vn,
                    nudgezone_diff,
                ),
                apply_nabla2_to_vn_in_lateral_boundary_numpy(
                    z_nabla2_e, area_edge, vn, fac_bdydiff_v
                ),
            )
        else:
            vn = np.where(
                condition,
                apply_nabla2_and_nabla4_global_to_vn_numpy(
                    area_edge, kh_smag_e, z_nabla2_e, z_nabla4_e2, diff_multfac_vn, vn
                ),
                vn,
            )

        # restriction of execution domain
        vn[0 : kwargs["horizontal_start"], :] = vn_cp[0 : kwargs["horizontal_start"], :]
        vn[kwargs["horizontal_end"] :, :] = vn_cp[kwargs["horizontal_end"] :, :]

        return dict(vn=vn)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        u_vert = data_alloc.random_field(grid, dims.VertexDim, dims.KDim)
        v_vert = data_alloc.random_field(grid, dims.VertexDim, dims.KDim)

        primal_normal_vert_v1 = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2VDim)
        primal_normal_vert_v2 = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2VDim)

        inv_vert_vert_length = data_alloc.random_field(grid, dims.EdgeDim)
        inv_primal_edge_length = data_alloc.random_field(grid, dims.EdgeDim)

        area_edge = data_alloc.random_field(grid, dims.EdgeDim)
        kh_smag_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_nabla2_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        diff_multfac_vn = data_alloc.random_field(grid, dims.KDim)
        vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        nudgecoeff_e = data_alloc.random_field(grid, dims.EdgeDim)

        limited_area = grid.limited_area if hasattr(grid, "limited_area") else True
        fac_bdydiff_v = 5.0
        nudgezone_diff = 9.0

        edge_domain = h_grid.domain(dims.EdgeDim)
        start_2nd_nudge_line_idx_e = grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5))
        horizontal_end = grid.end_index(edge_domain(h_grid.Zone.LOCAL))

        return dict(
            u_vert=u_vert,
            v_vert=v_vert,
            primal_normal_vert_v1=primal_normal_vert_v1,
            primal_normal_vert_v2=primal_normal_vert_v2,
            z_nabla2_e=z_nabla2_e,
            inv_vert_vert_length=inv_vert_vert_length,
            inv_primal_edge_length=inv_primal_edge_length,
            area_edge=area_edge,
            kh_smag_e=kh_smag_e,
            diff_multfac_vn=diff_multfac_vn,
            nudgecoeff_e=nudgecoeff_e,
            vn=vn,
            nudgezone_diff=nudgezone_diff,
            fac_bdydiff_v=fac_bdydiff_v,
            start_2nd_nudge_line_idx_e=start_2nd_nudge_line_idx_e,
            limited_area=limited_area,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=grid.num_levels,
        )


# import os
# import pytest
# import numpy as np
# import xarray as xr
# import gt4py.next as gtx
# from gt4py.next.program_processors.program_setup_utils import setup_program
# from gt4py.next.program_processors.runners.gtfn import run_gtfn_cached as gtfn_cpu
# from gt4py.next.program_processors.runners.dace import run_dace_cpu

# from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_vn import apply_diffusion_to_vn_cart
# from icon4py.model.common import dimension as dims

# # Import your translator methods
# from gt4py.next.modules.translator import pack_edge_field, unpack_edge_field, build_index_map_from_lonlat_e2v, pack_vertex_field

# # ---------------------------------------------------------
# # Numpy References
# # ---------------------------------------------------------
# # def calculate_nabla4_numpy(e2c2v, u_vert, v_vert, pn_v1, pn_v2, z_nabla2_e, inv_vv_len, inv_pe_len):
# #     u_vert_e2c2v = u_vert[e2c2v]
# #     v_vert_e2c2v = v_vert[e2c2v]
# #     pn_v1 = np.expand_dims(pn_v1, axis=-1)
# #     pn_v2 = np.expand_dims(pn_v2, axis=-1)
# #     inv_vv_len = np.expand_dims(inv_vv_len, axis=-1)
# #     inv_pe_len = np.expand_dims(inv_pe_len, axis=-1)
# #     nabv_tang = (u_vert_e2c2v[:, 0] * pn_v1[:, 0] + v_vert_e2c2v[:, 0] * pn_v2[:, 0]) + (u_vert_e2c2v[:, 1] * pn_v1[:, 1] + v_vert_e2c2v[:, 1] * pn_v2[:, 1])
# #     nabv_norm = (u_vert_e2c2v[:, 2] * pn_v1[:, 2] + v_vert_e2c2v[:, 2] * pn_v2[:, 2]) + (u_vert_e2c2v[:, 3] * pn_v1[:, 3] + v_vert_e2c2v[:, 3] * pn_v2[:, 3])
# #     return 4.0 * ((nabv_norm - 2.0 * z_nabla2_e) * inv_vv_len**2 + (nabv_tang - 2.0 * z_nabla2_e) * inv_pe_len**2)

# # def apply_nabla2_and_nabla4_to_vn_numpy(area_edge, kh_smag_e, z_nabla2_e, z_nabla4_e2, diff_multfac_vn, nudgecoeff_e, vn, nudgezone_diff):
# #     area_edge = np.expand_dims(area_edge, axis=-1)
# #     diff_multfac_vn = np.expand_dims(diff_multfac_vn, axis=0)
# #     nudgecoeff_e = np.expand_dims(nudgecoeff_e, axis=-1)
# #     return vn + area_edge * (np.maximum(nudgezone_diff * nudgecoeff_e, kh_smag_e) * z_nabla2_e - diff_multfac_vn * z_nabla4_e2 * area_edge)

# # def apply_nabla2_to_vn_in_lateral_boundary_numpy(z_nabla2_e, area_edge, vn, fac_bdydiff_v):
# #     return vn + (z_nabla2_e * np.expand_dims(area_edge, axis=-1) * fac_bdydiff_v)

# # def apply_diffusion_to_vn_numpy(
# #     e2c2v_np, u_vert, v_vert, pn_v1, pn_v2, z_nabla2_e, inv_vv_len, inv_pe_len,
# #     area_edge, kh_smag_e, diff_multfac_vn, nudgecoeff_e, vn, nudgezone_diff, fac_bdydiff_v,
# #     is_interior_edge_np
# # ):
# #     z_nabla4_e2 = calculate_nabla4_numpy(e2c2v_np, u_vert, v_vert, pn_v1, pn_v2, z_nabla2_e, inv_vv_len, inv_pe_len)
# #     vn_interior = apply_nabla2_and_nabla4_to_vn_numpy(area_edge, kh_smag_e, z_nabla2_e, z_nabla4_e2, diff_multfac_vn, nudgecoeff_e, vn, nudgezone_diff)
# #     vn_boundary = apply_nabla2_to_vn_in_lateral_boundary_numpy(z_nabla2_e, area_edge, vn, fac_bdydiff_v)
# #     return np.where(is_interior_edge_np[:, None], vn_interior, vn_boundary)


# def build_e2c2v_and_pn(m, n_edges, pn_v1_np, pn_v2_np):
#     e2c2v = np.full((n_edges, 4), -1, dtype=np.int32)
#     ni, nj, _ = m.ijk_to_edge.shape
#     pn_v1_s = tuple(np.zeros((ni, nj, 3), dtype=np.float64) for _ in range(4))
#     pn_v2_s = tuple(np.zeros((ni, nj, 3), dtype=np.float64) for _ in range(4))

#     for i in range(ni):
#         for j in range(nj):
#             for k in range(3):
#                 e = m.ijk_to_edge[i, j, k]
#                 if e < 0: continue
#                 if k == 0:
#                     verts = [m.ij_to_vertex[i, j], m.ij_to_vertex[i, j+1] if j+1 < nj else -1, m.ij_to_vertex[i+1, j] if i+1 < ni else -1, m.ij_to_vertex[i-1, j+1] if (i > 0 and j+1 < nj) else -1]
#                 elif k == 1:
#                     verts = [m.ij_to_vertex[i, j], m.ij_to_vertex[i+1, j] if i+1 < ni else -1, m.ij_to_vertex[i, j+1] if j+1 < nj else -1, m.ij_to_vertex[i+1, j-1] if (i+1 < ni and j > 0) else -1]
#                 else:
#                     verts = [m.ij_to_vertex[i, j+1], m.ij_to_vertex[i+1, j] if (i+1 < ni and j > 0) else -1, m.ij_to_vertex[i, j] if i+1 < ni else -1, m.ij_to_vertex[i+1, j+1] if j > 0 else -1]
#                 e2c2v[e] = verts
#                 for v_idx in range(4):
#                     if verts[v_idx] >= 0:
#                         pn_v1_s[v_idx][i, j, k] = pn_v1_np[e, v_idx]
#                         pn_v2_s[v_idx][i, j, k] = pn_v2_np[e, v_idx]
#     return e2c2v, pn_v1_s, pn_v2_s


# def test_apply_diffusion_to_vn_cartesian(backend="gtfn_cpu"):
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
#     n_edges = ds.sizes["edge"]
#     num_levels = 10

#     index_map = build_index_map_from_lonlat_e2v(lonlat, e2v, nodes_size=nodes_size)
#     ni, nj = index_map.ij_to_vertex.shape

#     # 1. SORT THE EDGES: Boundary First, Interior Second
#     is_interior_old = np.zeros(n_edges, dtype=bool)
#     for i in range(ni):
#         for j in range(nj):
#             for k in range(3):
#                 e = index_map.ijk_to_edge[i, j, k]
#                 if e >= 0:
#                     if (4 <= i < ni - 4) and (4 <= j < nj - 4):
#                         is_interior_old[e] = True

#     boundary_indices = np.where(~is_interior_old)[0]
#     interior_indices = np.where(is_interior_old)[0]

#     # Map Boundary -> [0..B], Map Interior -> [B..N]
#     old_to_new = np.empty(n_edges, dtype=np.int32)
#     old_to_new[boundary_indices] = np.arange(len(boundary_indices))
#     old_to_new[interior_indices] = np.arange(len(boundary_indices), n_edges)

#     new_ijk_to_edge = np.where(index_map.ijk_to_edge >= 0, old_to_new[index_map.ijk_to_edge], -1)
#     index_map = replace(index_map, ijk_to_edge=new_ijk_to_edge)

#     start_2nd_nudge_line_idx_e = len(boundary_indices)

#     # 2. Generate Data
#     np.random.seed(42)
#     u_vert_np = np.random.rand(nodes_size, num_levels)
#     v_vert_np = np.random.rand(nodes_size, num_levels)
#     pn_v1_np = np.random.rand(n_edges, 4)
#     pn_v2_np = np.random.rand(n_edges, 4)
#     z_nabla2_e_np = np.random.rand(n_edges, num_levels)
#     inv_vv_len_np = np.random.rand(n_edges)
#     inv_pe_len_np = np.random.rand(n_edges)
#     area_edge_np = np.random.rand(n_edges)
#     kh_smag_e_np = np.random.rand(n_edges, num_levels)
#     diff_multfac_vn_np = np.random.rand(num_levels)
#     nudgecoeff_e_np = np.random.rand(n_edges)
#     vn_np = np.random.rand(n_edges, num_levels)
#     nudgezone_diff = ta.vpfloat("9.0")
#     fac_bdydiff_v = ta.wpfloat("5.0")

#     e2c2v_np, pn_v1_s, pn_v2_s = build_e2c2v_and_pn(index_map, n_edges, pn_v1_np, pn_v2_np)

#     u_vert_safe = np.vstack([u_vert_np, np.zeros((1, num_levels))])
#     v_vert_safe = np.vstack([v_vert_np, np.zeros((1, num_levels))])

#     # 3. GET GROUND TRUTH (Assumes TestApplyDiffusionToVn is imported/available in this file)
#     from .test_apply_diffusion_to_vn import TestApplyDiffusionToVn  # Adjust import if needed
    
#     expected_output = TestApplyDiffusionToVn.reference(
#         connectivities={dims.E2C2VDim: e2c2v_np},
#         u_vert=u_vert_safe,
#         v_vert=v_vert_safe,
#         primal_normal_vert_v1=pn_v1_np,
#         primal_normal_vert_v2=pn_v2_np,
#         z_nabla2_e=z_nabla2_e_np,
#         inv_vert_vert_length=inv_vv_len_np,
#         inv_primal_edge_length=inv_pe_len_np,
#         area_edge=area_edge_np,
#         kh_smag_e=kh_smag_e_np,
#         diff_multfac_vn=diff_multfac_vn_np,
#         nudgecoeff_e=nudgecoeff_e_np,
#         vn=vn_np.copy(),
#         nudgezone_diff=nudgezone_diff,
#         fac_bdydiff_v=fac_bdydiff_v,
#         start_2nd_nudge_line_idx_e=start_2nd_nudge_line_idx_e,
#         limited_area=True,
#         horizontal_start=0,
#         horizontal_end=n_edges,
#         vertical_start=0,
#         vertical_end=num_levels,
#     )
#     expected_vn = expected_output["vn"]

#     # 4. Cast to Fields
#     Kolor = getattr(dims, "Kolor", getattr(dims, "KolorDim", gtx.Dimension("Kolor")))
#     u_vert_f = gtx.as_field([dims.IDim, dims.JDim, Kolor, dims.KDim], pack_vertex_field(u_vert_np, index_map))
#     v_vert_f = gtx.as_field([dims.IDim, dims.JDim, Kolor, dims.KDim], pack_vertex_field(v_vert_np, index_map))
#     pn_v1_f = tuple(gtx.as_field([dims.IDim, dims.JDim, Kolor], p) for p in pn_v1_s)
#     pn_v2_f = tuple(gtx.as_field([dims.IDim, dims.JDim, Kolor], p) for p in pn_v2_s)
#     z_nabla2_e_f = gtx.as_field([dims.IDim, dims.JDim, Kolor, dims.KDim], pack_edge_field(z_nabla2_e_np, index_map))
#     inv_vv_len_f = gtx.as_field([dims.IDim, dims.JDim, Kolor], pack_edge_field(inv_vv_len_np, index_map))
#     inv_pe_len_f = gtx.as_field([dims.IDim, dims.JDim, Kolor], pack_edge_field(inv_pe_len_np, index_map))
#     area_edge_f = gtx.as_field([dims.IDim, dims.JDim, Kolor], pack_edge_field(area_edge_np, index_map))
#     kh_smag_e_f = gtx.as_field([dims.IDim, dims.JDim, Kolor, dims.KDim], pack_edge_field(kh_smag_e_np, index_map))
#     diff_multfac_vn_f = gtx.as_field([dims.KDim], diff_multfac_vn_np)
#     nudgecoeff_e_f = gtx.as_field([dims.IDim, dims.JDim, Kolor], pack_edge_field(nudgecoeff_e_np, index_map))
#     vn_f = gtx.as_field([dims.IDim, dims.JDim, Kolor, dims.KDim], pack_edge_field(vn_np, index_map))

#     selected_backend = gtfn_cpu

#     prog = setup_program(
#         apply_diffusion_to_vn_cart,
#         backend=selected_backend,
#         horizontal_sizes={
#             "domain_min_i": gtx.int32(0), "domain_max_i": gtx.int32(ni),
#             "domain_min_j": gtx.int32(0), "domain_max_j": gtx.int32(nj),
#             "domain_max_kolor": gtx.int32(3)
#         },
#     )

#     if hasattr(prog, "_static_args_names"):
#         prog._static_args_names = set(prog._static_args_names) | {"domain_min_i", "domain_min_j"}

#     prog(
#         u_vert=u_vert_f, v_vert=v_vert_f, pn_v1=pn_v1_f, pn_v2=pn_v2_f,
#         z_nabla2_e=z_nabla2_e_f, inv_vert_vert_length=inv_vv_len_f, inv_primal_edge_length=inv_pe_len_f,
#         area_edge=area_edge_f, kh_smag_e=kh_smag_e_f, diff_multfac_vn=diff_multfac_vn_f, nudgecoeff_e=nudgecoeff_e_f,
#         vn=vn_f, nudgezone_diff=nudgezone_diff, fac_bdydiff_v=fac_bdydiff_v,
#         domain_min_i=gtx.int32(0), domain_max_i=gtx.int32(ni),
#         domain_min_j=gtx.int32(0), domain_max_j=gtx.int32(nj),
#         domain_max_kolor=gtx.int32(3),
#         vertical_start=gtx.int32(0), vertical_end=gtx.int32(num_levels),
#         offset_provider={}
#     )

#     actual_vn_np = unpack_edge_field(vn_f.asnumpy(), index_map, n_edges)
#     np.testing.assert_allclose(actual_vn_np, expected_vn, rtol=1e-12, atol=0)