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
from dataclasses import replace


import icon4py.model.common.utils.data_allocation as data_alloc
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla4 import calculate_nabla4#, calculate_nabla4_cart
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.testing.stencil_tests import StandardStaticVariants, StencilTest


def calculate_nabla4_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    u_vert: np.ndarray,
    v_vert: np.ndarray,
    primal_normal_vert_v1: np.ndarray,
    primal_normal_vert_v2: np.ndarray,
    z_nabla2_e: np.ndarray,
    inv_vert_vert_length: np.ndarray,
    inv_primal_edge_length: np.ndarray,
) -> np.ndarray:
    e2c2v = connectivities[dims.E2C2VDim]
    u_vert_e2c2v = u_vert[e2c2v]
    v_vert_e2c2v = v_vert[e2c2v]

    primal_normal_vert_v1 = np.expand_dims(primal_normal_vert_v1, axis=-1)
    primal_normal_vert_v2 = np.expand_dims(primal_normal_vert_v2, axis=-1)
    inv_vert_vert_length = np.expand_dims(inv_vert_vert_length, axis=-1)
    inv_primal_edge_length = np.expand_dims(inv_primal_edge_length, axis=-1)

    nabv_tang = (
        u_vert_e2c2v[:, 0] * primal_normal_vert_v1[:, 0]
        + v_vert_e2c2v[:, 0] * primal_normal_vert_v2[:, 0]
    ) + (
        u_vert_e2c2v[:, 1] * primal_normal_vert_v1[:, 1]
        + v_vert_e2c2v[:, 1] * primal_normal_vert_v2[:, 1]
    )
    nabv_norm = (
        u_vert_e2c2v[:, 2] * primal_normal_vert_v1[:, 2]
        + v_vert_e2c2v[:, 2] * primal_normal_vert_v2[:, 2]
    ) + (
        u_vert_e2c2v[:, 3] * primal_normal_vert_v1[:, 3]
        + v_vert_e2c2v[:, 3] * primal_normal_vert_v2[:, 3]
    )
    z_nabla4_e2 = 4.0 * (
        (nabv_norm - 2.0 * z_nabla2_e) * inv_vert_vert_length**2
        + (nabv_tang - 2.0 * z_nabla2_e) * inv_primal_edge_length**2
    )
    return z_nabla4_e2


@pytest.mark.continuous_benchmarking
class TestCalculateNabla4(StencilTest):
    PROGRAM = calculate_nabla4
    OUTPUTS = ("z_nabla4_e2",)
    STATIC_PARAMS = {
        StandardStaticVariants.NONE: (),
        StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "horizontal_start",
            "horizontal_end",
            "vertical_start",
            "vertical_end",
        ),
        StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "vertical_start",
            "vertical_end",
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
        **kwargs,
    ) -> dict:
        z_nabla4_e2 = calculate_nabla4_numpy(
            connectivities,
            u_vert,
            v_vert,
            primal_normal_vert_v1,
            primal_normal_vert_v2,
            z_nabla2_e,
            inv_vert_vert_length,
            inv_primal_edge_length,
        )
        return dict(z_nabla4_e2=z_nabla4_e2)

    @pytest.fixture
    def input_data(self, grid) -> dict:
        u_vert = data_alloc.random_field(grid, dims.VertexDim, dims.KDim, dtype=ta.vpfloat)
        v_vert = data_alloc.random_field(grid, dims.VertexDim, dims.KDim, dtype=ta.vpfloat)

        primal_normal_vert_v1 = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2C2VDim, dtype=ta.wpfloat
        )
        primal_normal_vert_v2 = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2C2VDim, dtype=ta.wpfloat
        )

        z_nabla2_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        inv_vert_vert_length = data_alloc.random_field(grid, dims.EdgeDim, dtype=ta.wpfloat)
        inv_primal_edge_length = data_alloc.random_field(grid, dims.EdgeDim, dtype=ta.wpfloat)

        z_nabla4_e2 = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)

        return dict(
            u_vert=u_vert,
            v_vert=v_vert,
            primal_normal_vert_v1=primal_normal_vert_v1,
            primal_normal_vert_v2=primal_normal_vert_v2,
            z_nabla2_e=z_nabla2_e,
            inv_vert_vert_length=inv_vert_vert_length,
            inv_primal_edge_length=inv_primal_edge_length,
            z_nabla4_e2=z_nabla4_e2,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )


# import os
# import xarray as xr
# from gt4py.next.modules.translator import pack_edge_field, pack_vertex_field, unpack_edge_field, build_index_map_from_lonlat_e2v
# from gt4py.next.program_processors.runners.gtfn import run_gtfn_cached as gtfn_cpu
# from gt4py.next.program_processors.program_setup_utils import setup_program
# from icon4py.model.common import dimension as dims


# def build_e2c2v_and_pn(m, n_edges, pn_v1_np, pn_v2_np):
#     """Generates the e2c2v map perfectly aligned to our structured shifts."""
#     e2c2v = np.full((n_edges, 4), -1, dtype=np.int32)
#     ni, nj, _ = m.ijk_to_edge.shape

#     pn_v1_s = tuple(np.zeros((ni, nj, 3), dtype=np.float64) for _ in range(4))
#     pn_v2_s = tuple(np.zeros((ni, nj, 3), dtype=np.float64) for _ in range(4))

#     for i in range(ni):
#         for j in range(nj):
#             for k in range(3):
#                 e = m.ijk_to_edge[i, j, k]
#                 if e < 0:
#                     continue

#                 if k == 0:
#                     verts = [
#                         m.ij_to_vertex[i, j],
#                         m.ij_to_vertex[i, j+1] if j+1 < nj else -1,
#                         m.ij_to_vertex[i+1, j] if i+1 < ni else -1,
#                         m.ij_to_vertex[i-1, j+1] if (i > 0 and j+1 < nj) else -1
#                     ]
#                 elif k == 1:
#                     verts = [
#                         m.ij_to_vertex[i, j],
#                         m.ij_to_vertex[i+1, j] if i+1 < ni else -1,
#                         m.ij_to_vertex[i, j+1] if j+1 < nj else -1,
#                         m.ij_to_vertex[i+1, j-1] if (i+1 < ni and j > 0) else -1
#                     ]
#                 else: # k == 2
#                     verts = [
#                         m.ij_to_vertex[i, j+1],
#                         m.ij_to_vertex[i+1, j] if (i+1 < ni and j > 0) else -1,
#                         m.ij_to_vertex[i, j] if i+1 < ni else -1,
#                         m.ij_to_vertex[i+1, j+1] if j > 0 else -1
#                     ]

#                 e2c2v[e] = verts

#                 # Only assign normal vectors if the neighbor vertex actually exists
#                 for v_idx in range(4):
#                     if verts[v_idx] >= 0:
#                         pn_v1_s[v_idx][i, j, k] = pn_v1_np[e, v_idx]
#                         pn_v2_s[v_idx][i, j, k] = pn_v2_np[e, v_idx]

#     return e2c2v, pn_v1_s, pn_v2_s


# def test_calculate_nabla4_cartesian(backend="gtfn_cpu"):
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

#     # 1. SORT THE EDGES: Boundary First, Interior Second (Correct ICON Unstructured Order)
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

#     # 2. Generate Data
#     np.random.seed(42)
#     u_vert_np = np.random.rand(nodes_size, num_levels)
#     v_vert_np = np.random.rand(nodes_size, num_levels)
#     pn_v1_np = np.random.rand(n_edges, 4)
#     pn_v2_np = np.random.rand(n_edges, 4)
#     z_nabla2_e_np = np.random.rand(n_edges, num_levels)
#     inv_vv_len_np = np.random.rand(n_edges)
#     inv_pe_len_np = np.random.rand(n_edges)

#     e2c2v_np, pn_v1_s, pn_v2_s = build_e2c2v_and_pn(index_map, n_edges, pn_v1_np, pn_v2_np)

#     # Pad safe vertices for the unstructured numpy reference evaluation (-1 array fallback)
#     u_vert_safe = np.vstack([u_vert_np, np.zeros((1, num_levels))])
#     v_vert_safe = np.vstack([v_vert_np, np.zeros((1, num_levels))])

#     # 3. GET GROUND TRUTH: Official icon4py reference method
#     expected_output = calculate_nabla4_numpy(
#         connectivities={dims.E2C2VDim: e2c2v_np},
#         u_vert=u_vert_safe,
#         v_vert=v_vert_safe,
#         primal_normal_vert_v1=pn_v1_np,
#         primal_normal_vert_v2=pn_v2_np,
#         z_nabla2_e=z_nabla2_e_np,
#         inv_vert_vert_length=inv_vv_len_np,
#         inv_primal_edge_length=inv_pe_len_np,
#     )

#     # 4. Cast to Fields
#     Kolor = getattr(dims, "Kolor", getattr(dims, "KolorDim", gtx.Dimension("Kolor")))
#     u_vert_f = gtx.as_field([dims.IDim, dims.JDim, Kolor, dims.KDim], pack_vertex_field(u_vert_np, index_map))
#     v_vert_f = gtx.as_field([dims.IDim, dims.JDim, Kolor, dims.KDim], pack_vertex_field(v_vert_np, index_map))
#     pn_v1_f = tuple(gtx.as_field([dims.IDim, dims.JDim, Kolor], p) for p in pn_v1_s)
#     pn_v2_f = tuple(gtx.as_field([dims.IDim, dims.JDim, Kolor], p) for p in pn_v2_s)
#     z_nabla2_e_f = gtx.as_field([dims.IDim, dims.JDim, Kolor, dims.KDim], pack_edge_field(z_nabla2_e_np, index_map))
#     inv_vv_len_f = gtx.as_field([dims.IDim, dims.JDim, Kolor], pack_edge_field(inv_vv_len_np, index_map))
#     inv_pe_len_f = gtx.as_field([dims.IDim, dims.JDim, Kolor], pack_edge_field(inv_pe_len_np, index_map))
#     z_nabla4_e2_f = gtx.as_field([dims.IDim, dims.JDim, Kolor, dims.KDim], pack_edge_field(np.zeros_like(z_nabla2_e_np), index_map))

#     if backend == "gtfn_cpu":
#         selected_backend = gtfn_cpu
#     else:
#         raise ValueError(f"Backend {backend} not supported in this test.")

#     # 5. Set up with MIN static bounds to ensure pure int64 compilation inside the AST
#     prog = setup_program(
#         calculate_nabla4_cart,
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
#         z_nabla4_e2=z_nabla4_e2_f,
#         domain_min_i=gtx.int32(0), domain_max_i=gtx.int32(ni),
#         domain_min_j=gtx.int32(0), domain_max_j=gtx.int32(nj),
#         domain_max_kolor=gtx.int32(3),
#         vertical_start=gtx.int32(0), vertical_end=gtx.int32(num_levels),
#         offset_provider={}
#     )

#     actual_z_nabla4_np = unpack_edge_field(z_nabla4_e2_f.asnumpy(), index_map, n_edges)
#     np.testing.assert_allclose(actual_z_nabla4_np, expected_output, rtol=1e-12, atol=1e-14)