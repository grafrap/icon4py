# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import numpy as np
from gt4py.next import neighbor_sum
from gt4py.next.experimental import concat_where

from icon4py.model.common import type_alias as ta
from icon4py.model.common.dimension import IDim, JDim, Kolor


# KolorOff = gtx.FieldOffset("KolorOff", source=Kolor, target=(Kolor))


@gtx.field_operator
def compute_zavgS_cartesian_0(
    pp: gtx.Field[[IDim, JDim, Kolor], float],
    S_M: gtx.Field[[IDim, JDim, Kolor], float],
    domain_max_j: gtx.int32,
) -> gtx.Field[[IDim, JDim, Kolor], float]:
    zavg = 0.5 * concat_where(
        JDim == domain_max_j - 1,
        pp,
        pp + pp(JDim + 1))
    # zavg = 0.5 * (pp + pp(JDim + 1))
    return S_M * zavg


@gtx.field_operator
def compute_zavgS_cartesian_1(
    pp: gtx.Field[[IDim, JDim, Kolor], float],
    S_M: gtx.Field[[IDim, JDim, Kolor], float],
    domain_max_i: gtx.int32,
) -> gtx.Field[[IDim, JDim, Kolor], float]:
    zavg = 0.5 * concat_where(
        IDim == domain_max_i - 1,
        pp(Kolor-1),
        pp(Kolor-1) + pp(IDim + 1)(Kolor-1))
    # zavg = 0.5 * (pp + pp(IDim + 1))
    return S_M * zavg


@gtx.field_operator
def compute_zavgS_cartesian_2(
    pp: gtx.Field[[IDim, JDim, Kolor], float],
    S_M: gtx.Field[[IDim, JDim, Kolor], float],
    domain_max_i: gtx.int32,
    domain_max_j: gtx.int32,
) -> gtx.Field[[IDim, JDim, Kolor], float]:
    zavg = 0.5 * concat_where(
        IDim == domain_max_i - 1, concat_where(
            JDim == domain_max_j - 1,
            0.0,
            pp(Kolor-2) + pp(JDim + 1)(Kolor-2)),
        concat_where(JDim == domain_max_j - 1,
            pp(IDim + 1)(Kolor-2), pp(IDim + 1)(Kolor-2) + pp(JDim + 1)(Kolor-2))
	)
    
    # zavg = 0.5 * (pp(IDim + 1) + pp(JDim + 1))
    return S_M * zavg

@gtx.field_operator
def on_edges(
    f0: gtx.Field[[IDim, JDim, Kolor], float],
    f1: gtx.Field[[IDim, JDim, Kolor], float],
    f2: gtx.Field[[IDim, JDim, Kolor], float],
) -> gtx.Field[[IDim, JDim, Kolor], float]:
    return concat_where(
        Kolor == 0, 
        f0,
        concat_where(Kolor == 1, f1, f2),
    )

@gtx.field_operator
def compute_zavgS_cartesian(
    pp: gtx.Field[[IDim, JDim, Kolor], float],
    S_M: gtx.Field[[IDim, JDim, Kolor], float],
    domain_max_i: gtx.int32,
    domain_max_j: gtx.int32,
) -> gtx.Field[[IDim, JDim, Kolor], float]:

	return on_edges(
		compute_zavgS_cartesian_0(pp, S_M, domain_max_j),
		compute_zavgS_cartesian_1(pp, S_M, domain_max_i),
		compute_zavgS_cartesian_2(pp, S_M, domain_max_i, domain_max_j),
	)

# @gtx.program
# def zavg(
#     pp: gtx.Field[[IDim, JDim, Kolor], float],
#     S_M: gtx.Field[[IDim, JDim, Kolor], float],
#     out: gtx.Field[[IDim, JDim, Kolor], float],
#     domain_max_i: gtx.int32,
#     domain_max_j: gtx.int32,
#     domain_max_kolor: gtx.int32,

# ):
#     compute_zavgS_cartesian(
#         pp,
#         S_M,
#         domain_max_i,
#         domain_max_j,
#         out=out,
#         domain={IDim: (0, domain_max_i), JDim: (0, domain_max_j)},
# 	)


@gtx.field_operator
def _compute_pnabla_cartesian(
	pp: gtx.Field[gtx.Dims[IDim, JDim, Kolor], ta.wpfloat],
	S_M: gtx.Field[gtx.Dims[IDim, JDim, Kolor], ta.wpfloat],
	sign: gtx.Field[gtx.Dims[IDim, JDim, Kolor], ta.wpfloat],
	vol: gtx.Field[gtx.Dims[IDim, JDim, Kolor], ta.wpfloat],
	domain_max_i: gtx.int32,
    domain_max_j: gtx.int32,
) -> gtx.Field[gtx.Dims[IDim, JDim, Kolor],	 ta.wpfloat]:
	zavg_s = compute_zavgS_cartesian(pp, S_M, domain_max_i, domain_max_j)
	pnabla_color = concat_where(
		Kolor == 0,
		concat_where(JDim == 0, zavg_s, zavg_s + zavg_s(JDim - 1)),
		concat_where(
			Kolor == 1,
			concat_where(IDim == 0, zavg_s, zavg_s + zavg_s(IDim - 1)),
			concat_where(
					IDim == 0,
					concat_where(JDim == 0, 0.0, zavg_s(JDim - 1)),
					concat_where(JDim == 0, zavg_s(IDim - 1), zavg_s(IDim - 1) + zavg_s(JDim - 1)),
			),
		),
	)
	pnabla_color_signed = pnabla_color * sign
	return neighbor_sum(pnabla_color_signed, axis=Kolor) / vol

@gtx.field_operator
def _compute_pnabla_cartesian_direct(
	pp: gtx.Field[gtx.Dims[IDim, JDim, Kolor], ta.wpfloat],
	S_M: gtx.Field[gtx.Dims[IDim, JDim, Kolor], ta.wpfloat],
	sign: gtx.Field[gtx.Dims[IDim, JDim, Kolor], ta.wpfloat],
	vol: gtx.Field[gtx.Dims[IDim, JDim, Kolor], ta.wpfloat],
	domain_max_i: gtx.int32,
    domain_max_j: gtx.int32,
) -> gtx.Field[gtx.Dims[IDim, JDim, Kolor],	 ta.wpfloat]:
	zavg_s = compute_zavgS_cartesian(pp, S_M, domain_max_i, domain_max_j)
	pnabla_color = concat_where(
		Kolor == 0,
		concat_where(JDim == 0, zavg_s, zavg_s + zavg_s(JDim - 1)),
		concat_where(
			Kolor == 1,
			concat_where(IDim == 0, zavg_s, zavg_s + zavg_s(IDim - 1)),
			concat_where(
					IDim == 0,
					concat_where(JDim == 0, 0.0, zavg_s(JDim - 1)),
					concat_where(JDim == 0, zavg_s(IDim - 1), zavg_s(IDim - 1) + zavg_s(JDim - 1)),
			),
		),
	)
	pnabla_color_signed = pnabla_color * sign
	# return (concat_where(Kolor==0, pnabla_color_signed + pnabla_color_signed(Kolor + 1) + pnabla_color_signed(Kolor +2), 0.0)) / vol
	return (pnabla_color_signed + pnabla_color_signed(Kolor + 1) + pnabla_color_signed(Kolor + 2)) / vol

# field(Kolor[0]) + field(Kolor[1]) + field(Kolor[2])
# domain_max_j: gtx.int32,
# ) -> gtx.Field[gtx.Dims[IDim, JDim], ta.wpfloat]:
# 	zavg_s = _compute_zavg_s_cartesian(pp, S_M, domain_max_i, domain_max_j)

# 	zavg_0 = concat_where(Kolor == 0, zavg_s, 0.0)
# 	zavg_1 = concat_where(Kolor == 1, zavg_s, 0.0)
# 	zavg_2 = concat_where(Kolor == 2, zavg_s, 0.0)

# 	pnabla_0 = concat_where(JDim == 0, zavg_0, zavg_0 + zavg_0(JDim - 1))
# 	pnabla_1 = concat_where(IDim == 0, zavg_1, zavg_1 + zavg_1(IDim - 1))
# 	pnabla_2 = concat_where(
# 		IDim == 0,
# 		concat_where(JDim == 0, 0.0, zavg_2(JDim - 1)),
# 		concat_where(JDim == 0, zavg_2(IDim - 1), zavg_2(IDim - 1) + zavg_2(JDim - 1)),
# 	)

# 	pnabla_color = pnabla_0 + pnabla_1 + pnabla_2

@gtx.program
def compute_pnabla_cartesian(
	pp: gtx.Field[gtx.Dims[IDim, JDim, Kolor], ta.wpfloat],
	S_M: gtx.Field[gtx.Dims[IDim, JDim, Kolor], ta.wpfloat],
	sign: gtx.Field[gtx.Dims[IDim, JDim, Kolor], ta.wpfloat],
	vol: gtx.Field[gtx.Dims[IDim, JDim, Kolor], ta.wpfloat],
	out: gtx.Field[gtx.Dims[IDim, JDim, Kolor], ta.wpfloat],
	domain_max_i: gtx.int32,
	domain_max_j: gtx.int32,
	domain_max_kolor: gtx.int32,
):
	_compute_pnabla_cartesian( # _direct(
		pp,
		S_M,
		sign,
		vol,
		domain_max_i,
		domain_max_j,
		out=out,
		domain={
			IDim: (0, domain_max_i),
			JDim: (0, domain_max_j),
			Kolor: (0, domain_max_kolor),
		},
	)


def _run_demo() -> None:
	from icon4py.model.common import model_backends
	from icon4py.model.common.model_options import customize_backend
	from gt4py.next.program_processors.program_setup_utils import setup_program
	import xarray as xr 

	# Load the dataset
	ds = xr.open_dataset("/home/raphael/Documents/Studium/Msc_thesis/grid-generator/parallelogram_grid.nc")
	# raw_edges = ds['edge_data'].values # The big 1D array

	# Define your grid sizes (from your description)
	nx = np.int32(ds.attrs['domain_length'] / ds.attrs['mean_edge_length'])
	ny = np.int32((ds.sizes['cell'])/(2*nx))
	max_j = np.int32(nx + 1)
	max_i = np.int32(ny + 1)
	# raw_edges = np.linspace(0, nx*(ny+1) + (nx+1)*ny + nx*ny - 1, nx*(ny+1) + (nx+1)*ny + nx*ny, dtype=np.float64)
	raw_edges = np.ones((nx*(ny+1) + (nx+1)*ny + nx*ny,), dtype=np.float64)
	raw_vertices = np.linspace(0, (nx+1)*(ny+1) - 1, (nx+1)*(ny+1), dtype=np.float64)
	# Calculate start/end indices for each block
	# 1. East Edges: nx * (ny + 1)
	count_east = nx * (ny + 1)
	edges_east = raw_edges[0 : count_east]

	# 2. NE Edges: (nx + 1) * ny
	count_ne = (nx + 1) * ny
	edges_ne = raw_edges[count_east : count_east + count_ne]

	# 3. SE Edges: nx * ny
	count_se = nx * ny
	edges_se = raw_edges[count_east + count_ne : ]

	edges_east_2d = edges_east.reshape((ny+1, nx))  # Shape: [nx, ny+1]
	edges_ne_2d   = edges_ne.reshape((ny, nx+1))    # Shape: [nx+1, ny]
	edges_se_2d   = edges_se.reshape((ny, nx))        # Shape: [nx, ny]

	print("Edges East:", edges_east, "\n2D version:\n", edges_east_2d)

	pp_2d = raw_vertices.reshape((ny + 1, nx + 1, 1))    # Shape: [nx+1, ny+1, 1]

	# To stack them in one field

	S_M_field = np.zeros((max_i, max_j, 3)) # [IDim, JDim, Kolor]

	# Fill Kolor 0 (East) - fits in [0:nx, 0:ny+1]
	S_M_field[0:ny+1, 0:nx, 0] = edges_east_2d

	# Fill Kolor 1 (NE) - fits in [0:nx+1, 0:ny]
	S_M_field[0:ny, 0:nx+1, 1] = edges_ne_2d

	# Fill Kolor 2 (SE) - fits in [0:nx, 0:ny]
	S_M_field[0:ny, 0:nx, 2]   = edges_se_2d

	# Prepare Vertices (pp)
	pp_field = np.zeros((max_i, max_j, 1))
	pp_field[:, :,:] = pp_2d

	pp = gtx.as_field([IDim, JDim, Kolor], pp_2d)
	S_M = gtx.as_field([IDim, JDim, Kolor], S_M_field)

	vol = np.ones((max_i, max_j, 1))
	sign = np.ones((max_i, max_j, 3))
	vol_field = gtx.as_field([IDim, JDim, Kolor], vol)
	sign_field = gtx.as_field([IDim, JDim, Kolor], sign)

	pnabla_out = gtx.as_field([IDim, JDim,Kolor], np.zeros((max_i, max_j,1)))
	from gt4py.next.program_processors.runners.dace import run_dace_cpu
	backend = customize_backend(compute_zavgS_cartesian, model_backends.BACKENDS["gtfn_cpu"])
	# run_zavg = setup_program(
	# 	program=compute_zavgS_cartesian,
	# 	backend=run_dace_cpu,
	# 	horizontal_sizes={
	# 		"domain_max_i": max_i,
	# 		"domain_max_j": max_j,
	# 	},
	# )
	run_pnabla = setup_program(
		program=compute_pnabla_cartesian,
		backend=backend,
		horizontal_sizes={
			"domain_max_i": max_i,
			"domain_max_j": max_j,
			"domain_max_kolor": 1,
		},
	)

	used_backend = "gtfn_cpu"

	# run_zavg(pp=pp, s_m=S_M, zavg_s=)
	run_pnabla(
		pp,
		S_M,
		sign=sign_field,
		vol=vol_field,
		out=pnabla_out,
		domain_max_kolor=gtx.int32(1),
	)

	print("=== simple_structured demo ===")
	print("backend:", used_backend)
	print("shape(pp):", pp.shape)
	print("shape(s_m):", S_M.shape)
	print("pnabla:\n", pnabla_out.asnumpy()[:,:,0])


if __name__ == "__main__":
	_run_demo()


