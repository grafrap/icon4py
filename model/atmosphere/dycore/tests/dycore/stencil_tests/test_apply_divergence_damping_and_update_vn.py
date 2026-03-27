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
import os

import icon4py.model.common.type_alias as ta
import icon4py.model.testing.stencil_tests as test_helpers
from icon4py.model.atmosphere.dycore.dycore_states import DivergenceDampingOrder
from icon4py.model.atmosphere.dycore.stencils.compute_edge_diagnostics_for_dycore_and_update_vn import (
    apply_divergence_damping_and_update_vn,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc

from . import test_dycore_utils


divergence_damp_order = DivergenceDampingOrder()


@pytest.mark.embedded_remap_error
@pytest.mark.continuous_benchmarking
class TestApplyDivergenceDampingAndUpdateVn(test_helpers.StencilTest):
    PROGRAM = apply_divergence_damping_and_update_vn
    OUTPUTS = ("next_vn",)
    STATIC_PARAMS = {
        test_helpers.StandardStaticVariants.NONE: (),
        test_helpers.StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "horizontal_start",
            "horizontal_end",
            "vertical_start",
            "vertical_end",
            "is_iau_active",
            "limited_area",
        ),
        test_helpers.StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "vertical_start",
            "vertical_end",
            "is_iau_active",
            "limited_area",
        ),
    }

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        horizontal_gradient_of_normal_wind_divergence: np.ndarray,
        next_vn: np.ndarray,
        current_vn: np.ndarray,
        dwdz_at_cells_on_model_levels: np.ndarray,
        predictor_normal_wind_advective_tendency: np.ndarray,
        corrector_normal_wind_advective_tendency: np.ndarray,
        normal_wind_tendency_due_to_slow_physics_process: np.ndarray,
        normal_wind_iau_increment: np.ndarray,
        theta_v_at_edges_on_model_levels: np.ndarray,
        horizontal_pressure_gradient: np.ndarray,
        second_order_divdamp_scaling_coeff: ta.wpfloat,
        horizontal_mask_for_3d_divdamp: np.ndarray,
        scaling_factor_for_3d_divdamp: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        nudgecoeff_e: np.ndarray,
        geofac_grdiv: np.ndarray,
        interpolated_fourth_order_divdamp_factor: np.ndarray,
        advection_explicit_weight_parameter: ta.wpfloat,
        advection_implicit_weight_parameter: ta.wpfloat,
        dtime: ta.wpfloat,
        iau_wgt_dyn: ta.wpfloat,
        is_iau_active: bool,
        limited_area: bool,
        apply_2nd_order_divergence_damping: bool,
        apply_4th_order_divergence_damping: bool,
        divdamp_order: gtx.int32,
        mean_cell_area: float,
        second_order_divdamp_factor: float,
        max_nudging_coefficient: float,
        dbl_eps: float,
        horizontal_start: gtx.int32 | tuple[gtx.int32, gtx.int32],
        horizontal_end: gtx.int32 | tuple[gtx.int32, gtx.int32],
        vertical_start: gtx.int32,
        vertical_end: gtx.int32,
    ) -> dict:
        
        is_structured = os.environ.get("USE_STRUCTURED_BACKEND", "0") == "1"
        
        # 1. DYNAMICALLY BUILD THE MASK
        lateral_margin = int(os.environ.get("GT4PY_TRANSLATOR_LATERAL", "5"))
        nx = 0
        ny = 0
        if lateral_margin > 0:
            nx = int((horizontal_start - lateral_margin + 1) / lateral_margin)
            ny = int((horizontal_end - nx) / (3 * nx + 1))
        if is_structured and lateral_margin > 0:
            
            # Use your exact math to infer nx and ny from the 1D boundaries
            
            start_i, start_j = lateral_margin, lateral_margin
            end_i, end_j = ny + 1 - lateral_margin, nx + 1 - lateral_margin

            total_edges = current_vn.shape[0]
            mask_1d = np.zeros(total_edges, dtype=bool)
            
            # Reconstruct the 1D indices of the INTERIOR edges based on your layout
            # for i in range(start_i, end_i):
            #     for j in range(start_j, end_j):
            #         for e_type in range(3):
            #             idx = (i * ny) + (j * 3) + e_type 
            #             if idx < total_edges:
            #                 mask_1d[idx] = True
            # east edges:
            for i in range(start_i, end_i):
                for j in range(start_j, end_j-1):
                    idx = (i * nx) + j
                    if idx < total_edges:
                        mask_1d[idx] = True
            
            # northeast edges:
            for i in range(start_i, end_i-1):
                for j in range(start_j, end_j):
                    idx = (i * (nx+1)) + j + (nx * ny) + nx
                    if idx < total_edges:
                        mask_1d[idx] = True
            
            # southeast edges
            for i in range(start_i, end_i-1):
                for j in range(start_j, end_j-1):
                    idx = (i * nx) + j + (2 * nx * ny) + ny + nx
                    if idx < total_edges:
                        mask_1d[idx] = True
            # Expand for the K dimension
            compute_mask = mask_1d[:, np.newaxis]
        else: 
            horz_idx = np.arange(horizontal_end)[:, np.newaxis]
            compute_mask = (horizontal_start <= horz_idx) & (horz_idx < horizontal_end)

        print(f"compute_mask: ", compute_mask[:,0])
        scaling_factor_for_3d_divdamp = np.expand_dims(scaling_factor_for_3d_divdamp, axis=0)
        horizontal_mask_for_3d_divdamp = np.expand_dims(horizontal_mask_for_3d_divdamp, axis=-1)
        inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, axis=-1)

        e2c = connectivities[dims.E2CDim]
        dwdz_at_edges_on_model_levels = dwdz_at_cells_on_model_levels[e2c]
        weighted_dwdz_at_edges_on_model_levels = (
            dwdz_at_edges_on_model_levels[:, 1] - dwdz_at_edges_on_model_levels[:, 0]
        )

        # Adjust for new e2c layout: multiply first 13*17 and last 13*16 entries by -1
        weighted_dwdz_at_edges_on_model_levels[:nx*(ny+1), :] *= -1
        weighted_dwdz_at_edges_on_model_levels[(2*nx*ny) + ny + nx:, :] *= -1

        horizontal_gradient_of_total_divergence = horizontal_gradient_of_normal_wind_divergence + (
            horizontal_mask_for_3d_divdamp
            * scaling_factor_for_3d_divdamp
            * inv_dual_edge_length
            * weighted_dwdz_at_edges_on_model_levels
        )

        # 2. USE THE MASK FOR THE CALCULATIONS
        next_vn = np.where(
           compute_mask,
            current_vn
            + dtime
            * (
                advection_explicit_weight_parameter * predictor_normal_wind_advective_tendency
                + advection_implicit_weight_parameter * corrector_normal_wind_advective_tendency
                + normal_wind_tendency_due_to_slow_physics_process
                - constants.CPD * theta_v_at_edges_on_model_levels * horizontal_pressure_gradient
            ),
            next_vn,
        )

        if apply_2nd_order_divergence_damping:
            next_vn = np.where(
                compute_mask,
                next_vn
                + (second_order_divdamp_scaling_coeff * horizontal_gradient_of_total_divergence),
                next_vn,
            )

        if apply_4th_order_divergence_damping:
            e2c2eO = connectivities[dims.E2C2EODim]
            squared_horizontal_gradient_of_total_divergence = np.where(
                compute_mask,
                np.sum(
                    np.where(
                        (e2c2eO != -1)[:, :, np.newaxis],
                        horizontal_gradient_of_total_divergence[e2c2eO]
                        * np.expand_dims(geofac_grdiv, axis=-1),
                        0,
                    ),
                    axis=1,
                ),
                np.zeros_like(horizontal_gradient_of_total_divergence),
            )
            fourth_order_divdamp_scaling_coeff = (
                test_dycore_utils.fourth_order_divdamp_scaling_coeff_numpy(
                    interpolated_fourth_order_divdamp_factor, divdamp_order, second_order_divdamp_factor, mean_cell_area
                )
            )
            reduced_fourth_order_divdamp_coeff_at_nest_boundary = test_dycore_utils.calculate_reduced_fourth_order_divdamp_coeff_at_nest_boundary_numpy(
                fourth_order_divdamp_scaling_coeff, max_nudging_coefficient
            )
            if limited_area:
                next_vn = np.where(
                    compute_mask,
                    next_vn
                    + (fourth_order_divdamp_scaling_coeff + reduced_fourth_order_divdamp_coeff_at_nest_boundary * np.expand_dims(nudgecoeff_e, axis=-1))
                    * squared_horizontal_gradient_of_total_divergence,
                    next_vn,
                )
            else:
                next_vn = np.where(
                    compute_mask,
                    next_vn + (np.expand_dims(fourth_order_divdamp_scaling_coeff, axis=0) * squared_horizontal_gradient_of_total_divergence),
                    next_vn,
                )

        if is_iau_active:
            next_vn = np.where(
                compute_mask,
                next_vn + (iau_wgt_dyn * normal_wind_iau_increment),
                next_vn,
            )

        return dict(next_vn=next_vn)

    @pytest.fixture(
        params=[
            {"divdamp_order": do, "is_iau_active": ia, "second_order_divdamp_factor": sodf}
            for do, ia, sodf in [
                (
                    DivergenceDampingOrder.COMBINED,
                    True,
                    0.012,
                ),  # For testing the whole functionality of the stencil
                (
                    DivergenceDampingOrder.COMBINED,
                    False,
                    0.032,
                ),  # For benchmarking against MCH experiments
            ]
        ],
        ids=lambda param: f"divdamp_order[{param['divdamp_order']}]__is_iau_active[{param['is_iau_active']}]__second_order_divdamp_factor[{param['second_order_divdamp_factor']}]",
    )
    def input_data(self, request: pytest.FixtureRequest, grid: base.Grid) -> dict:
        current_vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        horizontal_mask_for_3d_divdamp = data_alloc.random_field(grid, dims.EdgeDim)
        scaling_factor_for_3d_divdamp = data_alloc.random_field(grid, dims.KDim)
        dwdz_at_cells_on_model_levels = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        inv_dual_edge_length = data_alloc.random_field(grid, dims.EdgeDim)
        corrector_normal_wind_advective_tendency = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim
        )
        predictor_normal_wind_advective_tendency = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim
        )
        normal_wind_tendency_due_to_slow_physics_process = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim
        )
        horizontal_gradient_of_normal_wind_divergence = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim
        )
        normal_wind_iau_increment = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        next_vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        theta_v_at_edges_on_model_levels = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        horizontal_pressure_gradient = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        geofac_grdiv = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2EODim)
        interpolated_fourth_order_divdamp_factor = data_alloc.random_field(grid, dims.KDim)
        nudgecoeff_e = data_alloc.random_field(grid, dims.EdgeDim)

        mean_cell_area = 1000.0
        max_nudging_coefficient = 0.3
        dbl_eps = constants.DBL_EPS

        dtime = 0.9
        advection_implicit_weight_parameter = 0.75
        advection_explicit_weight_parameter = 0.25
        iau_wgt_dyn = 1.0
        is_iau_active = request.param["is_iau_active"]
        fourth_order_divdamp_factor = 0.004
        second_order_divdamp_factor = request.param["second_order_divdamp_factor"]
        divdamp_order = request.param["divdamp_order"]
        second_order_divdamp_scaling_coeff = 34497.62082646618  # for icon-ch1(_medium)
        apply_2nd_order_divergence_damping = (divdamp_order == divergence_damp_order.COMBINED) and (
            second_order_divdamp_scaling_coeff > 1.0e-6
        )
        apply_4th_order_divergence_damping = (
            divdamp_order == divergence_damp_order.FOURTH_ORDER
        ) or (
            (divdamp_order == divergence_damp_order.COMBINED)
            and (second_order_divdamp_factor <= (4.0 * fourth_order_divdamp_factor))
        )

        limited_area = grid.limited_area if hasattr(grid, "limited_area") else True
        edge_domain = h_grid.domain(dims.EdgeDim)

        start_edge_nudging_level_2 = grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))
        # start_edge_nudging_level_2 = 0
        end_edge_local = grid.end_index(edge_domain(h_grid.Zone.LOCAL))

        monkeypatch = request.getfixturevalue("monkeypatch")
        lateral_margin = 5 # edge field, 4 lateral layers + 1 nudging layer
        monkeypatch.setenv("GT4PY_TRANSLATOR_LATERAL", str(lateral_margin))
        # is_structured = os.environ.get("USE_STRUCTURED_BACKEND", "1") == "1"

        # # Detect if we are on a structured grid by checking the rank of the allocated field
        # print(f"Current vn shape: {current_vn.shape}")
        # if is_structured:
        #     nx = int((start_edge_nudging_level_2 - lateral_margin + 1)/lateral_margin)
        #     ny = int((end_edge_local - nx) / (3*nx + 1))
            
        #     start_i, start_j = lateral_margin, lateral_margin
        #     end_i, end_j = ny + 1 - lateral_margin, nx + 1 - lateral_margin

        #     # Build a 1D boolean mask of False values
        #     total_edges = current_vn.shape[0]
        #     compute_mask = np.zeros(total_edges, dtype=bool)
            
        #     # Reconstruct the 1D indices of the INTERIOR edges based on your mapping layout.
        #     # (You will need to adjust the line below to match EXACTLY how your 
        #     # parallelogram_grid.nc maps (i, j, edge_type) to the 1D index E)
        #     for i in range(start_i, end_i):
        #         for j in range(start_j, end_j):
        #             # Assuming you have 3 edges per cell (horizontal, vertical, diagonal)
        #             for e_type in range(3):
                        
        #                 # Replace this pseudo-code with your actual 1D index mapping formula
        #                 idx = (i * ny * 3) + (j * 3) + e_type 
                        
        #                 if idx < total_edges:
        #                     compute_mask[idx] = True
        # print(f"Using horizontal_start: {start_edge_nudging_level_2}, horizontal_end: {end_edge_local} for structured grid.")
        print(f"next_vn before the test stencil: {next_vn.asnumpy()[:,0]}")
        return dict(
            horizontal_gradient_of_normal_wind_divergence=horizontal_gradient_of_normal_wind_divergence,
            next_vn=next_vn,
            current_vn=current_vn,
            dwdz_at_cells_on_model_levels=dwdz_at_cells_on_model_levels,
            predictor_normal_wind_advective_tendency=predictor_normal_wind_advective_tendency,
            corrector_normal_wind_advective_tendency=corrector_normal_wind_advective_tendency,
            normal_wind_tendency_due_to_slow_physics_process=normal_wind_tendency_due_to_slow_physics_process,
            normal_wind_iau_increment=normal_wind_iau_increment,
            theta_v_at_edges_on_model_levels=theta_v_at_edges_on_model_levels,
            horizontal_pressure_gradient=horizontal_pressure_gradient,
            second_order_divdamp_scaling_coeff=second_order_divdamp_scaling_coeff,
            horizontal_mask_for_3d_divdamp=horizontal_mask_for_3d_divdamp,
            scaling_factor_for_3d_divdamp=scaling_factor_for_3d_divdamp,
            inv_dual_edge_length=inv_dual_edge_length,
            nudgecoeff_e=nudgecoeff_e,
            geofac_grdiv=geofac_grdiv,
            interpolated_fourth_order_divdamp_factor=interpolated_fourth_order_divdamp_factor,
            advection_explicit_weight_parameter=advection_explicit_weight_parameter,
            advection_implicit_weight_parameter=advection_implicit_weight_parameter,
            dtime=dtime,
            iau_wgt_dyn=iau_wgt_dyn,
            is_iau_active=is_iau_active,
            limited_area=limited_area,
            apply_2nd_order_divergence_damping=apply_2nd_order_divergence_damping,
            apply_4th_order_divergence_damping=apply_4th_order_divergence_damping,
            divdamp_order=divdamp_order,
            mean_cell_area=mean_cell_area,
            second_order_divdamp_factor=second_order_divdamp_factor,
            max_nudging_coefficient=max_nudging_coefficient,
            dbl_eps=dbl_eps,
            horizontal_start=start_edge_nudging_level_2,
            horizontal_end=end_edge_local,
            vertical_start=0,
            vertical_end=grid.num_levels,
        )
'''
horizontal_gradient_of_normal_wind_divergence
horizontal_mask_for_3d_divdamp
scaling_factor_for_3d_divdamp
inv_dual_edge_length
dwdz_at_cells_on_model_levels
'''
