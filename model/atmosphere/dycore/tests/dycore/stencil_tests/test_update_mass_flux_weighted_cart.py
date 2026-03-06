import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.update_mass_flux_weighted import (
    update_mass_flux_weighted_cart,
    IDim,
    JDim,
    Kolor,
)
from icon4py.model.common import dimension as dims

def test_update_mass_flux_weighted_cart():
    nx, ny, nz = 10, 10, 5
    color_size = 2
    
    # Using simple numpy backend for functional verification
    from gt4py.next.program_processors.runners.dace import run_dace_cpu

    backend = run_dace_cpu

    # Shape: (IDim, JDim, Kolor, KDim) -> (nx, ny, 2, nz)
    shape_k = (nx, ny, color_size, nz)
    shape_no_k = (nx, ny, color_size)

    rho_ic = np.random.rand(*shape_k).astype(np.float64)
    vwind_expl_wgt = np.random.rand(*shape_no_k).astype(np.float64)
    vwind_impl_wgt = np.random.rand(*shape_no_k).astype(np.float64)
    w_now = np.random.rand(*shape_k).astype(np.float64)
    w_new = np.random.rand(*shape_k).astype(np.float64)
    # w_concorr_c is vpfloat, assuming it behaves like float for this test
    w_concorr_c = np.random.rand(*shape_k).astype(np.float64)
    mass_flx_ic = np.random.rand(*shape_k).astype(np.float64)
    
    r_nsubsteps = 0.5

    # Broadcast weights to K dimension for calculation
    vwind_expl_wgt_k = vwind_expl_wgt[..., np.newaxis]
    vwind_impl_wgt_k = vwind_impl_wgt[..., np.newaxis]
    
    expected = mass_flx_ic + (
        r_nsubsteps * rho_ic * (
            vwind_expl_wgt_k * w_now + 
            vwind_impl_wgt_k * w_new - 
            w_concorr_c
        )
    )

    # We copy mass_flx_ic because 'out' argument modifies it in-place
    mass_flx_ic_gt = gtx.as_field((IDim, JDim, Kolor, dims.KDim), mass_flx_ic, allocator=backend.allocator)
    rho_ic_gt = gtx.as_field((IDim, JDim, Kolor, dims.KDim), rho_ic, allocator=backend.allocator)
    w_now_gt = gtx.as_field((IDim, JDim, Kolor, dims.KDim), w_now, allocator=backend.allocator)
    w_new_gt = gtx.as_field((IDim, JDim, Kolor, dims.KDim), w_new, allocator=backend.allocator)
    w_concorr_c_gt = gtx.as_field((IDim, JDim, Kolor, dims.KDim), w_concorr_c, allocator=backend.allocator)
    
    # These fields lack K dim
    vwind_expl_wgt_gt = gtx.as_field((IDim, JDim, Kolor), vwind_expl_wgt, allocator=backend.allocator)
    vwind_impl_wgt_gt = gtx.as_field((IDim, JDim, Kolor), vwind_impl_wgt, allocator=backend.allocator)
    from gt4py.next.program_processors.runners.dace import run_dace_cpu

    update_mass_flux_weighted_cart.with_backend(run_dace_cpu)(
        rho_ic=rho_ic_gt,
        vwind_expl_wgt=vwind_expl_wgt_gt,
        vwind_impl_wgt=vwind_impl_wgt_gt,
        w_now=w_now_gt,
        w_new=w_new_gt,
        w_concorr_c=w_concorr_c_gt,
        mass_flx_ic=mass_flx_ic_gt,
        r_nsubsteps=r_nsubsteps,
        horizontal_start=0,
        horizontal_end=nx,
        vertical_start=0,
        vertical_end=nz,
        offset_provider={} # No neighbors accessed
    )

    assert np.allclose(mass_flx_ic_gt.asnumpy(), expected)

if __name__ == "__main__":
    test_update_mass_flux_weighted_cart()