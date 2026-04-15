# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.model.testing.fixtures.datatest import backend_like
from icon4py.model.testing.fixtures.stencil_tests import grid, grid_manager

import os
import pytest

from gt4py.next.program_processors.runners import gtfn as gtfn_runner
from gt4py.next.iterator.transforms.cart_unroll import CartesianDomainAndTypeRemapper
from gt4py.next.modules.cartesian_interceptor import GenericStructuredWrapper, get_global_grid_mapping

@pytest.fixture
def _configured_program(request, grid):
    """
    Overrides icon4py's internal _configured_program fixture.
    This injects our Cartesian wrapper right before execution, bypassing the caching pool!
    """
    # 1. Extract the operator (stencil) from the icon4py test instance
    operator = request.instance.stencil

    print(f"\n=== CONFIGURING PROGRAM FOR OPERATOR: {operator.name} ===")
    
    use_structured = os.environ.get("USE_STRUCTURED_BACKEND", "1") == "1"
    
    if use_structured:
        if getattr(grid, "id", None) == "simple_grid":
            raise RuntimeError(
                "Structured backend is disabled for 'simple_grid'. "
                "Use a structured-compatible grid file via '--grid <ICON_GRID_FILE>:<levels>' "
                "or disable USE_STRUCTURED_BACKEND."
            )
        print(f"\n---> INTERCEPTING FIXTURE FOR: {operator.name} <---")
        
        # 2. Get the globally cached mapping
        e2v_conn = grid.connectivities.get("E2V")
        e2v_array = e2v_conn.asnumpy() if e2v_conn is not None else None
        index_map, remap_sizes = get_global_grid_mapping(e2v_override=e2v_array)
        
        # 3. Inject sizes explicitly into the compiler pass ClassVars!
        CartesianDomainAndTypeRemapper.MAX_I = int(remap_sizes.max_i)
        CartesianDomainAndTypeRemapper.MAX_J = int(remap_sizes.max_j)
        
        # 4. Create and return our magic wrapper
        wrapper = GenericStructuredWrapper(
            operator=operator,
            backend_factory=gtfn_runner.GTFNBackendFactory,
            index_map=index_map,
            remap_sizes=remap_sizes,
            allocator=None, # We extract this during __call__ dynamically
            offset_provider=grid.connectivities
        )
        return wrapper
        
    # Fallback to standard unstructured execution
    from gt4py.next.program_processors.program_setup_utils import setup_program
    backend = request.getfixturevalue("backend")
    return setup_program(operator, backend=backend, offset_provider=grid.connectivities)