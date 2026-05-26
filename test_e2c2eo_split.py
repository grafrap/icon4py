"""Tests for E2C2EO per-kolor split: _e2c2e_on_local_intermediate guard and
the _conn_name_and_is_edge_to_non_edge fix that enables current_kolor pass-through.

Root cause (bug being tested):
  For non-local-intermediate E2C2EO stencils (e.g. compute_avg_vn_and_graddiv_vn_and_vt),
  SetAtRemapper now per-kolor splits the SetAt into three (Kolor:[k,k+1)) SetAts.
  Inside each per-kolor SetAt, NeighborReductionUnroller visits with current_kolor=k.

  BUT: _conn_name_and_is_edge_to_non_edge returns is_ene=False for E2C2EO (because
  E2C2EO is in _EDGE_TO_EDGE_CONNECTIVITIES), so visit_FunCall passes current_kolor=None
  to _build_field_concat_where_from_branches for E2C2EO.  This generates the full
  3-branch concat_where inside a Kolor:[0,1) SetAt.

  After infer_program(keep_existing_domains=True) + prune_empty_concat_where, the
  Kolor:[1,2) and Kolor:[2,3) branches cannot be pruned (keep_existing_domains stops
  propagation into as_fieldop lambda bodies). DaCe allocates a transient with
  Kolor:0:0 (zero-sized Kolor dimension), causing:
    InvalidSDFGEdgeError: Dimensionality mismatch between src/dst subsets
    gtir_tmp_211(0) [0:17, 0:18, 0, ...] -> [0:17, 0:18, 0:0, ...]

Fix:
  _conn_name_and_is_edge_to_non_edge: E2C2EO branches are keyed by OUTPUT EDGE kolor
  (confirmed from map_dict.py: Kolor:[0,1) → shift Kolor:+2, etc.).  So E2C2EO should
  be treated the same as E2C and E2V — pass current_kolor through so _build_field_concat_where_from_branches
  path-1 selects the single correct shift directly, with no concat_where.

Run via srun (NOT on login node):
    srun --partition=debug --time=00:05:00 --uenv=icon/25.2:v3 --view=default \\
      bash -c 'cd /scratch/mch/rgraf/icon4py && .venv/bin/python test_e2c2eo_split.py'
"""

import sys
sys.path.insert(0, "/scratch/mch/rgraf/gt4py/src")
sys.path.insert(0, "/scratch/mch/rgraf/icon4py/model/common/src")

from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.transforms.structured_backend_passes import (
    _build_field_concat_where_from_branches,
    _conn_name_and_is_edge_to_non_edge,
    _e2c2e_on_local_intermediate,
)
from gt4py.next.iterator.transforms.map_dict import map_dict

IDim = common.Dimension("IDim", kind=common.DimensionKind.HORIZONTAL)
JDim = common.Dimension("JDim", kind=common.DimensionKind.HORIZONTAL)
Kolor = common.Dimension("Kolor", kind=common.DimensionKind.HORIZONTAL)
K = common.Dimension("K", kind=common.DimensionKind.VERTICAL)


def _outer_domain(k_lo, k_hi):
    """Cartesian domain with given Kolor range; IDim/JDim match 26x26 test grid interior."""
    return im.domain(
        common.GridType.CARTESIAN,
        {
            IDim: (itir.OffsetLiteral(value=4), itir.OffsetLiteral(value=22)),
            JDim: (itir.OffsetLiteral(value=4), itir.OffsetLiteral(value=22)),
            Kolor: (itir.OffsetLiteral(value=k_lo), itir.OffsetLiteral(value=k_hi)),
            K: (im.ref("vs"), im.ref("ve")),
        },
    )


def _e2c2eo_neighbors_fieldop(field_ref):
    """Construct the IR node that matches _e2c2e_on_local_intermediate's pattern:
    as_fieldop(λ it → neighbors(E2C2EO, it))(field_ref)
    """
    return im.as_fieldop(
        im.lambda_("it")(
            im.call("neighbors")(itir.OffsetLiteral(value="E2C2EO"), im.ref("it"))
        ),
        _outer_domain(0, 3),
    )(field_ref)


# ── Test 1: _e2c2e_on_local_intermediate detects lambda-bound intermediate ──────

def test_e2c2e_on_local_intermediate_detects_lambda_param():
    """E2C2EO on a lambda-bound local field must return True.

    Simulates: (λ local_h_grad → as_fieldop(λit → neighbors(E2C2EO, it))(local_h_grad))(...)
    This is the pattern for apply_divergence_damping_and_update_vn where
    horizontal_gradient_of_total_divergence is a lambda-bound intermediate.
    Per-kolor split must be BLOCKED for such stencils.
    """
    local_sym = im.ref("local_h_grad")
    fieldop = _e2c2eo_neighbors_fieldop(local_sym)
    # Wrap in a lambda that binds local_h_grad — making it a lambda parameter
    expr = itir.FunCall(
        fun=im.lambda_("local_h_grad")(fieldop),
        args=[im.ref("some_input")],
    )
    assert _e2c2e_on_local_intermediate(expr), (
        "_e2c2e_on_local_intermediate must return True when E2C2EO field is lambda-bound. "
        "This stencil must NOT be per-kolor split."
    )


# ── Test 2: _e2c2e_on_local_intermediate ignores program parameters ─────────────

def test_e2c2e_on_local_intermediate_ignores_program_param():
    """E2C2EO directly on a non-lambda-bound SymRef must return False.

    Simulates: as_fieldop(λit → neighbors(E2C2EO, it))(vn)
    where 'vn' is a program parameter — not declared as a lambda parameter anywhere.
    Per-kolor split is SAFE for such stencils (vn covers all 3 kolors).
    """
    vn = im.ref("vn")
    fieldop = _e2c2eo_neighbors_fieldop(vn)
    # No lambda wrapping 'vn' → 'vn' is not in any lambda_params set
    assert not _e2c2e_on_local_intermediate(fieldop), (
        "_e2c2e_on_local_intermediate must return False when E2C2EO field is a program "
        "parameter. This stencil IS safe to per-kolor split."
    )


# ── Test 3: _e2c2e_on_local_intermediate — iterator param is not a local field ──

def test_e2c2e_on_local_intermediate_iterator_param_not_flagged():
    """The lambda INSIDE as_fieldop (e.g. λ it → ...) has 'it' as param.
    'it' must NOT be mistaken for the accessed field.  Only the ARGUMENT to
    the outer as_fieldop (not the iterator) is checked against lambda_params.
    """
    # The iterator lambda has param 'it'.  The outer as_fieldop arg is 'vn'.
    # 'it' is in lambda_params, but 'vn' is NOT — so must return False.
    vn = im.ref("vn")
    fieldop = _e2c2eo_neighbors_fieldop(vn)
    # Sanity: 'it' is in lambda_params but should not cause a false positive
    assert not _e2c2e_on_local_intermediate(fieldop), (
        "The iterator parameter 'it' of the inner lambda must not be confused with "
        "the accessed field argument."
    )


# ── Test 4: _conn_name_and_is_edge_to_non_edge — E2C2EO must be is_ene=True ────

def test_conn_name_e2c2eo_is_edge_source_after_fix():
    """After fix: E2C2EO must return is_ene=True so current_kolor is passed through.

    Root of the dimensionality-mismatch bug:
      Before fix: is_ene=False → current_kolor=None passed to _build_field_concat_where_from_branches
      → full 3-branch concat_where generated inside Kolor:[0,1) SetAt
      → prune_empty_concat_where cannot prune Kolor:[1,2)/[2,3) branches
      → DaCe allocates Kolor:0:0 transient → Dimensionality mismatch

    After fix: is_ene=True → current_kolor=k passed through → single as_fieldop selected.

    The E2C2EO map_dict branches ARE keyed by output-edge kolor:
      slot 0: Kolor:[0,1) → shift Kolor:+2 (kolor-0 output reads kolor-2 source)
              Kolor:[1,2) → shift Kolor:-1 (kolor-1 output reads kolor-0 source)
              else        → shift Kolor:-1 (kolor-2 output reads kolor-1 source)
    So passing current_kolor=k correctly selects the single relevant shift.
    """
    for slot in range(4):
        key = (itir.OffsetLiteral(value="E2C2EO"), itir.OffsetLiteral(value=slot))
        conn_name, is_ene = _conn_name_and_is_edge_to_non_edge(key)
        assert conn_name.rstrip("ₒ") == "E2C2EO"
        assert is_ene, (
            f"E2C2EO slot {slot}: _conn_name_and_is_edge_to_non_edge must return "
            f"is_ene=True after fix (was False before). "
            f"E2C2EO branches are keyed by output-edge kolor — same as E2C, E2V — "
            f"so current_kolor must be passed through, not suppressed."
        )


# ── Test 5: Same for E2C2E ────────────────────────────────────────────────────

def test_conn_name_e2c2e_is_edge_source_after_fix():
    """E2C2E (used for tangential wind interpolation) must also return is_ene=True."""
    for slot in range(4):
        key = (itir.OffsetLiteral(value="E2C2E"), itir.OffsetLiteral(value=slot))
        if key not in map_dict:
            continue
        conn_name, is_ene = _conn_name_and_is_edge_to_non_edge(key)
        assert is_ene, (
            f"E2C2E slot {slot}: is_ene must be True after fix."
        )


# ── Test 6: _build_field_concat_where_from_branches — E2C2EO + current_kolor=k ─

def test_e2c2eo_build_branches_with_current_kolor_selects_single_shift():
    """_build_field_concat_where_from_branches with current_kolor=k must return a
    single as_fieldop (path 1), NOT a concat_where.

    This is the desired post-fix behavior: inside a per-kolor split SetAt (Kolor:[k,k+1)),
    the E2C2EO access resolves to a single deterministic shift with no runtime branching.
    """
    key = (itir.OffsetLiteral(value="E2C2EO"), itir.OffsetLiteral(value=0))
    entry = map_dict[key]
    assert entry["kind"] == "concat_where", "E2C2EO slot 0 must have concat_where branches"
    branches = entry["branches"]

    vn = im.ref("vn")
    # per-kolor split SetAt domain: Kolor:[0,1)
    domain = _outer_domain(0, 1)

    result = _build_field_concat_where_from_branches(
        vn, branches, domain,
        apply_edge_shape_bounds=False,
        current_kolor=0,
    )

    assert not cpm.is_call_to(result, "concat_where"), (
        "With current_kolor=0, E2C2EO must produce a SINGLE as_fieldop, NOT a concat_where. "
        "A concat_where here would cause prune_empty_concat_where to leave dead Kolor:[1,2) "
        "and Kolor:[2,3) branches, leading to zero-sized Kolor transients in DaCe."
    )
    assert cpm.is_applied_as_fieldop(result), (
        "Result must be an as_fieldop"
    )

    # For E2C2EO slot 0, kolor 0: expected shift is Kolor:+2
    # (output kolor-0 edge reads the neighbor at kolor-2 via E2C2EO)
    from gt4py.next.iterator.pretty_printer import pformat
    ir_str = pformat(result)
    assert "_OffKolor" in ir_str, f"Expected Kolor shift in result, got: {ir_str[:300]}"


def test_e2c2eo_build_branches_with_current_kolor_selects_correct_shift_per_kolor():
    """Verify the CORRECT shift is selected for each output kolor.

    From map_dict.py E2C2EO slot 0:
      kolor 0 → shift Kolor:+2   (reads source at kolor 2)
      kolor 1 → shift Kolor:-1   (reads source at kolor 0)
      kolor 2 → shift Kolor:-1   (reads source at kolor 1, via trailing-else)
    """
    key = (itir.OffsetLiteral(value="E2C2EO"), itir.OffsetLiteral(value=0))
    branches = map_dict[key]["branches"]
    vn = im.ref("vn")

    expected_kolor_shifts = {0: 2, 1: -1, 2: -1}

    for kolor, expected_dk in expected_kolor_shifts.items():
        domain = _outer_domain(kolor, kolor + 1)
        result = _build_field_concat_where_from_branches(
            vn, branches, domain,
            apply_edge_shape_bounds=False,
            current_kolor=kolor,
        )
        assert not cpm.is_call_to(result, "concat_where"), (
            f"kolor={kolor}: must produce single as_fieldop, not concat_where"
        )
        # Verify the kolor shift is embedded in the result
        from gt4py.next.iterator.pretty_printer import pformat
        ir_str = pformat(result)
        assert str(abs(expected_dk)) in ir_str, (
            f"kolor={kolor}: expected Kolor shift {expected_dk} in result, "
            f"got: {ir_str[:200]}"
        )


# ── Test 7: _build_field_concat_where_from_branches — E2C2EO + current_kolor=None ─

def test_e2c2eo_build_branches_without_current_kolor_generates_concat_where():
    """With current_kolor=None (non-split or local-intermediate path), the full
    3-branch concat_where must be generated.  This is the correct behavior for
    non-split SetAts (e.g. apply_divergence_damping_and_update_vn).
    """
    key = (itir.OffsetLiteral(value="E2C2EO"), itir.OffsetLiteral(value=0))
    branches = map_dict[key]["branches"]
    vn = im.ref("vn")
    domain = _outer_domain(0, 3)  # full 3-kolor non-split domain

    result = _build_field_concat_where_from_branches(
        vn, branches, domain,
        apply_edge_shape_bounds=False,
        current_kolor=None,
    )
    assert cpm.is_call_to(result, "concat_where"), (
        "With current_kolor=None, E2C2EO must produce a full concat_where "
        "(for runtime kolor selection in non-split SetAts)."
    )


# ── Test 8: E2C still returns is_ene=True (no regression) ────────────────────

def test_conn_name_e2c_still_is_ene():
    """E2C must still return is_ene=True after the fix (no regression)."""
    key = (itir.OffsetLiteral(value="E2C"), itir.OffsetLiteral(value=0))
    conn_name, is_ene = _conn_name_and_is_edge_to_non_edge(key)
    assert is_ene, "E2C must remain is_ene=True after fix"


def test_conn_name_c2e_still_not_is_ene():
    """C2E must still return is_ene=False (non-edge source, no regression)."""
    key = (itir.OffsetLiteral(value="C2E"), itir.OffsetLiteral(value=0))
    conn_name, is_ene = _conn_name_and_is_edge_to_non_edge(key)
    assert not is_ene, "C2E must remain is_ene=False after fix (cell-source branches not edge-kolor keyed)"


# ── Test 9: NeighborReductionUnroller.visit_FunCall — the real regression test ──

def test_neighbor_reduction_unroller_e2c2eo_single_asfo_when_current_kolor_set():
    """THE REGRESSION TEST: tests at the visit_FunCall level, which is where
    the actual bug was.

    Before fix (_conn_name_and_is_edge_to_non_edge returns is_ene=False for E2C2EO):
      visit_FunCall sees E2C2EO as_fieldop with current_kolor=0.
      is_ene=False → passes current_kolor=None to _build_field_concat_where_from_branches.
      Path 2 fires → generates full 3-branch concat_where inside Kolor:[0,1) SetAt.
      prune_empty_concat_where cannot prune Kolor:[1,2) and Kolor:[2,3) branches
      (keep_existing_domains=True stops domain propagation into as_fieldop lambdas).
      DaCe allocates a zero-sized Kolor transient →
        InvalidSDFGEdgeError: Dimensionality mismatch between src/dst subsets
        e.g. gtir_tmp_211(0) [0:17, 0:18, 0, ...] -> [0:17, 0:18, 0:0, ...]

    After fix (is_ene=True for E2C2EO):
      current_kolor=0 is passed through → path 1 fires → single as_fieldop returned.
      No concat_where → no dead branches → no zero-sized Kolor transient.
    """
    from gt4py.next.iterator.transforms.structured_backend_passes import (
        NeighborReductionUnroller,
    )

    # Construct the IR pattern that visit_FunCall recognises as Path 1 (deref-shift):
    #   as_fieldop(λ it → deref(shift(E2C2EO, 0)(it)), domain)(vn)
    domain = _outer_domain(0, 1)  # per-kolor split: Kolor:[0,1)
    vn = im.ref("vn")
    stencil = im.lambda_("it")(im.deref(im.shift("E2C2EO", 0)(im.ref("it"))))
    e2c2eo_node = im.as_fieldop(stencil, domain)(vn)

    unroller = NeighborReductionUnroller()
    result = unroller.visit(
        e2c2eo_node,
        current_domain=domain,
        current_kolor=0,          # from per-kolor split SetAt domain
        symbolic_domain_sizes={},
    )

    # Before fix: generates concat_where (3-branch, dead Kolor:[1,2)/[2,3) branches
    #             survive prune_empty → zero-sized Kolor transient in DaCe)
    # After fix:  generates single as_fieldop (correct shift directly selected)
    assert not cpm.is_call_to(result, "concat_where"), (
        "visit_FunCall with current_kolor=0 must produce a single as_fieldop for E2C2EO, "
        "NOT a concat_where. "
        "Before fix: is_ene=False in _conn_name_and_is_edge_to_non_edge → current_kolor=None "
        "passed → full 3-branch concat_where generated inside Kolor:[0,1) SetAt → dead "
        "Kolor:[1,2)/[2,3) branches survive infer/prune → DaCe Dimensionality mismatch. "
        "After fix: is_ene=True → current_kolor=0 passed → path-1 selects single shift."
    )
    assert cpm.is_applied_as_fieldop(result), "Result must be an as_fieldop"


def test_neighbor_reduction_unroller_e2c2eo_full_concat_when_current_kolor_none():
    """Complementary test: with current_kolor=None (non-split path), a full
    concat_where must still be generated.  This covers apply_divergence_damping
    (E2C2EO on local intermediate) where the full branching is semantically needed.
    """
    from gt4py.next.iterator.transforms.structured_backend_passes import (
        NeighborReductionUnroller,
    )

    domain = _outer_domain(0, 3)  # non-split: full Kolor:[0,3) domain
    vn = im.ref("vn")
    stencil = im.lambda_("it")(im.deref(im.shift("E2C2EO", 0)(im.ref("it"))))
    e2c2eo_node = im.as_fieldop(stencil, domain)(vn)

    unroller = NeighborReductionUnroller()
    result = unroller.visit(
        e2c2eo_node,
        current_domain=domain,
        current_kolor=None,       # non-split or local-intermediate path
        symbolic_domain_sizes={},
    )

    assert cpm.is_call_to(result, "concat_where"), (
        "With current_kolor=None, E2C2EO must produce a full concat_where "
        "(for runtime kolor selection in non-split SetAts or local-intermediate stencils)."
    )


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_e2c2e_on_local_intermediate_detects_lambda_param,
        test_e2c2e_on_local_intermediate_ignores_program_param,
        test_e2c2e_on_local_intermediate_iterator_param_not_flagged,
        test_conn_name_e2c2eo_is_edge_source_after_fix,
        test_conn_name_e2c2e_is_edge_source_after_fix,
        test_e2c2eo_build_branches_with_current_kolor_selects_single_shift,
        test_e2c2eo_build_branches_with_current_kolor_selects_correct_shift_per_kolor,
        test_e2c2eo_build_branches_without_current_kolor_generates_concat_where,
        test_conn_name_e2c_still_is_ene,
        test_conn_name_c2e_still_not_is_ene,
        # The regression tests that capture the actual bug:
        test_neighbor_reduction_unroller_e2c2eo_single_asfo_when_current_kolor_set,
        test_neighbor_reduction_unroller_e2c2eo_full_concat_when_current_kolor_none,
    ]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failures.append((t.__name__, e))
            print(f"  FAIL  {t.__name__}:\n        {e}")
        except Exception as e:
            failures.append((t.__name__, e))
            print(f"  ERR   {t.__name__}: {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"{len(failures)}/{len(tests)} tests FAILED")
        sys.exit(1)
    print(f"All {len(tests)} tests passed")
