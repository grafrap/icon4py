# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py import eve
from gt4py.next import (
    config as gtx_config,
    constructors,
    named_collections as gtx_named_collections,
    typing as gtx_typing,
)

# TODO(havogt): import will disappear after FieldOperators support `.compile`
from gt4py.next.ffront.decorator import FieldOperator
from gt4py.next.instrumentation import metrics as gtx_metrics

from icon4py.model.common import model_backends, model_options
from icon4py.model.common.grid import base
from icon4py.model.common.utils import device_utils


def allocate_data(
    allocator: gtx_typing.FieldBufferAllocationUtil | None,
    input_data: dict[
        str, Any
    ],  # `Field`s or collection of `Field`s are re-allocated, the rest is passed through
) -> dict[str, Any]:
    def _allocate_field(f: gtx.Field) -> gtx.Field:
        if not hasattr(f, "domain"):
            return f
        return constructors.as_field(domain=f.domain, data=f.ndarray, allocator=allocator)

    input_data = {
        k: gtx_named_collections.tree_map_named_collection(_allocate_field)(v)
        if not gtx.is_scalar_type(v) and k != "domain"
        else v
        for k, v in input_data.items()
    }
    return input_data


@dataclasses.dataclass(frozen=True)
class Output:
    name: str
    refslice: tuple[slice, ...] = dataclasses.field(default_factory=lambda: (slice(None),))
    gtslice: tuple[slice, ...] = dataclasses.field(default_factory=lambda: (slice(None),))


@dataclasses.dataclass
class _ConnectivityConceptFixer:
    """
    This works around a misuse of dimensions as an identifier for connectivities.
    Since GT4Py might change the way the mesh is represented, we could
    keep this for a while, otherwise we need to touch all StencilTests.
    """

    _grid: base.Grid

    def __getitem__(self, dim: gtx.Dimension | str) -> np.ndarray:
        if isinstance(dim, gtx.Dimension):
            dim = dim.value
        return self._grid.get_connectivity(dim).asnumpy()


_CHAR_TO_GRID_OBJECT = {"E": "Edge", "C": "Cell", "V": "Vertex"}


def _to_unstructured(field: np.ndarray, backtransform: np.ndarray) -> np.ndarray:
    out = np.zeros_like(field)
    out[backtransform] = field
    return out


def _to_structured(field: np.ndarray, backtransform: np.ndarray) -> np.ndarray:
    return field[backtransform]


def _infer_connectivity_objects(dim_name: str) -> tuple[str | None, str | None]:
    letters = [c for c in dim_name if c in _CHAR_TO_GRID_OBJECT]
    if len(letters) < 2:
        return None, None
    return _CHAR_TO_GRID_OBJECT[letters[0]], _CHAR_TO_GRID_OBJECT[letters[-1]]


def _horizontal_grid_object(field: gtx.Field) -> str | None:
    if not hasattr(field, "domain"):
        return None
    for dim in field.domain.dims:
        if dim.kind == gtx.DimensionKind.HORIZONTAL and dim.value in ("Edge", "Cell", "Vertex"):
            return dim.value
    return None


@dataclasses.dataclass
class _StructuredConnectivityConceptFixer:
    _grid: base.Grid
    _backtransform_by_object: dict[str, np.ndarray]
    _cache: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

    def __getitem__(self, dim: gtx.Dimension | str) -> np.ndarray:
        dim_name = dim.value if isinstance(dim, gtx.Dimension) else dim
        if dim_name in self._cache:
            return self._cache[dim_name]

        connectivity = np.array(self._grid.get_connectivity(dim_name).asnumpy(), copy=True)
        center_object, neighbor_object = _infer_connectivity_objects(dim_name)

        if center_object in self._backtransform_by_object:
            center_backtransform = self._backtransform_by_object[center_object]
            if connectivity.shape[0] == center_backtransform.shape[0]:
                connectivity = _to_unstructured(connectivity, center_backtransform)

        if neighbor_object in self._backtransform_by_object:
            neighbor_backtransform = self._backtransform_by_object[neighbor_object]
            valid = connectivity >= 0
            connectivity[valid] = neighbor_backtransform[connectivity[valid]]

        self._cache[dim_name] = connectivity
        return connectivity


class StandardStaticVariants(eve.StrEnum):
    NONE = "none"
    COMPILE_TIME_DOMAIN = "compile_time_domain"
    COMPILE_TIME_VERTICAL = "compile_time_vertical"


def test_and_benchmark(
    self: StencilTest,
    benchmark: Any,  # should be `pytest_benchmark.fixture.BenchmarkFixture` but pytest_benchmark is not typed
    grid: base.Grid,
    _properly_allocated_input_data: dict[str, gtx.Field | tuple[gtx.Field, ...]],
    _configured_program: Callable[..., None],
    request: pytest.FixtureRequest,
) -> None:
    skip_stenciltest_verification = request.config.getoption(
        "skip_stenciltest_verification"
    )  # skip verification if `--skip-stenciltest-verification` CLI option is set
    if not skip_stenciltest_verification:
        reference_connectivities, reference_inputs = self._prepare_reference_inputs(
            grid=grid,
            input_data=_properly_allocated_input_data,
        )
        reference_outputs = self.reference(reference_connectivities, **reference_inputs)

        _configured_program(**_properly_allocated_input_data, offset_provider=grid.connectivities)
        self._verify_stencil_test(
            input_data=_properly_allocated_input_data, reference_outputs=reference_outputs
        )

    if benchmark is not None and benchmark.enabled:
        warmup_rounds = int(os.getenv("ICON4PY_STENCIL_TEST_WARMUP_ROUNDS", "1"))
        iterations = int(os.getenv("ICON4PY_STENCIL_TEST_ITERATIONS", "10"))

        # Use of `pedantic` to explicitly control warmup rounds and iterations
        benchmark.pedantic(
            _configured_program,
            args=(),
            kwargs=dict(**_properly_allocated_input_data, offset_provider=grid.connectivities),
            rounds=int(
                os.getenv("ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS", "3")
            ),  # 30 iterations in total should be stable enough
            warmup_rounds=warmup_rounds,
            iterations=iterations,
        )

        # Collect GT4Py runtime metrics if enabled
        if gtx_config.COLLECT_METRICS_LEVEL > 0:
            assert (
                len(_configured_program._compiled_programs.compiled_programs) == 1
            ), "Multiple compiled programs found, cannot extract metrics."
            # Get compiled programs from the _configured_program passed to test
            compiled_programs = _configured_program._compiled_programs.compiled_programs
            # Get the pool key necessary to find the right metrics key. There should be only one compiled program in _configured_program
            pool_key = next(iter(compiled_programs.keys()))
            # Get the metrics key from the pool key to read the corresponding metrics
            metrics_key = _configured_program._compiled_programs._metrics_key_from_pool_key(
                pool_key
            )
            metrics_data = gtx_metrics.sources
            compute_samples = metrics_data[metrics_key].metrics["compute"].samples
            # exclude warmup iterations, one extra iteration for calibrating pytest-benchmark and one for validation (if executed)
            initial_program_iterations_to_skip = warmup_rounds * iterations + (
                1 if skip_stenciltest_verification else 2
            )
            benchmark.extra_info["gtx_metrics"] = compute_samples[
                initial_program_iterations_to_skip:
            ]


class StencilTest:
    """
    Base class to be used for testing stencils.

    Example (pseudo-code):

        >>> class TestMultiplyByTwo(StencilTest):  # doctest: +SKIP
        ...     PROGRAM = multiply_by_two  # noqa: F821
        ...     OUTPUTS = ("some_output",)
        ...     STATIC_PARAMS = {"category_a": ["flag0"], "category_b": ["flag0", "flag1"]}
        ...
        ...     @pytest.fixture
        ...     def input_data(self):
        ...         return {"some_input": ..., "some_output": ...}
        ...
        ...     @staticmethod
        ...     def reference(some_input, **kwargs):
        ...         return dict(some_output=np.asarray(some_input) * 2)
    """

    PROGRAM: ClassVar[gtx_typing.Program | gtx_typing.FieldOperator]
    OUTPUTS: ClassVar[tuple[str | Output, ...]]
    STATIC_PARAMS: ClassVar[dict[str, Sequence[str]] | None] = None
    ENABLE_REFERENCE_TRANSLATION_FOR_STRUCTURED_BACKEND: ClassVar[bool] = False
    REFERENCE_TRANSLATION_NX: ClassVar[int] = 13
    REFERENCE_BOUNDARY_LEVEL_BY_OBJECT: ClassVar[dict[str, int]] = {
        "Edge": 3,
        "Cell": 2,
        "Vertex": 2,
    }

    reference: ClassVar[Callable[..., Mapping[str, np.ndarray | tuple[np.ndarray, ...]]]]

    @pytest.fixture
    def _configured_program(
        self,
        backend_like: model_backends.BackendLike,
        static_variant: Sequence[str],
        input_data: dict[str, gtx.Field | tuple[gtx.Field, ...]],
        grid: base.Grid,
    ) -> Callable[..., None]:
        use_structured_backend = os.environ.get("USE_STRUCTURED_BACKEND", "0") == "1"
        if use_structured_backend:
            if getattr(grid, "id", None) == "simple_grid":
                raise RuntimeError(
                    "Structured backend is disabled for 'simple_grid'. "
                    "Use a structured-compatible grid (e.g. pass '--grid <ICON_GRID_FILE>:<levels>') "
                    "or disable USE_STRUCTURED_BACKEND."
                )
            try:
                from gt4py.next.modules.cartesian_interceptor import (
                    GenericStructuredWrapper,
                    get_global_grid_mapping,
                )
                from gt4py.next.program_processors.runners import gtfn as gtfn_runner

                e2v_conn = grid.connectivities.get("E2V")
                e2v_array = e2v_conn.asnumpy() if e2v_conn is not None else None
                index_map, remap_sizes = get_global_grid_mapping(e2v_override=e2v_array)
                wrapper = GenericStructuredWrapper(
                    operator=self.PROGRAM,
                    backend_factory=gtfn_runner.GTFNBackendFactory,
                    index_map=index_map,
                    remap_sizes=remap_sizes,
                    allocator=model_backends.get_allocator(backend_like),
                    offset_provider=grid.connectivities,
                )
                return device_utils.synchronized_function(
                    wrapper, allocator=model_backends.get_allocator(backend_like)
                )
            except Exception as exc:
                raise RuntimeError("Failed to initialize structured stencil wrapper") from exc

        unused_static_params = set(static_variant) - set(input_data.keys())
        if unused_static_params:
            raise ValueError(
                f"Parameter defined in 'STATIC_PARAMS' not in 'input_data': {unused_static_params}"
            )
        static_args = {name: [input_data[name]] for name in static_variant}
        backend = model_options.customize_backend(self.PROGRAM, backend_like)
        program = self.PROGRAM.with_backend(backend)  # type: ignore[arg-type]  # TODO(havogt): gt4py should accept `None` in with_backend
        if backend is not None:
            if isinstance(program, FieldOperator):
                if len(static_args) > 0:
                    raise NotImplementedError(
                        "'FieldOperator's do not support static arguments yet."
                    )
            else:
                program.compile(
                    offset_provider=grid.connectivities,
                    **static_args,  # type: ignore[arg-type]
                )

        test_func = device_utils.synchronized_function(program, allocator=backend)
        return test_func

    @pytest.fixture
    def _properly_allocated_input_data(
        self,
        input_data: dict[str, gtx.Field | tuple[gtx.Field, ...]],
        backend_like: model_backends.BackendLike,
    ) -> dict[str, Any]:
        # TODO(havogt): this is a workaround,
        # because in the `input_data` fixture provided by the user
        # it does not allocate for the correct device.
        allocator = model_backends.get_allocator(backend_like)
        return allocate_data(allocator=allocator, input_data=input_data)

    def _prepare_reference_inputs(
        self,
        grid: base.Grid,
        input_data: dict[str, gtx.Field | tuple[gtx.Field, ...]],
    ) -> tuple[_ConnectivityConceptFixer, dict[str, Any]]:
        reference_inputs = {
            k: v.asnumpy() if isinstance(v, gtx.Field) else v
            for k, v in input_data.items()
        }
        self._reference_output_backtransform_by_name: dict[str, np.ndarray] = {}

        use_structured_backend = os.environ.get("USE_STRUCTURED_BACKEND", "0") == "1"
        if not (use_structured_backend and self.ENABLE_REFERENCE_TRANSLATION_FOR_STRUCTURED_BACKEND):
            return (
                _ConnectivityConceptFixer(
                    grid  # TODO(havogt): pass as keyword argument (needs fixes in some tests)
                ),
                reference_inputs,
            )

        from gt4py.next.modules.translator import transform_to_unstructured

        sample_by_object: dict[str, np.ndarray] = {}
        for name, value in input_data.items():
            if not isinstance(value, gtx.Field):
                continue
            grid_object = _horizontal_grid_object(value)
            if grid_object is None or grid_object in sample_by_object:
                continue
            arr = reference_inputs[name]
            sample_by_object[grid_object] = arr if arr.ndim == 1 else arr[:, 0]

        backtransform_by_object: dict[str, np.ndarray] = {}
        horizontal_start_by_object: dict[str, int] = {}
        for grid_object, sample in sample_by_object.items():
            boundary_level = self.REFERENCE_BOUNDARY_LEVEL_BY_OBJECT.get(grid_object, 0)
            _, backtransform, _, horizontal_start = transform_to_unstructured(
                sample,
                self.REFERENCE_TRANSLATION_NX,
                grid_object,
                boundary_level,
            )
            backtransform_by_object[grid_object] = backtransform
            horizontal_start_by_object[grid_object] = horizontal_start

        for name, value in input_data.items():
            if not isinstance(value, gtx.Field):
                continue
            grid_object = _horizontal_grid_object(value)
            if grid_object in backtransform_by_object:
                reference_inputs[name] = _to_unstructured(
                    reference_inputs[name],
                    backtransform_by_object[grid_object],
                )

        if "Edge" in horizontal_start_by_object:
            edge_horizontal_start = horizontal_start_by_object["Edge"]
            for key, value in reference_inputs.items():
                if key.startswith("horizontal_start"):
                    reference_inputs[key] = type(value)(edge_horizontal_start)
        elif "Cell" in horizontal_start_by_object:
            cell_horizontal_start = horizontal_start_by_object["Cell"]
            for key, value in reference_inputs.items():
                if key.startswith("horizontal_start"):
                    reference_inputs[key] = type(value)(cell_horizontal_start)
        elif "Vertex" in horizontal_start_by_object:
            vertex_horizontal_start = horizontal_start_by_object["Vertex"]
            for key, value in reference_inputs.items():
                if key.startswith("horizontal_start"):
                    reference_inputs[key] = type(value)(vertex_horizontal_start)

        for out in self.OUTPUTS:
            out_name = out.name if isinstance(out, Output) else out
            out_value = input_data.get(out_name)
            if isinstance(out_value, gtx.Field):
                grid_object = _horizontal_grid_object(out_value)
                if grid_object in backtransform_by_object:
                    self._reference_output_backtransform_by_name[out_name] = backtransform_by_object[
                        grid_object
                    ]

        return (
            _StructuredConnectivityConceptFixer(
                _grid=grid,
                _backtransform_by_object=backtransform_by_object,
            ),
            reference_inputs,
        )

    def _verify_stencil_test(
        self,
        input_data: dict[str, gtx.Field | tuple[gtx.Field, ...]],
        reference_outputs: Mapping[str, np.ndarray | tuple[np.ndarray, ...]],
    ) -> None:
        def _backtransform_reference_output(
            out_name: str,
            output: np.ndarray | tuple[np.ndarray, ...],
        ) -> np.ndarray | tuple[np.ndarray, ...]:
            backtransform = getattr(self, "_reference_output_backtransform_by_name", {}).get(out_name)
            if backtransform is None:
                return output
            if isinstance(output, tuple):
                return tuple(_to_structured(arr, backtransform) for arr in output)
            return _to_structured(output, backtransform)

        for out in self.OUTPUTS:
            name, refslice, gtslice = (
                (out.name, out.refslice, out.gtslice)
                if isinstance(out, Output)
                else (out, (slice(None),), (slice(None),))
            )

            input_data_name = input_data[name]  # for mypy
            # TODO(iomaganaris, havogt, nfarabullini): tolerance was increased from 1e-7 to 1e-6
            # to cover floating point descripancies observed in CI tests. Failing CI can be found in
            # https://gitlab.com/cscs-ci/ci-testing/webhook-ci/mirrors/5125340235196978/2255149825504673/-/pipelines/2184694383
            # from PR#861. Reason is probably derivatives of random data. Investigate and lower tolerance back to 1e-7 if possible.
            relative_tolerance = 3e-6
            if isinstance(input_data_name, tuple):
                reference_outputs_tuple = _backtransform_reference_output(name, reference_outputs[name])
                assert isinstance(reference_outputs_tuple, tuple)
                for i_out_field, out_field in enumerate(input_data_name):
                    np.testing.assert_allclose(
                        out_field.asnumpy()[gtslice],
                        reference_outputs_tuple[i_out_field][refslice],
                        equal_nan=True,
                        err_msg=f"Verification failed for '{name}[{i_out_field}]'",
                        rtol=relative_tolerance,  # TODO(iomaganaris, havogt, nfarabullini): check above comment
                    )
            else:
                reference_outputs_name = _backtransform_reference_output(
                    name, reference_outputs[name]
                )  # for mypy
                assert isinstance(reference_outputs_name, np.ndarray)
                print(f"output: {name}, values: {input_data_name.asnumpy()[gtslice][:,0]}")
                print(f"reference: {reference_outputs_name[refslice][:,0]}")
                print(f"slices - gt4py: {gtslice}, reference: {refslice}")
                print(f"input_data_name shape: {input_data_name.asnumpy().shape}, reference_outputs_name shape: {reference_outputs_name.shape}")
                np.testing.assert_allclose(
                    input_data_name.asnumpy()[gtslice],
                    reference_outputs_name[refslice],
                    equal_nan=True,
                    err_msg=f"Verification failed for '{name}'",
                    rtol=relative_tolerance,  # TODO(iomaganaris, havogt, nfarabullini): check above comment
                )

    @staticmethod
    def static_variant(request: pytest.FixtureRequest) -> Sequence[str]:
        """
        Fixture for parametrization over the `STATIC_PARAMS` of the test class.

        Note: the actual `pytest.fixture()`  decoration happens inside `__init_subclass__`,
          when all information is available.
        """
        _, variant = request.param
        return () if variant is None else variant

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        setattr(cls, f"test_{cls.__name__}", test_and_benchmark)

        # decorate `static_variant` with parametrized fixtures, since the
        # parametrization is only available in the concrete subclass definition
        if cls.STATIC_PARAMS is None:
            # not parametrized, return an empty tuple
            cls.static_variant = staticmethod(pytest.fixture(lambda: ()))  # type: ignore[method-assign, assignment] # we override with a non-parametrized function
        else:
            cls.static_variant = staticmethod(  # type: ignore[method-assign]
                pytest.fixture(params=cls.STATIC_PARAMS.items(), scope="class", ids=lambda p: p[0])(
                    cls.static_variant
                )
            )
