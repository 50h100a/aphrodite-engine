# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A request whose grammar fails to compile must not take the engine with it.

Grammar compilation runs on a thread pool, so the exception is parked in a
Future and only surfaces when the scheduler polls it while promoting the
request out of WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR. That poll happens inside
`Scheduler.schedule()`, which `EngineCore.step()` calls unguarded: an exception
there is fatal to the whole engine, not to the one bad request.
"""

from concurrent.futures import Future

import pytest

from aphrodite.sampling_params import StructuredOutputsParams
from aphrodite.v1.engine import FinishReason
from aphrodite.v1.outputs import ModelRunnerOutput
from aphrodite.v1.request import RequestStatus
from aphrodite.v1.structured_output.request import StructuredOutputRequest
from tests.v1.core.utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput(req_ids=[], req_id_to_index={})


def _block_on_failed_grammar(request, exc=None):
    """Put `request` where a request sits while its grammar compiles, with a
    compilation that has already failed."""
    future: Future = Future()
    future.set_exception(exc or RuntimeError("Cannot find field 0 in #/properties/x/anyOf/0"))

    structured_req = StructuredOutputRequest(params=StructuredOutputsParams(json="{}"))
    structured_req.grammar = future
    request.structured_output_request = structured_req
    request.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    return request


def test_schedule_survives_a_grammar_that_will_not_compile():
    scheduler = create_scheduler()
    (request,) = create_requests(num_requests=1)
    scheduler.add_request(_block_on_failed_grammar(request))

    scheduler.schedule()

    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.request_id not in [req.request_id for req in scheduler.waiting]
    assert request.request_id not in [req.request_id for req in scheduler.skipped_waiting]
    assert not scheduler.running


def test_the_client_is_told_the_request_failed():
    """finish_reason=error is the engine's request-scoped failure channel and
    is contracted to become a 500. Finishing the request without emitting an
    output would just hang the caller until it timed out."""
    scheduler = create_scheduler()
    (request,) = create_requests(num_requests=1)
    scheduler.add_request(_block_on_failed_grammar(request))

    scheduler_output = scheduler.schedule()
    engine_core_outputs = scheduler.update_from_output(scheduler_output, EMPTY_MODEL_RUNNER_OUTPUT)

    outputs = engine_core_outputs[request.client_index].outputs
    assert [out.request_id for out in outputs] == [request.request_id]
    assert outputs[0].finish_reason == FinishReason.ERROR


def test_the_failure_is_reported_exactly_once():
    """The pending list has to be drained when it is flushed; a client that
    receives two terminal outputs for one request hits a KeyError in the
    output processor."""
    scheduler = create_scheduler()
    (request,) = create_requests(num_requests=1)
    scheduler.add_request(_block_on_failed_grammar(request))

    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(scheduler_output, EMPTY_MODEL_RUNNER_OUTPUT)

    scheduler_output = scheduler.schedule()
    engine_core_outputs = scheduler.update_from_output(scheduler_output, EMPTY_MODEL_RUNNER_OUTPUT)
    assert all(not eco.outputs for eco in engine_core_outputs.values())


def test_the_rest_of_the_batch_is_unaffected():
    """The point of containing the failure. Also guards the waiting queue: the
    failed request is removed by finish_requests(), so the scheduling loop must
    not go on to pop a request it never looked at."""
    scheduler = create_scheduler()
    doomed, healthy = create_requests(num_requests=2, req_ids=["doomed", "healthy"])
    scheduler.add_request(_block_on_failed_grammar(doomed))
    scheduler.add_request(healthy)

    scheduler_output = scheduler.schedule()

    assert [req.req_id for req in scheduler_output.scheduled_new_reqs] == ["healthy"]
    assert [req.request_id for req in scheduler.running] == ["healthy"]
    assert doomed.status == RequestStatus.FINISHED_ERROR


def test_a_grammar_still_compiling_keeps_waiting():
    """The ordinary case must not be swept up: a Future that has not resolved
    yet means "check again next step", not "fail the request"."""
    scheduler = create_scheduler()
    (request,) = create_requests(num_requests=1)
    request.structured_output_request = StructuredOutputRequest(params=StructuredOutputsParams(json="{}"))
    request.structured_output_request.grammar = Future()
    request.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    scheduler.add_request(request)

    scheduler.schedule()

    assert request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    assert list(scheduler.skipped_waiting) == [request]
