"""Admit an explicit native completion prompt array in one engine operation.

Only the selected native engine with cohort scheduling participates. Requests
are prepared and registered by AsyncLLM as usual, then buffered in the frontend;
the core never waits for missing members and other HTTP requests are not paused.
"""

import asyncio
import contextlib
import logging
from contextvars import ContextVar
from functools import wraps

logger = logging.getLogger("vllm.gguf.admission")
_SERVING_SCOPE = ContextVar("gguf_native_completion_scope", default=None)
_MEMBER_SCOPE = ContextVar("gguf_native_completion_member", default=None)
_UTILITY = "gguf_add_completion_batch"
_TIMEOUT = 60.0


def _selected(config):
    additional = getattr(config, "additional_config", None)
    marker = (
        additional.get("gguf_native_runtime") if isinstance(additional, dict) else None
    )
    return (
        isinstance(marker, dict)
        and marker.get("features", {}).get("llama_batch_schedule") is True
    )


def _array_size(request):
    prompt = request.prompt
    if (
        not isinstance(prompt, list)
        or not 2 <= len(prompt) <= 8
        or (request.n or 1) != 1
        or getattr(request, "use_beam_search", False)
        or getattr(request, "best_of", None) not in (None, 1)
    ):
        return None
    if all(isinstance(item, str) for item in prompt) or all(
        isinstance(item, list) and all(type(token) is int for token in item)
        for item in prompt
    ):
        return len(prompt)
    return None


class _AdmissionGroup:
    def __init__(self, engine, size, *, timeout=_TIMEOUT):
        self.engine = engine
        self.client = engine.engine_core
        self.size = size
        self.timeout = timeout
        self.requests = {}
        self.ready = asyncio.get_running_loop().create_future()
        self.error = None
        self.commit_task = None
        self.cleanup_task = None

    async def add(self, index, client, request):
        if client is not self.client or not 0 <= index < self.size:
            raise RuntimeError("Native completion batch transport changed")
        if index in self.requests:
            raise RuntimeError("Native completion member submitted more than once")
        request.client_index = client.client_index
        self.requests[index] = request
        if self.error is not None:
            # This member may finish preprocessing after its siblings failed.
            # AsyncLLM has already registered its local output collector.
            await self.engine.abort([request.request_id], internal=True)
            raise self.error
        if len(self.requests) == self.size:
            self.commit_task = asyncio.create_task(self._commit())
        try:
            await asyncio.wait_for(asyncio.shield(self.ready), self.timeout)
        except BaseException as error:
            await self.fail(error)
            raise

    async def _commit(self):
        try:
            await self.client.call_utility_async(
                _UTILITY, [self.requests[index] for index in range(self.size)]
            )
        except BaseException as error:
            if not self.ready.done():
                self.ready.set_exception(error)
                self.ready.exception()
        else:
            if not self.ready.done():
                self.ready.set_result(None)

    async def fail(self, error):
        if self.error is None:
            self.error = error
            if not self.ready.done():
                self.ready.set_exception(error)
                # Also consume failures when every producer was cancelled.
                self.ready.exception()
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._cleanup())
        await asyncio.shield(self.cleanup_task)

    async def _cleanup(self):
        # Finish/cancel the send before enqueueing aborts on the same transport.
        # A request already delivered to the core is therefore aborted after
        # admission; a cancelled unsent utility cannot add it afterwards.
        if self.commit_task is not None:
            if not self.commit_task.done():
                self.commit_task.cancel()
            with contextlib.suppress(BaseException):
                await self.commit_task
        ids = [request.request_id for request in self.requests.values()]
        if ids:
            try:
                await self.engine.abort(ids, internal=True)
            except Exception:
                logger.exception("Failed to abort native completion batch")


async def _member(iterator, group, index):
    finished = False
    try:
        while True:
            # merge_async_iterators creates a new task for each anext(). Reset
            # before yielding: ContextVar tokens cannot cross task contexts.
            token = _MEMBER_SCOPE.set((group, index))
            try:
                item = await anext(iterator)
            except StopAsyncIteration:
                finished = True
                return
            finally:
                _MEMBER_SCOPE.reset(token)
            yield item
    except BaseException as error:
        await group.fail(error)
        raise
    finally:
        if not finished:
            await group.fail(asyncio.CancelledError())
        await iterator.aclose()


def _admit_batch(self, requests):
    """Core utility: convert/preprocess every member before adding any member."""
    import msgspec
    from vllm.v1.engine import EngineCoreRequest

    if not _selected(self.vllm_config):
        raise ValueError("Atomic GGUF admission requires the selected native profile")
    # CoreProc converts a top-level msgspec.Struct, but not a typed list.
    requests = msgspec.convert(requests, type=list[EngineCoreRequest])
    maximum = min(8, self.vllm_config.scheduler_config.max_num_seqs)
    if not 2 <= len(requests) <= maximum:
        raise ValueError("Native completion batch needs 2..8 supported requests")
    ids = [request.request_id for request in requests]
    if len(set(ids)) != len(ids) or any(self.scheduler.has_request(i) for i in ids):
        raise ValueError("Native completion batch request IDs must be new and unique")
    if len({request.client_index for request in requests}) != 1 or any(
        request.prompt_token_ids is None
        or request.prompt_embeds is not None
        or request.mm_features
        or request.lora_request is not None
        or request.pooling_params is not None
        or request.sampling_params is None
        or request.sampling_params.n != 1
        or request.abort_immediately
        for request in requests
    ):
        raise ValueError("Native completion batches require independent text requests")
    prepared = [self.preprocess_add_request(request) for request in requests]
    rejected = [
        self._reject_add_in_shutdown(request) or self._reject_add_while_paused(request)
        for request, _ in prepared
    ]
    if any(rejected):
        # Preserve ordinary paused/shutdown rejection outputs. Do not partially
        # admit a group if only some members were rejected.
        remaining = [
            request.request_id
            for (request, _), reject in zip(prepared, rejected)
            if not reject
        ]
        if remaining:
            self._send_abort_outputs_to_client(remaining, requests[0].client_index)
        return 0
    try:
        for request, wave in prepared:
            self.add_request(request, wave)
    except BaseException:
        self.abort_requests(ids)
        raise
    logger.info("Atomically admitted native completion array: %d requests", len(ids))
    return len(ids)


def install_completion_batch_hook():
    from vllm.entrypoints.openai.completion import serving
    from vllm.v1.engine.core import EngineCoreProc
    from vllm.v1.engine.core_client import AsyncMPClient

    cls = serving.OpenAIServingCompletion
    if getattr(cls._create_completion, "_gguf_atomic_completion", False):
        return
    original_create = cls._create_completion
    original_merge = serving.merge_async_iterators
    original_add = AsyncMPClient.add_request_async

    @wraps(original_create)
    async def create(self, request, *args, **kwargs):
        engine = self.engine_client
        config = getattr(engine, "vllm_config", None)
        size = _array_size(request)
        scope = None
        if (
            size is not None
            and _selected(config)
            and type(engine.engine_core) is AsyncMPClient
            and size <= config.scheduler_config.max_num_seqs
        ):
            scope = (engine, size)
        token = _SERVING_SCOPE.set(scope)
        try:
            return await original_create(self, request, *args, **kwargs)
        finally:
            _SERVING_SCOPE.reset(token)

    @wraps(original_merge)
    def merge(*iterators):
        scope = _SERVING_SCOPE.get()
        if scope is None or len(iterators) != scope[1]:
            return original_merge(*iterators)
        # Capture scope now; streaming starts iterating after create() returned.
        group = _AdmissionGroup(*scope)
        return original_merge(
            *(
                _member(iterator, group, index)
                for index, iterator in enumerate(iterators)
            )
        )

    @wraps(original_add)
    async def add(self, request):
        scope = _MEMBER_SCOPE.get()
        if scope is None or self is not scope[0].client:
            return await original_add(self, request)
        group, index = scope
        return await group.add(index, self, request)

    create._gguf_atomic_completion = True
    cls._create_completion = create
    serving.merge_async_iterators = merge
    AsyncMPClient.add_request_async = add
    setattr(EngineCoreProc, _UTILITY, _admit_batch)
