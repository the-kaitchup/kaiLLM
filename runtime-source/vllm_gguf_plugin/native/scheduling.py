"""Native prompt checkpoints and optional atomic-prefill split_equal scheduling.

Only fresh idle-engine cohorts are regrouped. Ordinary decode, single requests,
long prompts and mixed arrivals retain vLLM's existing scheduler, with checkpoint
caps when the native tail hook is enabled. This is not a general replica of
llama-server admission scheduling.
"""

import logging
import os
from dataclasses import dataclass, field
from functools import wraps
from itertools import islice

logger = logging.getLogger("vllm.gguf.schedule")
ENVIRONMENT_FLAG = "GGUF_PARITY_LLAMA_BATCH_SCHEDULE"


def reserve_prefill_boundary(computed, total, requested, tail=4, budget=512):
    """Stop before native prompt checkpoints without enlarging a reservation.

    The reference server reserves P-(n_ubatch+4) and P-4 for its default
    checkpoint-enabled completion path (logical batch 2048, microbatch 512).
    ``budget`` is that reference microbatch size, not vLLM's scheduling budget.
    A different vLLM budget changes ``requested`` and may add smaller chunks,
    but must not move either checkpoint. Matching boundaries alone does not
    guarantee identical arithmetic when the intervening chunk sizes differ.
    """
    if tail <= 0 or budget <= 0:
        raise ValueError("Checkpoint tail and reference budget must be positive")
    for boundary in (total - budget - tail, total - tail):
        if 0 <= computed < boundary:
            requested = min(requested, boundary - computed)
    return requested


def split_equal_steps(lengths, budget=512):
    """Port native sequential=true, n_keep_tail=0 for independent sequences.

    See llama-batch.cpp split_equal: exhausted slot IDs create gaps, so only
    the first contiguous run of remaining IDs participates in each ubatch.
    """
    if budget < 1 or any(type(n) is not int or n < 0 for n in lengths):
        raise ValueError("Expected nonnegative lengths and a positive token budget")
    remaining = list(lengths)
    steps = []
    while any(remaining):
        first = next(i for i, n in enumerate(remaining) if n)
        end = first
        while end < len(remaining) and remaining[end] and end - first < budget:
            end += 1
        slots = range(first, end)
        count = min(budget // len(slots), *(remaining[i] for i in slots))
        steps.append({i: count for i in slots})
        for i in slots:
            remaining[i] -= count
    return steps


@dataclass
class PrefillCohort:
    request_ids: tuple[str, ...]
    lengths: tuple[int, ...]
    steps: list[dict[str, int]] = field(init=False)
    positions: dict[str, int] = field(init=False)
    step: int = 0

    def __post_init__(self):
        self.steps = [
            {self.request_ids[slot]: count for slot, count in part.items()}
            for part in split_equal_steps([n - 4 for n in self.lengths])
        ]
        self.steps.append(dict.fromkeys(self.request_ids, 4))
        self.positions = dict.fromkeys(self.request_ids, 0)

    @property
    def quotas(self):
        return self.steps[self.step]

    def valid(self, requests):
        for request_id, position in self.positions.items():
            request = requests.get(request_id)
            if request is None or request.num_computed_tokens != position:
                return False
            status = getattr(request.status, "name", request.status)
            if status not in ("RUNNING", "WAITING"):
                return False
        return True

    def consume(self, scheduled):
        if scheduled != self.quotas:
            return False
        for request_id, count in scheduled.items():
            self.positions[request_id] += count
        self.step += 1
        return True


def _scope_reason(scheduler, requests):
    config = scheduler.vllm_config
    if scheduler.running or scheduler.kv_holding_waiting:
        return "mixed_or_resumed_requests"
    if (
        scheduler.max_num_scheduled_tokens != 512
        or config.scheduler_config.max_num_batched_tokens != 512
    ):
        return "physical_budget_is_not_512"
    if (
        config.cache_config.enable_prefix_caching
        or scheduler.need_mamba_block_aligned_split
    ):
        return "prefix_cache_or_aligned_state"
    # VllmConfig.num_prefill_lookahead_tokens is zero without a drafter;
    # one is specifically the Eagle-family next-token lookahead.
    if config.speculative_config is not None or scheduler.num_prefill_lookahead != 0:
        return "speculative_decoding"
    if any(
        getattr(config.parallel_config, name, 1) != 1
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
        )
    ):
        return "parallel_execution"
    if getattr(scheduler.policy, "value", scheduler.policy) != "fcfs":
        return "non_fcfs_policy"
    if scheduler.connector is not None or scheduler.ec_connector is not None:
        return "cache_connector"
    if getattr(config, "lora_config", None) is not None:
        return "lora"
    for request in requests:
        if (
            getattr(request.status, "name", request.status) != "WAITING"
            or request.num_computed_tokens
            or request.num_output_tokens
            or request.num_output_placeholders
            or request.has_encoder_inputs
            or request.num_tokens != request.num_prompt_tokens
        ):
            return "request_is_not_fresh_text"
        # Longer prompts have another server checkpoint at P-(512+4).
        if not 5 <= request.num_prompt_tokens <= 516:
            return "prompt_length_outside_5_to_516"
    if sum(request.num_prompt_tokens - 4 for request in requests) > 2048:
        return "logical_head_batch_exceeds_2048"
    return None


def _record_fallback(scheduler, reason):
    stats = scheduler._gguf_llama_batch_schedule_stats
    previous = stats["fallbacks"].get(reason, 0)
    stats["fallbacks"][reason] = previous + 1
    if not previous:
        logger.info(
            "Native equal-prefill scheduling uses ordinary scheduling: %s", reason
        )


def _begin_cohort(scheduler):
    limit = min(8, scheduler.max_num_active_reqs, scheduler.max_num_running_reqs)
    requests = list(islice(scheduler.waiting, limit))
    if len(requests) < 2:
        return None
    reason = _scope_reason(scheduler, requests)
    if reason:
        _record_fallback(scheduler, reason)
        return None
    # The measured atomic /completion array admits prompt 0..N-1 into native
    # slots N-1..0. Keep those virtual slot IDs fixed through remainder groups.
    # Matrix reductions depend on the column's position, so matching only each
    # group's membership is insufficient. Reorder this fresh FCFS cohort before
    # allocation; running admission then preserves the same order through decode.
    # Request identities and the relative order of other waiting entries remain
    # unchanged. No completed SchedulerOutput is modified.
    requests.reverse()
    for request in requests:
        scheduler.waiting.remove_request(request)
    # prepend_requests reverses its iterable, like deque.extendleft.
    scheduler.waiting.prepend_requests(reversed(requests))
    cohort = PrefillCohort(
        tuple(request.request_id for request in requests),
        tuple(request.num_prompt_tokens for request in requests),
    )
    scheduler._gguf_llama_batch_schedule_stats["cohorts_started"] += 1
    logger.info(
        "Native equal-prefill cohort (reverse admission slots): lengths=%s, "
        "(requests,query_tokens)=%s",
        list(cohort.lengths),
        [(len(step), next(iter(step.values()))) for step in cohort.steps],
    )
    return cohort


def install_llama_batch_schedule(scheduler_class=None):
    """Cap before cache allocation; never modify a completed SchedulerOutput."""
    if scheduler_class is None:
        from vllm.v1.core.sched.scheduler import Scheduler

        scheduler_class = Scheduler
    if getattr(scheduler_class.schedule, "_gguf_llama_batch_schedule", False):
        return
    original_schedule = scheduler_class.schedule
    original_reserve = scheduler_class._reserve_prefill_lookahead

    @wraps(original_reserve)
    def reserve(self, request, num_computed_tokens, num_new_tokens):
        count = original_reserve(self, request, num_computed_tokens, num_new_tokens)
        caps = getattr(self, "_gguf_llama_batch_step_caps", None)
        return count if caps is None else min(count, caps.get(request.request_id, 0))

    @wraps(original_schedule)
    def schedule(self, *args, **kwargs):
        additional = self.vllm_config.additional_config or {}
        marker = (
            additional.get("gguf_native_runtime")
            if isinstance(additional, dict)
            else None
        )
        if isinstance(marker, dict):
            # The scheduler may live in a different process from the CUDA
            # worker. Its own selected profile is authoritative, including an
            # explicit false value, regardless of inherited diagnostic flags.
            enabled = marker.get("features", {}).get("llama_batch_schedule", False)
        else:
            enabled = (
                isinstance(additional, dict)
                and additional.get("gguf_parity_current_llama")
                and os.environ.get(ENVIRONMENT_FLAG) == "1"
            )
        if not enabled:
            return original_schedule(self, *args, **kwargs)
        if not hasattr(self, "_gguf_llama_batch_schedule_stats"):
            self._gguf_llama_batch_schedule_stats = {
                "cohorts_started": 0,
                "cohorts_completed": 0,
                "scheduled_steps": 0,
                "fallbacks": {},
            }
        cohort = getattr(self, "_gguf_llama_batch_cohort", None)
        if cohort is not None and not cohort.valid(self.requests):
            _record_fallback(self, "cohort_cancelled_or_preempted")
            cohort = None
        if cohort is None:
            cohort = _begin_cohort(self)
        self._gguf_llama_batch_cohort = cohort
        self._gguf_llama_batch_step_caps = cohort.quotas if cohort else None
        try:
            result = original_schedule(self, *args, **kwargs)
        except BaseException:
            self._gguf_llama_batch_cohort = None
            raise
        finally:
            self._gguf_llama_batch_step_caps = None
        if cohort is not None:
            scheduled = result.num_scheduled_tokens
            if getattr(result, "preempted_req_ids", None) or (
                scheduled and not cohort.consume(scheduled)
            ):
                _record_fallback(self, "allocation_or_scheduling_changed_cohort")
                self._gguf_llama_batch_cohort = None
            elif scheduled:
                self._gguf_llama_batch_schedule_stats["scheduled_steps"] += 1
                if cohort.step == len(cohort.steps):
                    self._gguf_llama_batch_schedule_stats["cohorts_completed"] += 1
                    self._gguf_llama_batch_cohort = None
        return result

    schedule._gguf_llama_batch_schedule = True
    scheduler_class.schedule = schedule
    scheduler_class._reserve_prefill_lookahead = reserve
