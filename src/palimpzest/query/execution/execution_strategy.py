import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import numpy as np
from chromadb.api.models.Collection import Collection

from palimpzest.core.data.dataset import Dataset
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import GenerationStats, PlanStats, SentinelPlanStats
from palimpzest.policy import Policy
from palimpzest.query.operators.convert import LLMConvert
from palimpzest.query.operators.filter import LLMFilter
from palimpzest.query.operators.join import JoinOp
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.scan import ContextScanOp, ScanPhysicalOp
from palimpzest.query.operators.topk import TopKOp
from palimpzest.query.execution.document_sampling import DocumentSourceSampler
from palimpzest.query.execution.record_set_quality import score_record_sets_with_validator
from palimpzest.query.optimizer.plan import PhysicalPlan, SentinelPlan
from palimpzest.utils.progress import PZSentinelProgressManager
from palimpzest.validator.validator import Validator
from palimpzest.utils.trace_env import palimpzest_trace_sentinel

logger = logging.getLogger(__name__)

class BaseExecutionStrategy:
    def __init__(self,
                 scan_start_idx: int = 0, 
                 max_workers: int | None = None,
                 batch_size: int | None = None,
                 num_samples: int | None = None,
                 verbose: bool = False,
                 progress: bool = True,
                 *args,
                 **kwargs):
        self.scan_start_idx = scan_start_idx
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.verbose = verbose
        self.progress = progress


class ExecutionStrategy(BaseExecutionStrategy, ABC):
    """Base strategy for executing query plans. Defines how to execute a PhysicalPlan.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info(f"Initialized ExecutionStrategy {self.__class__.__name__}")
        logger.debug(f"ExecutionStrategy initialized with config: {self.__dict__}")

    @abstractmethod
    def execute_plan(self, plan: PhysicalPlan) -> tuple[list[DataRecord], PlanStats]:
        """Execute a single plan according to strategy"""
        pass

    def _create_input_queues(self, plan: PhysicalPlan) -> dict[str, dict[str, list]]:
        """Initialize input queues for each operator in the plan."""
        input_queues = {f"{topo_idx}-{op.get_full_op_id()}": {} for topo_idx, op in enumerate(plan)}
        for topo_idx, op in enumerate(plan):
            full_op_id = op.get_full_op_id()
            unique_op_id = f"{topo_idx}-{full_op_id}"
            if isinstance(op, ScanPhysicalOp):
                scan_end_idx = (
                    len(op.datasource)
                    if self.num_samples is None
                    else min(self.scan_start_idx + self.num_samples, len(op.datasource))
                )
                input_queues[unique_op_id][f"source_{full_op_id}"] = [idx for idx in range(self.scan_start_idx, scan_end_idx)]
            elif isinstance(op, ContextScanOp):
                input_queues[unique_op_id][f"source_{full_op_id}"] = [None]
            else:
                for source_unique_full_op_id in plan.get_source_unique_full_op_ids(topo_idx, op):
                    input_queues[unique_op_id][source_unique_full_op_id] = []

        return input_queues

class SentinelExecutionStrategy(BaseExecutionStrategy, ABC):
    """Base strategy for executing sentinel query plans. Defines how to execute a SentinelPlan."""
    """
    Specialized query processor that implements MAB sentinel strategy
    for coordinating optimization and execution.
    """
    def __init__(
        self,
        policy: Policy,
        k: int = 6,
        j: int = 4,
        sample_budget: int = 100,
        sample_cost_budget: float | None = None,
        priors: dict | None = None,
        use_final_op_quality: bool = False,
        seed: int = 42,
        exp_name: str | None = None,
        dont_use_priors: bool = False,
        document_sampling_method: Literal["random", "stratified"] = "random",
        stratified_num_strata: int = 8,
        document_sampler: DocumentSourceSampler | None = None,
        on_sample=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.j = j
        self.sample_budget = sample_budget
        self.sample_cost_budget = sample_cost_budget
        self.policy = policy
        self.priors = priors
        self.use_final_op_quality = use_final_op_quality
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.exp_name = exp_name
        self.dont_use_priors = dont_use_priors
        self.document_sampling_method = document_sampling_method
        self.stratified_num_strata = stratified_num_strata
        self.document_sampler = document_sampler
        self.on_sample = on_sample

        # general cache which maps hash(logical_op_id, phys_op_id, hash(input)) --> record_set
        self.cache: dict[int, DataRecordSet] = {}

        # progress manager used to track progress of the execution
        self.progress_manager: PZSentinelProgressManager | None = None

    def _score_quality(
        self,
        validator: Validator,
        source_indices_to_record_sets: dict[tuple[str], list[tuple[DataRecordSet, PhysicalOperator]]],
    ) -> tuple[dict[int, list[DataRecordSet]], GenerationStats]:
        return score_record_sets_with_validator(validator, source_indices_to_record_sets, self.max_workers)

    def _execute_op_set(self, unique_logical_op_id: str, op_inputs: list[tuple[PhysicalOperator, str | tuple, int | DataRecord | list[DataRecord] | tuple[list[DataRecord]]]]) -> tuple[dict[int, list[tuple[DataRecordSet, PhysicalOperator, bool]]], dict[str, int]]:
        def execute_op_wrapper(operator: PhysicalOperator, source_indices: str | tuple, input: int | DataRecord | list[DataRecord] | tuple[list[DataRecord]]) -> tuple[DataRecordSet, PhysicalOperator, list[DataRecord] | list[int]]:
            # operator is a join
            record_set = operator(input[0], input[1]) if isinstance(operator, JoinOp) else operator(input)
            return record_set, operator, source_indices, input

        def get_hash(operator: PhysicalOperator, input: int | DataRecord | list[DataRecord] | tuple[list[DataRecord]]):
            if isinstance(input, list):
                input = tuple(input)
            elif isinstance(input, tuple):
                input = (tuple(input[0]), tuple(input[1]))
            return hash(f"{operator.get_full_op_id()}{hash(input)}")

        # initialize mapping from source indices to output record sets
        source_indices_to_record_sets_and_ops = {source_indices: [] for _, source_indices, _ in op_inputs}

        # if any operations were previously executed, read the results from the cache
        final_op_inputs = []
        for operator, source_indices, input in op_inputs:
            # compute hash
            op_input_hash = get_hash(operator, input)

            # get result from cache
            if op_input_hash in self.cache:
                record_set, operator = self.cache[op_input_hash]
                source_indices_to_record_sets_and_ops[source_indices].append((record_set, operator, False))
                if palimpzest_trace_sentinel():
                    logger.info(
                        "[sentinel-sampling] cache_hit logical_op=%s physical_op=%s source_indices=%s",
                        unique_logical_op_id,
                        operator.get_full_op_id(),
                        source_indices,
                    )

            # otherwise, add to final_op_inputs
            else:
                if palimpzest_trace_sentinel():
                    logger.info(
                        "[sentinel-sampling] run_op logical_op=%s physical_op=%s source_indices=%s llm=%s",
                        unique_logical_op_id,
                        operator.get_full_op_id(),
                        source_indices,
                        self._is_llm_op(operator),
                    )
                final_op_inputs.append((operator, source_indices, input))

        # keep track of the number of llm operations
        num_llm_ops = 0

        # create thread pool w/max workers and run futures over worker pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # create futures
            futures = [
                executor.submit(execute_op_wrapper, operator, source_indices, input)
                for operator, source_indices, input in final_op_inputs
            ]

            output_record_sets = []
            for future in as_completed(futures):
                # update output record sets
                record_set, operator, source_indices, input = future.result()

                # if the operator is a join, get record_set from tuple output
                if isinstance(operator, JoinOp):
                    record_set = record_set[0]

                output_record_sets.append((record_set, operator, source_indices, input))

                # update cache
                op_input_hash = get_hash(operator, input)
                self.cache[op_input_hash] = (record_set, operator)

                # update progress manager
                if self._is_llm_op(operator):
                    num_llm_ops += 1
                    self.progress_manager.incr(unique_logical_op_id, num_samples=1, total_cost=record_set.get_total_cost())

            # update mapping from source_indices to record sets and operators
            for record_set, operator, source_indices, input in output_record_sets:
                # add record_set to mapping from source_indices --> record_sets
                record_set.input = input
                source_indices_to_record_sets_and_ops[source_indices].append((record_set, operator, True))

        return source_indices_to_record_sets_and_ops, num_llm_ops

    def _is_llm_op(self, physical_op: PhysicalOperator) -> bool:
        is_llm_convert = isinstance(physical_op, LLMConvert)
        is_llm_filter = isinstance(physical_op, LLMFilter)
        is_llm_topk = isinstance(physical_op, TopKOp) and isinstance(physical_op.index, Collection)
        is_llm_join = isinstance(physical_op, JoinOp)
        return is_llm_convert or is_llm_filter or is_llm_topk or is_llm_join

    @abstractmethod
    def execute_sentinel_plan(self, sentinel_plan: SentinelPlan, train_dataset: dict[str, Dataset], validator: Validator) -> SentinelPlanStats:
        """Execute a SentinelPlan according to strategy"""
        pass
