"""
Validator-based quality scores for DataRecordSet outputs (sentinel + final physical plans).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from palimpzest.constants import Cardinality
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import GenerationStats
from palimpzest.query.operators.convert import LLMConvert
from palimpzest.query.operators.filter import LLMFilter
from palimpzest.query.operators.join import JoinOp
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.topk import TopKOp
from palimpzest.validator.validator import Validator

logger = logging.getLogger(__name__)


def _is_perfect_quality_op(op: PhysicalOperator) -> bool:
    return (
        not isinstance(op, LLMConvert)
        and not isinstance(op, LLMFilter)
        and not isinstance(op, TopKOp)
        and not isinstance(op, JoinOp)
    )


def _join_input_ok(record_set: DataRecordSet) -> bool:
    inp = record_set.input
    return isinstance(inp, tuple) and len(inp) == 2 and isinstance(inp[0], list) and isinstance(inp[1], list)


def score_record_sets_with_validator(
    validator: Validator,
    source_indices_to_record_sets: dict[tuple, list[tuple[DataRecordSet, PhysicalOperator]]],
    max_workers: int | None,
) -> tuple[dict[tuple, list[tuple[DataRecordSet, PhysicalOperator]]], GenerationStats]:
    """
    Mutates ``record_op_stats[].quality`` on each ``DataRecordSet`` (same logic as sentinel scoring).

    Keys in ``source_indices_to_record_sets`` are only used for grouping; values hold
    ``(record_set, op)`` pairs to score.
    """
    futures, full_hashes, full_hash_to_bool_output = [], set(), {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _, record_set_tuples in source_indices_to_record_sets.items():
            for record_set, op in record_set_tuples:
                if _is_perfect_quality_op(op):
                    for record_op_stats in record_set.record_op_stats:
                        record_op_stats.quality = 1.0
                    continue

                if len(record_set) == 0:
                    record_set.record_op_stats[0].quality = 0.0
                    continue

                if isinstance(op, LLMConvert) and op.cardinality is Cardinality.ONE_TO_ONE:
                    if not isinstance(record_set.input, DataRecord):
                        logger.debug("Skipping map quality: DataRecordSet.input is not set (expected DataRecord)")
                        continue
                    fields = op.generated_fields
                    input_record: DataRecord = record_set.input
                    output = record_set.data_records[0].to_dict(project_cols=fields)
                    output_str = record_set.data_records[0].to_json_str(project_cols=fields, bytes_to_str=True, sorted=True)
                    full_hash = f"{hash(input_record)}{hash(output_str)}"
                    if full_hash not in full_hashes:
                        full_hashes.add(full_hash)
                        futures.append(executor.submit(validator._score_map, op, fields, input_record, output, full_hash))

                elif isinstance(op, LLMConvert) and op.cardinality is Cardinality.ONE_TO_MANY:
                    if not isinstance(record_set.input, DataRecord):
                        logger.debug("Skipping flat_map quality: DataRecordSet.input is not set (expected DataRecord)")
                        continue
                    fields = op.generated_fields
                    input_record: DataRecord = record_set.input
                    output, output_strs = [], []
                    for data_record in record_set.data_records:
                        output.append(data_record.to_dict(project_cols=fields))
                        output_strs.append(data_record.to_json_str(project_cols=fields, bytes_to_str=True, sorted=True))
                    full_hash = f"{hash(input_record)}{hash(tuple(sorted(output_strs)))}"
                    if full_hash not in full_hashes:
                        full_hashes.add(full_hash)
                        futures.append(executor.submit(validator._score_flat_map, op, fields, input_record, output, full_hash))

                elif isinstance(op, TopKOp):
                    if not isinstance(record_set.input, DataRecord):
                        logger.debug("Skipping topk quality: DataRecordSet.input is not set (expected DataRecord)")
                        continue
                    fields = op.generated_fields
                    input_record: DataRecord = record_set.input
                    output = record_set.data_records[0].to_dict(project_cols=fields)
                    output_str = record_set.data_records[0].to_json_str(project_cols=fields, bytes_to_str=True, sorted=True)
                    full_hash = f"{hash(input_record)}{hash(output_str)}"
                    if full_hash not in full_hashes:
                        full_hashes.add(full_hash)
                        futures.append(executor.submit(validator._score_topk, op, fields, input_record, output, full_hash))

                elif isinstance(op, LLMFilter):
                    if not isinstance(record_set.input, DataRecord):
                        logger.debug("Skipping filter quality: DataRecordSet.input is not set (expected DataRecord)")
                        continue
                    filter_str = op.filter_obj.filter_condition
                    input_record: DataRecord = record_set.input
                    output = record_set.data_records[0]._passed_operator
                    full_hash = f"{filter_str}{hash(input_record)}"
                    if full_hash not in full_hashes:
                        full_hash_to_bool_output[full_hash] = output
                        full_hashes.add(full_hash)
                        futures.append(executor.submit(validator._score_filter, op, filter_str, input_record, output, full_hash))

                elif isinstance(op, JoinOp):
                    if not _join_input_ok(record_set):
                        logger.debug("Skipping join quality: DataRecordSet.input is not (left_list, right_list)")
                        continue
                    condition = op.condition
                    for left_idx, left_input_record in enumerate(record_set.input[0]):
                        for right_idx, right_input_record in enumerate(record_set.input[1]):
                            record_idx = left_idx * len(record_set.input[1]) + right_idx
                            output = record_set.data_records[record_idx]._passed_operator
                            full_hash = f"{condition}{hash(left_input_record)}{hash(right_input_record)}"
                            if full_hash not in full_hashes:
                                full_hash_to_bool_output[full_hash] = output
                                full_hashes.add(full_hash)
                                futures.append(
                                    executor.submit(
                                        validator._score_join,
                                        op,
                                        condition,
                                        left_input_record,
                                        right_input_record,
                                        output,
                                        full_hash,
                                    )
                                )

    full_hash_to_score, validation_gen_stats = {}, GenerationStats()
    for future in as_completed(futures):
        score, gen_stats, full_hash = future.result()
        full_hash_to_score[full_hash] = score
        validation_gen_stats += gen_stats

    for _, record_set_tuples in source_indices_to_record_sets.items():
        for record_set, op in record_set_tuples:
            if _is_perfect_quality_op(op) or len(record_set) == 0:
                continue

            if isinstance(op, LLMConvert) and op.cardinality is Cardinality.ONE_TO_ONE:
                if not isinstance(record_set.input, DataRecord):
                    continue
                fields = op.generated_fields
                input_record: DataRecord = record_set.input
                output_str = record_set.data_records[0].to_json_str(project_cols=fields, bytes_to_str=True, sorted=True)
                full_hash = f"{hash(input_record)}{hash(output_str)}"
                record_set.record_op_stats[0].quality = full_hash_to_score[full_hash]

            elif isinstance(op, LLMConvert) and op.cardinality is Cardinality.ONE_TO_MANY:
                if not isinstance(record_set.input, DataRecord):
                    continue
                fields = op.generated_fields
                input_record: DataRecord = record_set.input
                output_strs = []
                for data_record in record_set.data_records:
                    output_strs.append(data_record.to_json_str(project_cols=fields, bytes_to_str=True, sorted=True))
                full_hash = f"{hash(input_record)}{hash(tuple(sorted(output_strs)))}"
                score = full_hash_to_score[full_hash]
                for record_op_stats in record_set.record_op_stats:
                    record_op_stats.quality = score

            elif isinstance(op, TopKOp):
                if not isinstance(record_set.input, DataRecord):
                    continue
                fields = op.generated_fields
                input_record: DataRecord = record_set.input
                output_str = record_set.data_records[0].to_json_str(project_cols=fields, bytes_to_str=True, sorted=True)
                full_hash = f"{hash(input_record)}{hash(output_str)}"
                score = full_hash_to_score[full_hash]
                record_set.record_op_stats[0].quality = score

            elif isinstance(op, LLMFilter):
                if not isinstance(record_set.input, DataRecord):
                    continue
                filter_str = op.filter_obj.filter_condition
                input_record: DataRecord = record_set.input
                output = record_set.data_records[0]._passed_operator
                full_hash = f"{filter_str}{hash(input_record)}"
                if output == full_hash_to_bool_output[full_hash]:
                    record_set.record_op_stats[0].quality = full_hash_to_score[full_hash]
                else:
                    record_set.record_op_stats[0].quality = 1.0 - full_hash_to_score[full_hash]

            elif isinstance(op, JoinOp):
                if not _join_input_ok(record_set):
                    continue
                condition = op.condition
                for left_idx, left_input_record in enumerate(record_set.input[0]):
                    for right_idx, right_input_record in enumerate(record_set.input[1]):
                        record_idx = left_idx * len(record_set.input[1]) + right_idx
                        output = record_set.data_records[record_idx]._passed_operator
                        full_hash = f"{condition}{hash(left_input_record)}{hash(right_input_record)}"
                        if output == full_hash_to_bool_output[full_hash]:
                            record_set.record_op_stats[record_idx].quality = full_hash_to_score[full_hash]
                        else:
                            record_set.record_op_stats[record_idx].quality = 1.0 - full_hash_to_score[full_hash]

    return source_indices_to_record_sets, validation_gen_stats


def apply_validator_to_record_sets(
    validator: Validator | None,
    max_workers: int | None,
    operator: PhysicalOperator,
    record_sets: list[DataRecordSet],
) -> None:
    """Score ``record_sets`` produced by ``operator`` when ``validator`` is set."""
    if validator is None or not record_sets:
        return
    mapping = {(i,): [(rs, operator)] for i, rs in enumerate(record_sets)}
    score_record_sets_with_validator(validator, mapping, max_workers)
