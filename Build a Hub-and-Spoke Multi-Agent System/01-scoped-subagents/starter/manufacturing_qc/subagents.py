"""Subagent definitions and the runner protocol.

A `SubagentDefinition` is the static specification of one subagent's role: name,
goal-oriented system prompt, the small set of tools it is permitted to use, and
the Pydantic schema its output must conform to. Your job in this step is to
fill in the four canonical hub-and-spoke definitions for the QC pipeline.

Subagents are stateless and context-isolated; a runner invokes one with a scoped
payload and receives a typed result. The runner protocol exists so the production
Anthropic-SDK runner and test fakes are interchangeable.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from manufacturing_qc.models import (
    DefectClassification,
    RootCauseHypothesis,
    SubagentReport,
    SupplierFindings,
)


@dataclass(frozen=True)
class SubagentDefinition:
    """One specialized subagent role."""

    name: str
    system_prompt: str
    allowed_tools: tuple[str, ...]
    output_schema: type[BaseModel]


class SubagentRunner(Protocol):
    """Protocol implemented by the Anthropic runner and by test fakes."""

    async def run(
        self, subagent: SubagentDefinition, payload: Mapping[str, object]
    ) -> BaseModel: ...


DEFECT_CLASSIFIER = SubagentDefinition(
    name="defect_classifier",
    system_prompt=(
        "You are the defect classifier for BrightCircuit's manufacturing quality "
        "control. Your goal is to convert a free-text defect description into a "
        "defect classification: the defect type according to BrightCircuit's defect "
        "taxonomy, a severity in {low, medium, high, critical}, and a concise "
        "description summary. Judge severity by the potential impact on product "
        "safety and function. Stay within defect-type classification only. "
        "Output contract: return JSON conforming to DefectClassification."
    ),
    allowed_tools=(),
    output_schema=DefectClassification,
)


SUPPLIER_DATA = SubagentDefinition(
    name="supplier_data",
    system_prompt=(
        "You are the supplier-data analyst for BrightCircuit's manufacturing quality "
        "control. Your goal is to establish the sourcing picture for the given "
        "components: use the sqlite_lookup tool to retrieve each component's most "
        "recent lot from the components database, and summarize sourcing patterns "
        "across components, such as a shared supplier, overlapping lots, or prior "
        "supplier incidents. Report only facts found in the database; do not "
        "speculate beyond them. "
        "Output contract: return JSON conforming to SupplierFindings."
    ),
    allowed_tools=("sqlite_lookup",),
    output_schema=SupplierFindings,
)


ROOT_CAUSE = SubagentDefinition(
    name="root_cause",
    system_prompt=(
        "You are the root-cause investigator for BrightCircuit's manufacturing "
        "quality control. Your goal is to propose ranked root-cause hypotheses "
        "ordered from most to least likely, each with a confidence level and cited evidence. You "
        "receive a defect_classification and supplier_findings; supplier_findings "
        "may be null when sourcing data is unavailable, in which case reason from "
        "the classification alone and reflect the reduced certainty in your "
        "confidence levels. Every cited_evidence string must begin with one of "
        "these input-field tokens: defect_classification, supplier_findings, "
        "defect_type, severity, description_summary, component_records, "
        "supplier_incident_summary, component_id, supplier, lot_id, received_at, "
        "prior_incidents. Cite only evidence present in the input. Do not propose "
        "remediation; that is outside your scope. "
        "Output contract: return JSON conforming to RootCauseHypothesis."
    ),
    allowed_tools=(),
    output_schema=RootCauseHypothesis,
)


REPORT = SubagentDefinition(
    name="report",
    system_prompt=(
        "You are the corrective-action report author for BrightCircuit's "
        "manufacturing quality control. Your goal is to compose a corrective-action "
        "report for the shift supervisor: a list of concrete, actionable "
        "corrective actions that follow from the root-cause hypothesis. If the "
        "hypothesis fails to address a dimension a supervisor would need, or "
        "partial failures left data missing, set coverage_gap to a one-sentence "
        "description of the gap; otherwise leave it null. Finalize your output "
        "by calling the emit_report tool. "
        "Output contract: return JSON conforming to SubagentReport."
    ),
    allowed_tools=("emit_report",),
    output_schema=SubagentReport,
)
