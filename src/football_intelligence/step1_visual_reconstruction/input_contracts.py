from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from football_intelligence.step1_visual_reconstruction.io import (
    STEP1B4_REVIEW_DECISION_TEMPLATE_PATH,
    STEP1B4_STEP1C_INPUT_CONTRACT_PATH,
    write_json,
    write_text,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
)


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def step1c_input_contract_markdown() -> str:
    return "\n".join(
        [
            "# Step1.C Input Contract",
            "",
            f"- Warning: `{VISUAL_ONLY_WARNING}`.",
            "- Step1.C must consume `step1b4_visible_person_base_rows.json`, not raw Step1.B rows.",
            "- Step1.C may add team-colour belief candidates only.",
            "- Step1.C must retain unknown and ambiguous team-colour states.",
            "- Step1.C must preserve candidate provenance from Step1.B4.",
            "- Step1.C must not assign player slots.",
            "- Step1.C must not do identity tracking.",
            "- Step1.C must not create expected 22-role states.",
            "- Step1.C must not use 2D projection as metric truth.",
            "- Step1.C must not calculate football conclusions or physical/tactical metrics.",
            "- Step1.C must keep `visual_only_warning=VISUAL_ONLY_NOT_METRIC`.",
            "- Step1.C must keep `production_ready=false`.",
            "- B3/B4 count policy is a visual QA count policy only, not football analysis.",
            "",
            "## Required Consumer Behaviour",
            "",
            "- Use `visible_person_base_id` as the row-level visual candidate key.",
            "- Treat `candidate_type`, `original_role_source`, and source labels as provenance only.",
            "- Do not force unknown/context rows into player or team roles.",
            "- Do not emit player-slot identifiers or identity identifiers.",
            "- Preserve review flags for ambiguous source-disagreement rows.",
        ]
    ) + "\n"


def review_decision_template_payload() -> dict[str, Any]:
    return {
        "reviewer_name": "",
        "reviewed_at": "",
        "contact_sheet_reviewed": False,
        "approve_b4_as_step1c_input_candidate": False,
        "known_issues": [],
        "frames_requiring_manual_followup": [],
        "notes": "",
        "production_ready": PRODUCTION_READY,
        "no_auto_promotion": True,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "created_at": utc_iso(),
    }


def build_and_write_step1c_input_contracts() -> tuple[str, dict[str, Any]]:
    contract = step1c_input_contract_markdown()
    decision_template = review_decision_template_payload()
    write_text(STEP1B4_STEP1C_INPUT_CONTRACT_PATH, contract)
    write_json(STEP1B4_REVIEW_DECISION_TEMPLATE_PATH, decision_template)
    return contract, decision_template
