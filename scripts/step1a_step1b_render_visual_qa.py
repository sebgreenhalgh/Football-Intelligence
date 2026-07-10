from __future__ import annotations

from stage1_bootstrap import bootstrap

bootstrap()

from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    build_review_pack,
    load_candidate_inventory,
    load_person_states,
    print_final_console,
)
from football_intelligence.step1_visual_reconstruction.qa_render import render_visual_qa_contact_sheets  # noqa: E402


def main() -> None:
    candidate_payload = load_candidate_inventory()
    state_payload = load_person_states()
    render_visual_qa_contact_sheets(candidate_payload, state_payload)
    manifest = build_review_pack(candidate_payload=candidate_payload, state_payload=state_payload)
    print_final_console(manifest)


if __name__ == "__main__":
    main()
