from __future__ import annotations

from stage1_bootstrap import bootstrap

bootstrap()

from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    build_step1c1_review_pack,
    print_step1c1_final_console,
)


def main() -> None:
    manifest = build_step1c1_review_pack()
    print_step1c1_final_console(manifest)


if __name__ == "__main__":
    main()
