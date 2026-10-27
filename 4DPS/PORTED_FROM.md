# Ported from

Source: `/home/adnan/Desktop/4DPS` (development checkout, READ-ONLY) at main `b51b91f`, copied 2026-10-27.

rsync excludes: .git outputs downstream models/hf webpage latex arxiv_latex misc baselines scene_editing paper_numbers *.so .build_stamp.json.
`seeds/`, `models/`, `third_party/` are copied but git-ignored (mirrored to the node by rsync).

All OctoNet adaptation happens in THIS copy. Never write to the source path.
