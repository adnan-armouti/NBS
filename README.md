# NBS

RF inverse rendering on OctoNet (SFCW radar, UWB, Wi-Fi CSI): the 3DPS point-splat renderer with
the SFCW/bistatic extensions, plus a snapshot of 4DPS for its human-in-scene preprocessing.

| Dir | What |
|---|---|
| `nbs/` | the NBS code: `mm3DPS/` (3DPS renderer, SFCW mode), `method/` (SFCW / WiFi forward models, experiments), `preprocessing/` (OctoNet loaders, config, sync) |
| `4DPS/` | read-only snapshot of the 4DPS development checkout (see `4DPS/PORTED_FROM.md`); OctoNet adaptation happens here |
| `docs/` | `WRITE_UP.md` (obstacle log), plans, `prompts/` (task prompts), `node_migration_plan.md` |
| `pdf/` | the OctoNet paper (only source of the node extrinsics) |
| `viz/` | figures |
| `tools/env_node.sh` | environment on the Brev node `mmsd-h100` |

Development runs on `mmsd-h100` (`/data/mmsd/NBS/code`, data at `/data/mmsd/datasets/OctoNet`).
See `CLAUDE.md` for the working rules.
