# Franka reference

Use the Franka implementation as a tested adapter, not as a universal task schema.

## Fixed contract

- Task: `Isaac-Pick-Place-Franka-Joint-Position-Replay-Camera-v0`
- State dimension: 8
- Action dimension: 8
- Action horizon: 32
- Minimum intervention: 64 steps
- Cameras: `wrist_camera`, `table_camera`
- Gripper binary command: close `-1`, open `+1`
- Binary physical targets: close `0.0`, open `0.08`
- Success metric: bin-local object-footprint containment plus task diagnostics

Print the machine-readable task spec:

```bash
python scripts/benchmarks/gr00t/franka/franka_dagger_task.py
```

## Reference outcome matrix

| Representation | Close threshold | Base SR | Base + DAgger SFT SR | Net |
| --- | ---: | ---: | ---: | ---: |
| Continuous gripper width | `0.065 m` | 120/150 = 80.00% | 132/150 = 88.00% | **+8.00 pp** |
| Direct binary target | `0.04 m` | 123/150 = 82.00% | 120/150 = 80.00% | −2.00 pp |

The matrix comes from 50 fixed scenes × 3 paired flow-noise repeats with all automatic failures reviewed. It validates
the Continuous end-to-end DAgger stage and shows that binary targets are not automatically superior.

## Authoritative docs

- `docs/vla_dagger_guide.md`: complete commands and adaptation workflow.
- `docs/vla_dagger_reference.md`: architecture and transferable conclusions.
- `scripts/benchmarks/gr00t/franka/franka_manipulation_gr00t_commands.md`: broader Franka GR00T setup.

## Porting boundary

Robot-independent: gate semantics, transition schema, complete-horizon masks, normalization rules, CRN pairing, JSONL
integrity, and paired outcome reporting.

Franka-specific: joint/body discovery, joint limits, SpaceMouse IK, 8-D vector order, sorting objects/bins, scene
snapshot, containment geometry, and failure buckets.
