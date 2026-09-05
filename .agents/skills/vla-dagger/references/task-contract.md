# Task contract

Use this reference when adapting DAgger to a new VLA task.

## Required frozen fields

Record these before collection:

| Area | Fields |
| --- | --- |
| Environment | task ID, robot/entity names, action term, device, asset version |
| Policy | checkpoint ID, policy type, embodiment tag, language instruction |
| Observation | policy keys, tensor shapes, camera keys/resolution/FPS, history indices |
| Action | action keys, dimension/order, units, absolute/relative semantics, action horizon |
| Gripper | command index, close/open command values, physical targets, inference threshold |
| Collection | teleop device, takeover/reset controls, minimum intervention length |
| Success | automatic metric version, task objects/targets, stability requirement |
| Provenance | base dataset ID, code commit, modality config, normalization hashes |

Represent scalar choices with `scripts/benchmarks/gr00t/dagger/task_spec.py`. Keep environment behavior in callbacks.

## Adapter callbacks

Implement and unit-test:

```text
observation(env) -> dict[str, ndarray]
policy_action(model_output) -> ndarray[action_dim]
expert_action(teleop_input, env) -> ndarray[action_dim]
success(env) -> bool + diagnostics
```

The recorder operates on canonical flat state and action vectors with independent `state_dim` and `action_dim`. If the
environment has multiple action terms, define the flatten/unflatten order once and store it in the task spec.

## Time alignment

Verify with an impulse test:

```text
state[t] -- command[t] --> achieved_state[t+1]
```

For a few frames, print or plot command, next achieved state, image timestamp, and recorder frame index. Check the
gripper separately because contact and mechanical travel can make its achieved state lag the command.

## Representation decision

Choose the target consumed by the inference decoder:

- command target: preferred when the true controller command is recorded;
- next achieved state: usable only when training and inference interpret it consistently;
- relative action: define the exact reference state and quaternion convention;
- binary gripper: still uses a continuous model head unless a classifier head is explicitly implemented.

Do not change representation and add DAgger data in the same comparison unless the user explicitly wants the combined
stage result. When representation changes, rebuild base and recovery data together.

## Porting the Franka implementation

Keep:

- `InterventionGate` and horizon-mask logic;
- complete-rollout HDF5 schema;
- JSONL run provenance;
- deterministic inference-seed protocol;
- paired analyzer integrity checks.

Replace:

- Franka joint IDs, body IDs, limits, and SpaceMouse IK;
- state/action extraction and GR00T key mapping;
- task object snapshots, failure buckets, and containment success;
- action dimension checks in Franka-only conversion code.
