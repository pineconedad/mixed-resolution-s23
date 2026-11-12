# Resuming this study

Read `STUDY_CHECKLIST.md` first. It is the global execution checklist and restart handoff. Update it with completed builds, artifact paths, and the next command before ending work.

Current user policy: BUILD ONLY. Do not call Runpod tools, provision/start/stop/delete resources, use SSH, launch GPU workloads, or start cloud watchdogs. The user operates GPU machines manually. Provide exact commands in `docs/GPU_HANDOFF.md`. Existing local archives may be inspected and restored without contacting the cloud.

Do not rediscover the entire project or repeat archived experiments and smoke tests. Use focused checks only for changed code and the required checkpoint-backed model identity gate. The two root Markdown plans define the scientific requirements; never manufacture passes for absent historical inputs or bypass dataset licenses.

Never silently replace missing scientific inputs or weaken frozen-run gates to make a command succeed. Preserve the $3 reserve guidance for the user's manual operation. Historical cloud procedures are archived, not authorization for new cloud actions.
