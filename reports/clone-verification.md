# Fresh-clone verification

Verified the published private GitHub repository at commit `37fbf5b330d666c3cb777ebe1cfe025ed8f061a3` in a separate directory, not the preparation worktree.

1. Clone `https://github.com/pineconedad/mixed-resolution-s23.git` with `--recurse-submodules`. All pinned upstream commits check out, including DA3's nested SALAD submodule.
2. Run `python -m pytest -q` against that clone using the existing Python-3.10 preparation environment: 68 passed in 1.40 seconds. Pytest imports the cloned source through its configured pythonpath.
3. Verify the archive downloaded through GitHub against SHA-256 `a947f97c055c980d95e829858ecdd4139ae65e24aa2419aa7d902b9145bc7868`: all 41 member checksums pass and restore succeeds.
4. Run `scripts/audit_results.py` against the cloned exported results: every one of the 20 camera-pose evaluations rescores successfully, and token/provenance checks pass.

This validates published code and artifacts, including byte-preserved CSVs. It does not claim a new clean Python-environment installation or another GPU experiment. No GPU resources were provisioned for this check.
