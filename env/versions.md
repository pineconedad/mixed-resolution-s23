# Source versions

Inspected and pinned on 2026-11-10. Submodules are immutable gitlinks, not floating branches.

| Component | Commit/revision |
| --- | --- |
| Original Pi3 code | `9fa3ddb3f8d53041f8b2738df404f62223bbaa7b` |
| Original Pi3 checkpoint | `yyfz233/Pi3` at `ae722e7039287d0c8fde9f11f197f804f44b510c` |
| Pi3 checkpoint SHA-256 | `33580e4702ac671558aedeab1148fd08118f7ce45bdbeb99f3e3cf340062875d` |
| DA3 code, inspection only | `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` |
| VGGT code, inspection only | `a288dd0f14786c93483e45524328726ab7b1b4ce` |

The local CPU environment uses Python 3.10 and torch 2.5.1, matching Pi3's declared torch version. CUDA environment capture happens on the GPU, not on this Mac. `requirements.txt` is the direct dependency pin set. `requirements.lock` records the verified local CPU dependencies, not a claimed complete CUDA lock. Run `scripts/record_environment.py` on each machine to capture the actual full environment.

Pi3 uses its PyTorch RoPE implementation. No compiled cuRoPE or xformers is installed. Native integer positions retain upstream RoPE; fractional aligned positions use the same formula evaluated at continuous coordinates. All experiment arms share the same code and precision.

VGGT is kept in a separate environment matching its repository's pins: torch 2.3.1, torchvision 0.18.1 and numpy 1.26.1. The public facebook/VGGT-1B checkpoint is pinned to revision 860abec7937da0a4c03c41d3c269c366e82abdf9. Tests live under tests_vggt/ and run explicitly with that environment; they are not silently skipped in the Pi3 test count.
