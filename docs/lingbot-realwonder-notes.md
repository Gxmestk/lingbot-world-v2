# LingBot-World-v2 × RealWonder — discussion notes (Nov 2026)

Verified facts from code and this conversation, with file:line references.
Companion doc to `RUNBOOK-1.3B-causal-fast.md`.
**Current goal & plan live in `docs/GOAL-AND-PLAN.md`** (living document —
may change); this notes doc is the evidence/facts behind it.

---

## 1. Replication status (done)

- Fork of Robbyant/lingbot-world-v2, pushes to Gxmestk/lingbot-world-v2 (upstream push blocked).
- **1.3B causal-fast replicated on 1 GPU** (Sep 26), both runs EXIT_RC=0:
  - smoke: 81 frames → `output/smoke_1p3b_81f_la18.mp4` (6.3 MB, 6m11s)
  - full: 269 frames → `output/full_1p3b_269f_la18.mp4` (19 MB, 9m41s)
- Local commits: SDPA instead of flash-attn (3cf1af2), runbook (6697ca2),
  VAE context video on GPU (eb65390), streamed loading, 28 GB RAM cap
  (2b17bfc, ff29a1c), T5 bf16 (a0ff4c6).
- 14B: only T5+VAE downloaded (no DiT) — 14B never run here.
- Video server: `python -m http.server 8123` in `output/` →
  http://10.1.240.16:8123/ or http://10.2.1.169:8123/ (index.html = 2 video
  players). Session-bound; killed once by low memory; no persistent service.

## 2. How lingbot takes input today

- One text prompt at launch, T5-encoded once (`wan/image2video.py:706-718`).
- Premade action files via `--action_path`: `poses.npy`, `wasd_action.npy`,
  `intrinsics.npy`.
- Chunked generation with KV cache: 269 frames → 68 latent frames → 17
  chunks × 4 latent frames (16 pixel frames) each.
- Batch script only — no live input path.
- Mid-run text change = T5 re-encode + cross-attn K/V rebuild
  (`image2video.py:673-675`), ~1–2 s per change. Self-attn history cache
  stays valid.
- Official real-time deployment (60 fps, mid-run events) not released.

## 3. The 4 input combinations

| # | text | camera/action | result |
|---|------|---------------|--------|
| 1 | launch-only | premade | batch video (current) |
| 2 | launch-only | live | slow-mo interactive, fixed text |
| 3 | premade mid-stream text events | premade | batch video, scripted scene changes — **chosen plan** |
| 4 | live events | live | slow-mo interactive + live events |

Live camera/action is the easy part (small per-chunk tensors). The wall is
latency, not difficulty: ~1.1 s/frame → 16-frame chunk ≈ 18 s → ~0.9 fps vs
24 fps (~25x too slow). All combos need the same core: per-chunk injection.

## 4. Combined plan (combo 3 + RealWonder)

Two stages:
- **Setup** (human, ~10–30 min/scene, once): clicks → SAM2 masks → SAM-3D
  reconstruction → mesh fixes → Genesis scene.
- **Control** (general, live): user forces via per-chunk event queue.

Interface sketch:
```
Event = { when: chunk_idx, type: text|force|camera|action, data: ... }
```
Receiver dictionary:
- camera/action → works now (poses/WASD slots).
- text → small refactor: T5 re-encode + cross-attn cache rebuild (~1–2 s).
- force → **no lingbot receiver** (no objects, only pixels). Needs the
  RealWonder stage: setup + sim + SDEdit injection. No fine-tuning needed
  (RealWonder injects at inference time only).

## 5. RealWonder pipeline (arxiv 2603.05449, Stanford+USC)

Repo: `../video-gen-physics/RealWonder`. One image →
1. SAM2 segmentation (user clicks select objects)
2. 3D reconstruction (facebook/sam-3d-objects) → meshes
3. Genesis physics sim (gravity, forces, materials)
4. Render sim trajectory → warped RGB → SDEdit into video model

Video model: Wan2.1-Fun-V1.1-1.3B-InP, distilled by the authors into a
4-step causal AR model (`Realwonder-Distilled-AR-I2V-Flow`,
frame_per_block=3). SDEdit conditioning is inference-time only (no new
network input port): sim frames noised to 300/1000, denoised 300→200→100
(`vidgen/pipeline_sdedit.py`). Model learns: 4-step speed, causal streaming,
faithfulness to injected sim frames. Lingbot causal_fast inherits 2 of 3
(speed + streaming); faithfulness to sim-frame conditioning is unproven for
lingbot.

## 6. Persimmon case anatomy

`cases/persimmon/config.yaml` = the world (48 lines): click points, mask
candidate index (`all_object_masks_idx`, segmenter.py:263), `object_id`
mesh grouping [[3,11],[4],[5,10]] (fruit split in two by reconstruction,
hand-grouped; unused by SAM2 path, only repvit branch reads it,
`single_view_reconstructor.py:123-128`), physics (dt 0.01, gravity -9.8,
ρ=1000, frictions 0.01/1.0), 81 sim frames / 21 output frames per block,
denoising [300,200,100], `vgen_prompt`, `crop_start: 165`
(`simulation/utils.py:340`).

`persimmon.py` = the event script: pushes ±x ×100 N while `sid <= 5`
(called per physics step, genesis_simulator.py:236), z-nudges +2.5/+1.5 cm.
Forces are never in YAML — they're behavior ("when, what order").

## 7. Case comparison (hand-tuning level)

| case | lines | event |
|------|-------|-------|
| two_duck | 28 | feedback: force = 0.2 × direction between live duck positions each step — closest to general |
| lamp | 19 | constant push +x 1.0 N, forever |
| persimmon | 31 | timed pushes + z-nudges |
| tree | 58 | wind from mesh bbox ×1.5, pins trunk |
| santa_cloth | 100 | pins runtime-found particles, custom force func |
| sand_house | 234 | robot arms, hand-placed poses — most authored |

## 8. Two stages, and are hand-fitted scenes OK?

**Setup** is hand-fitted today. "Not general" = "not yet automated":
mask-pick easy (top-scoring mask), mesh grouping medium (by mask overlap),
settling medium, force_scale hardest. Root blocker = reconstruction quality.

**Control** is already general (demo_web `update_forces`, app.py:118-133,
applies per-object vectors live, base.py:119-135).

Acceptability: for us — 10–30 min/scene paid once, then unlimited live
control: sound. Creative-tool product: expected (onboarding). Mass-market
"any photo instantly": not acceptable. Paper: OK only if disclosed
(disclosure unverified).

## 9. Vocabulary note ("cheating")

Three things lumped under one word earlier:
1. Scripted demo events = staged inputs — normal, not cheating.
2. Hand-patching pipeline failures = automation gap, not deception.
3. Clicks/sliders = designed user input, like lingbot's text prompt.

Note: lingbot `poses.npy` (scene-agnostic camera data) ≠ RealWonder event
scripts (scene-specific knowledge). Categorically different.

## 10. Feasibility unknowns + key experiment

- (a) Does lingbot causal_fast obey SDEdit-injected sim frames? Never
  trained for sim-frame conditioning (only camera/WASD/text).
- (b) Latency ~18 s/chunk on 1.3B GPU — acceptable for slow-mo interactivity?
- Key experiment: warp lingbot's first frame along a hand-made trajectory,
  SDEdit-inject the noised render into ONE lingbot chunk (denoise
  300→100), compare obedience vs no injection.

## 11. File pointers

- lingbot chunk loop: `wan/image2video.py` — T5 cache :706-718, cross-attn
  cache :673-675.
- RealWonder: `../video-gen-physics/RealWonder/` — `simulation/` (Genesis),
  `vidgen/pipeline_sdedit.py`, `demo_web/` (app.py update_forces,
  case_handlers/base.py apply_forces).
- Paper text: `../papers/realwonder.txt`.
