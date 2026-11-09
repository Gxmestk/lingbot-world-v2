# Goal & Plan — lingbot-world-v2 interactive events

**Status: living document.** The goal and plan are decisions so far, not
promises — they may change as we learn. Update this file whenever they do.
Written: 2026-11-07. Restructured: 2026-11-08 (finish lingbot's own
interaction surface first, physics second; stay on 1.3B).
Updated: 2026-11-09 (Phase B rescope — RealWonder replication already done;
settled-conclusions section added; action.npy open question).

---

## Goal

Recreate the paper's interactive features (arXiv 2607.07534) on
**lingbot-world-v2-1.3b-causal-fast, 1 GPU** — the features the released
weights actually support, at batch speed — then add RealWonder-style physics
on top.

**Not in scope:** 14B models (need ~8 GPUs), real-time 60 fps serving
(not released), full agentic harness (not released — optional stretch later).

### Paper feature vs release status (checked 2026-11-08)

| paper feature | released? | fact |
|---|---|---|
| causal chunked generation | ✅ | replicated: 269 frames, 9m41s |
| camera (poses+intrinsics) | ✅ wired | `wan/image2video.py:647,720` |
| WASD movement | ✅ wired | `:756-766`, 4-dim per-frame channel fused into plucker |
| mid-run text events | ⚠️ model yes, code no | batch script: one prompt only |
| attack/archery/spell/shoot actions | ⚠️ data yes, code no | `examples/03/action.npy` is (961,4) binary flags; **no released code reads `action.npy`** |
| real-time 60 fps | ❌ | not released (SGLang/flashdreams pointers only) |
| agentic harness | ❌ | paper concept only |

## Plan

### Phase A — finish lingbot's interaction surface (batch speed)

- **A1: mid-stream text events** — `generate_events.py` taking
  `[{when: chunk_idx, text: "..."}]`; before each chunk, if an event fires:
  T5 re-encode (~1–2 s) + cross-attn K/V rebuild (`image2video.py:673-675`)
  + drop self-attn cache so the new text enters cleanly.
- **A2: wire the action channel** — make the model accept `action.npy`
  (attack/shoot-style 4-dim flags), mirroring the WASD path at
  `:756-766`. Needs one experiment: does the DiT accept a second 4-dim
  channel (or is WASD the only trained port)?
  **Open question first:** `action.npy` has 961 rows but the video has 269
  frames — the time mapping is unknown; resolve before wiring.
- **A3: scripted camera/action generation** — procedurally generate
  `poses.npy` / `wasd_action.npy` / `intrinsics.npy` from the same event
  format (orbit, dolly, walk-forward, …).
- **A-test**: lake scene (`examples/03`), 269 frames, one text event +
  one action event. Check: scene changes visibly? quality stable after a
  cache swap?

### Phase B — physics via RealWonder (after A)

**Already done by another agent (Oct 2026)** — verified in
`../video-gen-physics/RealWonder/`: offline pipeline end-to-end on 3 author
cases (lamp, persimmon, two_duck; final videos in `result/`), with 16 GB
GPU / 28 GB RAM VM adaptations committed to the Gxmestk fork (`RUN_LOCAL.md`).
**NOT done there:** novel (non-author) scenes, live demo, tree/santa_cloth/
sand_house runs. RealWonder replication status is therefore:
"batch path on author-configured scenes ✅; setup stage ❌ never exercised;
live loop ❌ never run".

Remaining Phase B work:
- **B1: exercise the setup stage on one of our own images** — first-ever
  attempt here (fresh photo → clicks → masks → config); measures the real
  "10–30 min hand-setup" number. Independent of Phase A — can run in
  parallel if we choose.
- **B2: SDEdit-into-lingbot obedience test (unknown (a))** — the replication
  used RealWonder's own distilled model, so lingbot obedience is still
  untested. Key experiment unchanged (notes §10): warp lingbot's first frame
  along a hand-made trajectory, SDEdit-inject into ONE lingbot chunk
  (denoise 300→100), compare obedience vs no injection.
- **B3: live force loop** — RealWonder's released demo receives **forces
  only** live (`demo_web/app.py` handlers; camera/robot are paper-level
  only). Decide: RealWonder demo path vs lingbot-injection path; reuses A's
  event queue.

### Core idea both phases share

One event queue keyed by chunk index:
`Event = { when: chunk_idx, type: text|camera|wasd|action|force, data }`.
Phase A fills receivers that already exist; Phase B adds the force receiver.

## Settled conclusions & rejected ideas (2026-11-09 Q&A session)

Do not re-propose these without new evidence:

- **RealWonder is NOT a world model.** It is a classical physics simulator
  (Genesis: hand-coded dynamics, explicit 3D state) + a generative video
  model used as a realistic renderer (SDEdit photorealization). Its own
  paper motivates this by saying world models "lack architectural components
  to process such actions directly" (realwonder.txt:144). Lingbot, by
  contrast, is a world model: dynamics learned in the weights, no explicit
  state/objects.
- **Neither system covers arbitrary events.** Lingbot text events are
  bounded by training-data distribution (global/qualitative changes OK;
  precise quantities, exact timing, object-specific targeting are not).
  RealWonder is bounded by the sim-engine catalog + reconstruction quality
  (rigid/cloth/granular only, approximate contact, hand-set params with VLM
  appearance-prior guesses that users override — paper :272-278).
- **REJECTED: motion-fitting (candidate 1) and lingbot-as-probe-generator
  (candidate 2) in the single-image setting.** Both are circular: fitting
  sim parameters against model-generated motion measures the model's
  beliefs, not real physics ("cannot distinguish 'the fruit is soft' from
  'the model imagines soft'"). Evidence that video models' physical
  commonsense is weak: best T2V models score only ~39.6% (VideoPhy) and
  ~22% (VideoPhy-2, ICLR 2026) joint semantic+physical adherence vs humans;
  top models there include Wan 2.1-14B — lingbot's parent family — and our
  1.3B causal-fast is a lower-fidelity distilled tier of it.
- **If that novelty is ever wanted**, the honest setting is: **real input
  clip → fitted params → interactive world** (a video-input product —
  different from RealWonder's single-image one). Otherwise we are an honest
  replication/integration project, which is already valuable.
- **Vocabulary to keep straight:** lingbot `poses.npy` = scene-agnostic
  camera data ≠ RealWonder event scripts = scene-specific knowledge.
  "Hand-tuned" RealWonder cases = staged inputs + automation gaps, not
  cheating.

## Related docs

- `docs/lingbot-realwonder-notes.md` — discussion notes, verified
  file:line facts, RealWonder analysis.
- `RUNBOOK-1.3B-causal-fast.md` — how to run the 1.3B model here.
- `../video-gen-physics/RealWonder/RUN_LOCAL.md` — how RealWonder runs here.
