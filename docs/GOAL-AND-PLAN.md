# Goal & Plan — lingbot-world-v2 interactive events

**Status: living document.** The goal and plan are decisions so far, not
promises — they may change as we learn. Update this file whenever they do.
Written: 2026-11-07. Restructured: 2026-11-08 (interaction surface first,
physics second; stay on 1.3B). Updated: 2026-11-09 — Phase B rescope
(RealWonder replication already done), settled-conclusions section added
from the Q&A on physics/novelty. **User note: goal & plan may be revised
again based on the Q&A insights — sections marked OPEN are candidates, not
commitments.**

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
  OPEN QUESTION first: `action.npy` has 961 rows but the video has 269
  frames — the time mapping is unknown; resolve before wiring.
- **A3: scripted camera/action generation** — procedurally generate
  `poses.npy` / `wasd_action.npy` / `intrinsics.npy` from the same event
  format (orbit, dolly, walk-forward, …).
- **A-test**: lake scene (`examples/03`), 269 frames, one text event +
  one action event. Check: scene changes visibly? quality stable after a
  cache swap?

### Phase B — physics via RealWonder (after A)

**Already done by another agent (Oct 2026), verified on disk:** the offline
pipeline ran end-to-end on 3 of 6 author cases — lamp, persimmon, two_duck
(final videos in `RealWonder/result/`), with 16 GB GPU / 28 GB RAM VM
adaptations committed to `Gxmestk/RealWonder` (`RUN_LOCAL.md`). NOT done:
any novel (non-author) scene, the interactive demo, and the `tree` /
`santa_cloth` / `sand_house` cases.

So the remaining Phase B work:
- **B1: exercise the setup stage on one of our own images** — first-ever
  attempt here; measures the real hand-setup minutes (paper says users
  override VLM estimates; our persimmon config is exactly such an override).
  (Independent of Phase A — can run in parallel if wanted.)
- **B2: SDEdit-into-lingbot obedience test** — feasibility unknown (a). The
  replication used RealWonder's own distilled model, so whether LINGBOT
  causal_fast obeys injected sim frames is still untested. Key experiment in
  `lingbot-realwonder-notes.md` §10.
- **B3: live force loop** — RealWonder's released demo receives FORCES only
  live (camera/robot inputs are paper-level, not in the release). Decide the
  path: RealWonder demo loop vs injection into lingbot; reuses A's event
  queue either way.

### Core idea both phases share

One event queue keyed by chunk index:
`Event = { when: chunk_idx, type: text|camera|wasd|action|force, data }`.
Phase A fills receivers that already exist; Phase B adds the force receiver.

---

## Settled conclusions & rejected ideas (2026-11-09 Q&A)

Facts and decisions from the Q&A session, so dead ends are not re-proposed:

1. **RealWonder is NOT a world model** — it is a classical physics simulator
   (Genesis: hand-coded dynamics, explicit 3D state, deterministic/resettable)
   + a generative video model used as a realistic renderer (SDEdit). Its
   paper positions AGAINST world models: they "lack architectural components
   to process such actions directly" (realwonder.txt:144). lingbot IS a
   world model (dynamics learned in the weights, no explicit state).
2. **Neither system covers arbitrary events.** lingbot text events are
   bounded by training distribution (no precise quantities/objects/timing).
   RealWonder is bounded by the sim catalog + reconstruction quality. An
   arbitrary request falls in the gap between them.
3. **RealWonder DOES estimate materials** — a VLM (GPT-4V) classifies each
   object and estimates density/friction/moduli as initial values; users can
   override (realwonder.txt:272-278). It is appearance-guessing only — it
   never corrects from observed motion.
4. **REJECTED (single-image setting): system-identification fitting
   ("candidate 1") and world-model-as-active-probe-generator ("candidate
   2")** — both are circular: fitting sim params against model-generated
   motion measures the model's beliefs, not real physics; identifiability
   collapses ("object is soft" vs "model imagines soft"). The user caught
   this. Backing evidence: best video models score only ~22–40% on physical
   commonsense (VideoPhy 2024: ~39.6% best; VideoPhy-2 ICLR 2026: ~22%,
   top models CogVideoX & Wan2.1-14B) — and our 1.3B causal-fast is a
   low-fidelity, 4-step-distilled tier of that family.
   - **If that novelty is ever pursued**, the honest setting is: REAL input
     video → fitted params → interactive world. That is a different product
     from RealWonder's single-image one (system-ID from video is classical
     robotics; only the integration would be new).
   - Salvageable residue: sim-vs-model agreement as a *belief-consistency
     check* (plausibility flag, never measurement).
5. **RealWonder live inputs**: released demo = forces only (socket handler
   `update_forces`, applied per physics step — genuinely live, streaming
   architecture with producer threads). Camera poses and Franka gripper
   states are paper-level (realwonder.txt:166, applied at render :294), not
   in the released demo. Both systems are causal-streaming: input at time t
   affects only output after t.
6. **RealWonder replication status** (see Phase B) — done: offline batch on
   3 author cases; not done: novel-scene setup, live demo, lingbot-side
   SDEdit test.

## Related docs

- `docs/lingbot-realwonder-notes.md` — discussion notes, verified
  file:line facts, RealWonder analysis.
- `RUNBOOK-1.3B-causal-fast.md` — how to run the 1.3B model here.
- `../video-gen-physics/RealWonder/RUN_LOCAL.md` — how RealWonder runs on
  this VM (other agent's work).
