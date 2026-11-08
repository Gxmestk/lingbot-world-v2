# Goal & Plan — lingbot-world-v2 interactive events

**Status: living document.** The goal and plan are decisions so far, not
promises — they may change as we learn. Update this file whenever they do.
Date written: 2026-11-07.

---

## Goal

Make lingbot-world-v2 accept scripted events **during a run** (mid-stream),
the way RealWonder drives its scenes:

- Premade text events that change the scene at a chosen chunk
- Premade camera/action events
- Later (maybe): physics/force events via the RealWonder sim stage

Chosen first target: **combo 3** — one image + launch text, then premade
text events fire mid-run. Live (real-time) variants are parked: ~1.1 s/frame
on the 1.3B GPU → ~18 s per 16-frame chunk → ~0.9 fps, far below 24 fps.

## Why this shape

lingbot's model only has 3 receivers: text, camera/action tensors, past
frames. No object/physics receiver. So:

| event type | receiver exists? | effort |
|---|---|---|
| text | yes (`wan/image2video.py:706-718`, cross-attn rebuild `:673-675`) | small — re-encode + rebuild cache ~1–2 s |
| camera/action | yes (`poses.npy`, `wasd_action.npy`) | small — generate the arrays |
| force/physics | **no** | big — needs RealWonder stage: SAM2 → SAM-3D → Genesis → SDEdit injection (inference-time only, no fine-tune) |

## Plan (order matters)

1. **Text-event runner** — `generate_events.py` taking
   `[{when: chunk_idx, text: "..."}]`; before each chunk, if an event fires,
   T5 re-encode + cross-attn K/V rebuild + drop self-attn cache so the new
   text enters cleanly. (~2–4 s extra per event.)
2. **Test run** — lake scene (`examples/03`), 269 frames, events at chunk 4
   and 9. Check: scene actually changes? quality stable after switch?
3. **Camera/action scripting** — generate pose/action arrays for the same
   event format. Low effort.
4. **Force path decision** — only after 1–3. Key unknowns
   (see `lingbot-realwonder-notes.md` §10):
   (a) does lingbot causal_fast obey SDEdit-injected sim frames?
   (b) is ~18 s/chunk latency acceptable?

Everything funnels through one core: a per-chunk event queue. Steps 1–3 use
existing receivers; step 4 reuses the same injection loop the earlier steps
validate.

## Related docs

- `docs/lingbot-realwonder-notes.md` — full discussion notes, verified
  file:line facts, RealWonder analysis, feasibility unknowns.
- `docs/RUNBOOK-1.3B-causal-fast.md` — how to run the 1.3B model here.
