# RUNBOOK — LingBot-World 2.0 · 1.3B `causal_fast` on one RTX 4060 Ti (16 GB)

Operational guide for generating the `examples/03` lakeside video on this machine
(Incus container `s-gtksoon1`, 28 GiB cgroup RAM, no swap). Everything below was
executed and measured on 2026-09-25/26; the full investigation history lives in
the execution plan (`/root/.claude/plans/try-this-one-https-huggingface-co-robbya-synthetic-gem.md`).

## 1. What runs here

| Part | Choice |
|---|---|
| Model | `robbyant/lingbot-world-v2-1.3b-causal-fast` (DiT only, 6 shards, 6.84 GB) |
| Assets | from `robbyant/lingbot-world-v2-14b-causal-fast`: T5 `models_t5_umt5-xxl-enc-bf16.pth` (+ converted `.safetensors` sibling), `Wan2.1_VAE.pth`, `google/umt5-xxl/` |
| Mode | `causal_fast` — distilled, 4 denoiser steps/chunk, no CFG |
| Split | T5 on CPU (`--t5_cpu`), DiT + VAE on GPU |
| Cache | sliding window `--local_attn_size 18` + sinks `--sink_size 6` — **the window, not the frame count, caps GPU memory** |

## 2. Code state (all committed on `main`)

| Commit | What | Revert |
|---|---|---|
| `579677c` | `.gitignore` for model dirs + `.venv` | `git revert 579677c` |
| `ff29a1c` | mmap state dict + shard-wise DiT load | `git revert ff29a1c` |
| `a0ff4c6` | T5 built directly in bf16 | `git revert a0ff4c6` |
| `2b17bfc` | streaming weight loading (T5 safetensors `pread` + del-before-drop DiT shards) | `git revert 2b17bfc` |
| `eb65390` | VAE context video built on GPU (269-frame CPU burst fix) | `git revert eb65390` |
| *(uncommitted)* | 1-line SDPA swap at `wan/modules/model_fast.py:208` (`flash_attention`→`attention`) | `git checkout -- wan/modules/model_fast.py` |

**The T5 `.safetensors` sibling is required** (`t5.py` falls back to mmap without
it, which peaks at model+file ≈ 25–27 GiB — inside the host-watchdog kill band).
Regenerate with `output/convert_t5_safetensors.py` if lost (peak 15.2 GiB, every
tensor byte-verified).

**Never** install `flash-attn` or run `uv pip install -e .` — either voids the
SDPA patch (the dispatcher routes back to flash-attn when importable).

## 3. RAM safety model (read before any launch)

A host-side watchdog (outside this container's authority) SIGKILLs at cgroup
`memory.current` ≈ 26–28 GiB; `memory.current` counts **anonymous + page cache**.
Rules proven on this box:

- Keep peak current ≤ ~21 GiB. The committed loaders give: T5 phase flat
  **15.5 GiB**, whole-load peak **20.78 GiB**, end-of-load **18.66 GiB**.
- Steady-state during a run: `free` used ≈ 16.4 GiB, current ≈ 19–21 GiB.
- Every launch carries **PID-targeted guards** (never `pkill -f` patterns — they
  match the watchdog's own shell): kill at current > 22 GiB or available < 5 GiB.
- `torch.load(mmap=True)` pins the whole file until the state dict dies — mapped
  pages cannot be fadvise-dropped. That is why T5 streams via `pread`.
- `safetensors.load_file` tensors are mmap-backed: `del` the dict **before**
  dropping the file's cache.
- `memory.peak` reset does not work on this kernel; sample `memory.current`.

## 4. Launch (from the repo root)

```sh
# gate: VRAM must be ≥ 10.0 GiB free for window 18 (9.6→16, else 14)
nvidia-smi --query-gpu=memory.free --format=csv,noheader

# loggers (timestamps matter for phase attribution)
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader -l 2 \
  | while read -r l; do echo "$(date +%T) $l"; done > output/gpu.log 2>&1 &
( while :; do echo "$(date +%T) $(free -m | awk '/^Mem:/ {print $3}')"; sleep 2; done ) > output/ram.log 2>&1 &

# guard (PID-targeted; brackets stop self-match)
( while :; do P=$(pgrep -f 'python generate[.]py' | head -1); [ -n "$P" ] && break; sleep 1; done
  while [ -e /proc/$P ]; do
    CUR=$(awk '{print $1}' /sys/fs/cgroup/memory.current)
    [ "$CUR" -gt 23622320128 ] && { kill -9 "$P"; echo "GUARD_22G killed at ${CUR}"; break; }
    AV=$(free -m | awk '/^Mem:/ {print $7}')
    [ "$AV" -lt 5000 ] && { kill -9 "$P"; echo "GUARD_5G killed: ${AV} MiB avail"; break; }
    sleep 2
  done ) &

{ time PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
  .venv/bin/python generate.py --task i2v-1.3B --infer_mode causal_fast --size '480*832' \
    --ckpt_dir ./lingbot-world-v2-1.3b-causal-fast \
    --assets_dir ./lingbot-world-v2-14b-causal-fast \
    --image examples/03/image.jpg --action_path examples/03 \
    --t5_cpu --offload_model true \
    --frame_num 269 --local_attn_size 18 --sink_size 6 --base_seed 42 \
    --save_dir output --save_file output/full_1p3b_269f_la18.mp4 \
    --prompt "<prompt>"; echo "EXIT_RC=$?"
} 2>&1 | tr '\r' '\n' \
  | .venv/bin/python -u -c 'import sys,time;[print(time.strftime("[%H:%M:%S]"),l,end="")for l in sys.stdin]' \
  > output/run.log
```

Notes: the output pipe **buffers until process exit** — an empty `run.log`
mid-run is normal; use the asctime stamps inside the lines for phase timing.
`--frame_num 81` for the 77-frame smoke. On CUDA OOM step the window down
18→16→14 (sink 4 below 14) and rename the output file to match.

## 5. Measured profile (2026-09-26, window 18/6, seed 42)

| | smoke 81f (77 out) | full 269f |
|---|---|---|
| wall | 6m11s | 9m41s |
| load (T5 build 104 s / stream 26 s / VAE 1 s / DiT 26 s) | 2m37s | 2m36s |
| T5 CPU encode + prep | ~2m47s | 4m35s |
| sampling | 38 s (5 chunks, 6.3→8.9 s) | 2m24s (17 chunks, 6→9 s **flat**) |
| decode + save | ~3 s | ~3 s |
| TTFC (cold launch → first chunk) | — | ≈ 7m20s |
| per-chunk median (chunks 2+) | ~7.6 s | 9 s |
| sampling RTF (video 16.8 s) | ~8.6× | **8.6×** |
| VRAM peak | 15,506 MiB | 15,946 MiB (of 15,949 free) |
| RAM `used` peak | 17.3 GiB | 17.4 GiB |
| cgroup current peak | ≈ 20.8 GiB | ≈ 21.0 GiB (guard at 22 never fired) |
| PSNR frame 0 vs input | 26.6 dB | 26.5 dB |
| motion mean / min | 10.6 / 1.35 | 8.2 / 0.89 |
| frame-std min | 50.0 | 50.7 |

## 6. Quality verdict (3-agent review, 2026-09-26)

- **First-frame fidelity: pass** — faithful to the input photo; frame 0
  reproducible across runs (smoke vs full PSNR 31.6 dB).
- **Artifacts / drift: concerns** — quality declines with frame index
  (7.5→4/10); scene identity holds to ~frame 180, then classic autoregressive
  drift: the hero tree dissolves by F268, luma −39 %, saturation +48 %. Model
  characteristic of the 18-frame window at 17 chunks, not a run defect.
- Objective stats pass: no frozen stretches, no dead frames, no watermarks.

## 7. Failure playbook (condensed)

| Symptom | Response |
|---|---|
| SIGKILL at startup, current ≈ 26–28 GiB | Host watchdog. Do not relaunch unchanged; verify the `.safetensors` sibling exists and `2b17bfc` is applied; check guards are PID-targeted. |
| Guard kills at 22 GiB | Read which phase climbed (ram log `used` vs current gap = cache). Weight-file cache can be dropped externally mid-run (`posix_fadvise DONTNEED` on the safetensors/pth files). |
| `assert FLASH_ATTN_2_AVAILABLE` | SDPA patch not applied or flash-attn installed; check `git diff` + dispatcher probe. |
| CUDA OOM in sampling/decode | Window ladder down (18→16→14→12+sink4); decode is internally chunked, so VRAM scales with the window, not frames. |
| No mp4 | Look in repo root (save_file join bug), then imageio-ffmpeg; `save_video` swallows exceptions. |
