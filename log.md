# Change Log

## 2026-07-14 — Cancel button (v0.2)

A running batch could only be stopped by Forge's global interrupt (or not at all).
There's now a **⏹️ Cancel** next to Run: it aborts the image currently being sampled
and stops the batch, keeping whatever already landed in the gallery.

- `shared.state.interrupted` can't carry the request on its own — `state.begin()` at the
  top of every image clears it, so a cancel landing *between* two images would be wiped.
  A module-level `_cancel_requested` survives that and is reset only when a batch starts;
  the button also calls `shared.state.interrupt()` to abort the in-flight sampling.
- The click is bound with `queue=False`, or it would queue behind the very batch it is
  trying to stop.
- Same mechanism as the sibling batch-adetailer extension.

## 2026-07-12 — Project Initiation
- Created `log.md` (this file) and `context.md`
- Gathered initial requirements from user: batch hires-fix extension for Forge Neo via drag-and-drop of txt2img images

---

## 2026-07-12 — Extension Implementation (v0.1)

### Files Created
- `scripts/batch_hires_fix.py` — main extension script:
  - `_get_hires_fix_defaults()` — default hires-fix params matching Forge Neo UI
  - `_process_single_image(img, params)` — processes one image through the Forge pipeline (same path as ✨ button)
  - `batch_hires_fix_process(files, ...)` — orchestrates sequential batch processing with error handling
  - `_build_ui_tab()` — Gradio UI tab with file input, hires-fix controls, output gallery, status log
- `install.py` — no extra deps needed (empty sentinel)
- `README.md` — installation & usage instructions
- `LICENSE` — MIT license
- `context.md` — updated with Forge Neo architecture details

### Architecture Decisions
- **Reuses Forge's existing hires-fix pipeline** via `StableDiffusionProcessingTxt2Img` + `firstpass_image`
  pattern, identical to what the ✨ button does internally. No reimplementation needed.
- **Sequential processing** (one image at a time) — safer for VRAM; batch size cap configurable
- **Infotext extraction** per-image — each source image's seed, prompt, CFG etc. are parsed from its PNG info
- **Gradio File input with `file_count="multiple"` and `type="filepath"`** — native drag-and-drop support
- **Error handling**: skip-failed-images config option; abort-on-first-error alternative

### Issues Fixed During Development
- Removed dead `_create_hires_processing()` placeholder function
- Replaced non-existent shared.opts references (`sd_hires_denoising_strength`, `hires_fix_scale_by`, etc.)
  with hardcoded defaults that match Forge Neo's UI.py values (0.6 denoise, 2.0 scale)
- Fixed upscaler choices construction to match ui.py pattern: `latent_upscale_modes + sd_upscalers`

## 2026-07-12 — Bug Fixes & Feature Additions (v0.2)

### Bugs Fixed
- **Double registration**: Forge Neo imports extension scripts from both `scripts/` and as modules, causing duplicate "Batch Hires-Fix" entries in the UI ordering tab. Added `_initialized` guard to prevent double-fire of callbacks.
- **Settings not available at runtime**: Switched from `on_ui_settings()` dict return pattern to `shared.opts.add_option()` (same pattern used by agent-scheduler-neo). This ensures options are registered immediately at import time, before any processing occurs. Fixed `AttributeError: 'Options' object has no attribute 'batch_hires_fix_max_images'`.

### Feature Added — Full Hires-Fix UI in Own Tab
- Rebuilt the tab's settings panel as a near carbon-copy of Forge Neo's existing "Hires. fix" section:
  - Denoising strength slider (0–1, step 0.05)
  - Upscaler dropdown (latent + image upscalers)
  - Hires steps slider
  - Upscale by / Resize width to / Resize height to sliders (uses `res_step` from shared opts)
  - **Hires Distilled CFG Scale** and **Hires CFG Scale** sliders (1–24, step 0.5)
  - **Hires Checkpoint** dropdown with refresh button
  - **Hires VAE / Text Encoder** multiselect dropdown
  - **Hires sampling method** dropdown (`sd_samplers.visible_sampler_names()`)
  - **Hires schedule type** dropdown (`sd_schedulers.schedulers`)
  - **Hires prompt** and **Hires negative prompt** textboxes
- All parameters are passed through to `StableDiffusionProcessingTxt2Img` with the same logic as the ✨ button (e.g., "Use same sampler" → None, etc.)

## 2026-07-12 — Root-Cause Fixes from Forge Neo Source Trace (v0.3)

### Bugs Fixed
- **Double tab (issue #2)**: Not an import-guard problem at all — the extension was
  installed TWICE (`extensions/batch-hires-fix` + stale `extensions/hires-fix-extension`
  from an earlier install). Deleted the stale folder. No code change needed.
- **`IndexError: tuple index out of range` (issue #1)**: The `[None] * 100` placeholder
  `script_args` fed garbage to every installed alwayson script (adetailer, dynamic
  prompts, loractl, agent-scheduler, ...). Replaced with the same technique Forge Neo's
  own API uses (`modules/api/api.py :: init_default_script_args`): build a list of the
  exact required length, `0` at position 0 (no selectable script), and fill each script's
  `args_from:args_to` slice with that script's real UI default values (`script.ui()`
  inside a `gr.Blocks()` context). Removed the `scripts_setup_complete = True` hack —
  with valid args, `setup_scripts()` can run normally like the real pipeline.
- **Empty prompt during hires pass**: `parse_generation_parameters` was imported but
  never used — every image was hires-fixed with `prompt=""`. Now each image's infotext
  is read via `images.read_info_from_image()` and prompt / negative prompt / seed /
  steps / sampler / scheduler / CFG / distilled CFG are applied to the processing
  object, matching what the ✨ button gets from the live UI state.
- **Not running on the main thread**: the ✨ button routes processing through
  `main_thread.run_and_wait_result()` (Forge Neo executes all major GPU work on one
  main thread — see `modules_forge/main_thread.py`). We now do the same per image.
- **Swallowed tracebacks (issue #3)**: `_process_single_image` never raises — it
  returns `(images, infotexts, error_traceback_or_None)`, prints the full traceback to
  the server console, and surfaces it verbatim in the status log. Gradio can no longer
  eat the details.

### Other Changes
- Images are converted to RGB after infotext is read (RGBA input would break the
  `firstpass_image` tensor path; reading info first because `convert()` can drop PNG info).
- Warn (but continue) when a dropped image has no embedded generation info.
- Output dir fallback order now matches `txt2img_create_processing`
  (`outdir_samples or outdir_txt2img_samples`), after the extension's custom dir.
- Also calls `scripts_txt2img.run(p, *p.script_args)` before `process_images(p)` like
  the ✨ button does (returns None with script index 0, then falls through).

## 2026-07-12 — Fix Missing Tab (v0.3.1)
- **Tab not appearing after restart**: the `_initialize()` guard checked
  `hasattr(shared.opts, "batch_hires_fix_output_dir")` — but once the settings were
  saved into Forge Neo's `config.json`, `Options.__getattr__` serves them from saved
  data at startup, so the guard fired on a fresh launch and skipped ALL registration
  (settings labels AND the tab). Removed the guard: with the stale duplicate folder
  deleted, Forge loads the script exactly once per launch, so registration now runs
  unconditionally.

## 2026-07-12 — Preview Arrow Keys + Original Filename Saving (v0.4)

### Features Added
- **←/→ arrow-key navigation in the full-size preview**: gave the results gallery
  `elem_id="batch_hires_fix_gallery"` and `preview=True`. Forge's built-in lightbox
  (javascript/imageviewer.js) attaches to any gallery, but its arrow navigation
  (`all_gallery_buttons()` in ui.js) only finds galleries whose elem id ends in
  `_gallery` inside the active tab — no custom JS needed once the id matches.
- **Save as original filename + suffix**: new checkbox (default on) and suffix textbox
  (default `-hires`) in the tab. When enabled, `p.do_not_save_samples = True` suppresses
  the pipeline's own save (dated subfolders + `[number]-[seed]-...` naming pattern), and
  results are saved manually via `images.save_image(..., forced_filename=..., save_to_dirs=False)`
  as `<original name><suffix>.<samples_format>` flat in the output directory. Name
  collisions get a `-1`, `-2`, ... counter; extra result images (e.g. from adetailer)
  get an index suffix. Infotext is still embedded in the saved file.

## 2026-07-12 — Fix Silent 0-Image Results After UI Reload (v0.4.1)
- **"No output" for every image, `Total progress: 0it`**: Reload UI calls
  `state.request_restart()` → `state.interrupt()`, which sets `state.interrupted = True`.
  Only `state.begin()` resets it, and only the UI's `wrap_gradio_gpu_call` wrapper calls
  that — our handler didn't. `process_images_inner` then hit
  `if state.interrupted: break` before loading the model or sampling, returning an empty
  Processed with no error. Now each image is wrapped in
  `shared.state.begin(job="batch_hires_fix")` / `state.end()`.
- Bonus from the same change: the Interrupt button now stops the batch (flag checked
  after each image, before the next `begin()` resets it), progress bar totals are
  correct per image, and partial results stream into the gallery as each image finishes
  (per-image `yield`).

## 2026-07-12 — Fix Low-Quality Results vs Manual ✨ (v0.4.2)
- **Quality much worse than manual hires-fix**: extension hardcoded `hr_cfg=1.0`, but
  the txt2img UI's "Hires CFG Scale" defaults to 6.0 (ui.py:265) and the ✨ button sends
  the slider value. Worse, `hr_cfg == 1` makes the hires pass drop the negative prompt
  entirely (processing.py:1603, "Negative Prompts are Ignored when CFG = 1.0") and
  sample nearly unguided. Added "Hires CFG Scale" (default 6.0) and "Hires Distilled
  CFG Scale" (default 3.0) sliders matching the txt2img UI defaults.
- **Blank upscaler ambiguity**: blank dropdown silently meant Latent upscaling
  (`shared.latent_upscale_default_mode`). Dropdown now defaults to "Latent" explicitly
  so users can see and change what they're actually getting.

## 2026-07-12 — Match Optimal Manual Settings, Inherit Shift (v0.4.3)
- Diffed embedded infotext of a manual hires-fix result vs an extension result:
  pipeline params (seed/prompt/sampler) matched; remaining gap was user-side settings
  (Latent vs 4xUltrasharp upscaler, 15 vs 30 hires steps, hires shift 4.5 vs 3).
- **Removed the Hires Distilled CFG Scale slider** (user request). `hr_distilled_cfg`
  now inherits each image's own base "Distilled CFG Scale" (Shift) from infotext —
  matters for shift-based models (Qwen/Flux family); falls back to 3.0.
- **Tab defaults now match the user's optimal manual run**: denoise 0.3,
  upscale 1.25 (step 0.05), upscaler 4xUltrasharp_4xUltrasharpV10, Hires CFG 4.5,
  hires steps 0 = auto (same step count as the source image).

## 2026-07-22 — Test-Folder Mode: Hires-Fix In Place (v0.5.0)
- **New 📁 Test Folders panel**: scans configurable roots (Settings → "Test-folder
  scan roots", default Commissions + Requests) for `<set>/Tests` folders holding
  base images with no `-hires`/`-adetailer`/`-edited` variant yet, and lists them
  as checkboxes ("Commission 137 - M, Fluorite  (7 to do)").
- **Hires-Fix Selected Folders**: all pending bases across the ticked sets run as
  one sequential batch; each result saves back into the folder its source came
  from (`Tests/1r1.png → Tests/1r1-hires.png`), which is the layout the content
  manager's variant chain keys on. Original-name saving is forced on in this mode.
- Re-running is idempotent (bases with any existing variant are skipped) and the
  list rescans itself after a run. Drag-and-drop mode is unchanged.
- `test_scan.py`: standalone self-check for the scan logic (Forge modules stubbed).
- Review findings fixed before release:
  - **Cancel mid-image no longer saves the partial image**: an interrupted sampling
    loop returns the partially-denoised result (sd_samplers_common launch_sampling
    catches InterruptedException and returns last_latent); saving it as
    `<stem>-hires.png` would look finished — and in folder mode permanently hide the
    base from the pending scan. Both modes now drop the result when
    `state.interrupted/stopping_generation` is set.
  - **Folder mode forces the `-hires` suffix** (suffix textbox ignored there): the
    pending scan keys on the `-hires/-adetailer/-edited` tokens, so a custom suffix
    would reprocess every base on every run, multiplying files.
  - **Folder mode forces png output** regardless of the global samples_format —
    the content manager chain expects `<stem>-hires.png`, and some formats (heif)
    drop the embedded infotext. Scanner also recognizes jxl/avif/heif files now.

<!-- Future entries will be appended here -->

