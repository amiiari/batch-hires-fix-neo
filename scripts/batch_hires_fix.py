"""
Batch Hires-Fix Extension for Forge Neo
=======================================
Provides a UI tab where users can drag-and-drop multiple txt2img-generated images
and run them through hires-fix in batch mode, reusing Forge's existing pipeline.
"""
import os
import traceback
from contextlib import closing

import gradio as gr
from PIL import Image

from modules import images, processing, script_callbacks, scripts, shared
from modules.infotext_utils import parse_generation_parameters
from modules_forge import main_thread

# ──────────────────────────────────────────────
# Extension Settings (registered via add_option)
# ──────────────────────────────────────────────
def _register_settings():
    section = ("batch_hires_fix", "Batch Hires-Fix")

    shared.opts.add_option(
        "batch_hires_fix_output_dir",
        shared.OptionInfo("", "Output Directory", gr.Textbox, {}, section=section)
        .info("Leave empty to use the default txt2img output directory."),
    )

    shared.opts.add_option(
        "batch_hires_fix_max_images",
        shared.OptionInfo(50, "Max Images per Batch", gr.Slider,
                         {"minimum": 1, "maximum": 500, "step": 1}, section=section)
        .info("Maximum number of images that can be processed in one batch."),
    )

    shared.opts.add_option(
        "batch_hires_fix_skip_errors",
        shared.OptionInfo(True, "Skip Failed Images and Continue", gr.Checkbox,
                          {}, section=section)
        .info("If an image fails during hires-fix, skip it and continue with the rest."),
    )

# ──────────────────────────────────────────────
# Default script args — mirrors modules/api/api.py :: init_default_script_args
# ──────────────────────────────────────────────
_default_script_args_cache: list | None = None

def _get_default_script_args():
    """
    Build a script_args list of the exact length the txt2img ScriptRunner
    expects, with position 0 = 0 (no selectable script) and every alwayson
    script's slice filled with that script's own UI default values.

    This is the same technique Forge Neo's API uses (init_default_script_args)
    when there is no live UI to source args from. Passing placeholder Nones
    instead breaks alwayson scripts (adetailer, dynamic prompts, ...) that
    index into their args expecting real values.
    """
    global _default_script_args_cache

    runner = scripts.scripts_txt2img

    last_arg_index = 1
    for script in runner.scripts:
        if last_arg_index < script.args_to:
            last_arg_index = script.args_to

    if _default_script_args_cache is not None and len(_default_script_args_cache) == last_arg_index:
        return _default_script_args_cache

    script_args = [None] * last_arg_index
    script_args[0] = 0

    with gr.Blocks():  # script.ui() creates gradio components; needs a Blocks context
        for script in runner.scripts:
            ui_elems = script.ui(script.is_img2img)
            if ui_elems:
                script_args[script.args_from : script.args_to] = [elem.value for elem in ui_elems]

    _default_script_args_cache = script_args
    return script_args

# ──────────────────────────────────────────────
# Infotext extraction
# ──────────────────────────────────────────────
def _apply_source_image_parameters(p, geninfo: str):
    """
    Apply the source image's generation parameters (prompt, seed, sampler, ...)
    to the processing object, so the hires pass is conditioned the same way the
    original generation was. The ✨ button gets these for free from the live
    txt2img UI state; we must recover them from the image's infotext.
    """
    params = parse_generation_parameters(geninfo, [])

    p.prompt = params.get("Prompt", "")
    p.negative_prompt = params.get("Negative prompt", "")
    p.seed = params.get("Seed", -1)
    p.subseed = params.get("Variation seed", -1)

    try:
        p.steps = int(params["Steps"])
    except (KeyError, ValueError):
        pass
    try:
        p.cfg_scale = float(params["CFG scale"])
    except (KeyError, ValueError):
        pass
    try:
        p.distilled_cfg_scale = float(params["Distilled CFG Scale"])
        # Shift-based models (Qwen, Flux, ...): use the same shift for the
        # hires pass as the original generation, like "Use same" semantics.
        p.hr_distilled_cfg = p.distilled_cfg_scale
    except (KeyError, ValueError):
        pass

    if params.get("Sampler"):
        p.sampler_name = params["Sampler"]
    if params.get("Schedule type"):
        p.scheduler = params["Schedule type"]

# ──────────────────────────────────────────────
# Saving with the original filename + suffix
# ──────────────────────────────────────────────
def _save_with_original_name(processed, p, stem: str, suffix: str):
    """
    Save result images as <original filename><suffix>.<ext> directly in the
    output directory (no dated subfolders, no [seed]-[prompt] naming pattern).
    Collisions get a -1, -2, ... counter instead of overwriting.
    """
    outdir = p.outpath_samples
    os.makedirs(outdir, exist_ok=True)
    extension = shared.opts.samples_format

    for i, image in enumerate(processed.images):
        base = f"{stem}{suffix}" if i == 0 else f"{stem}{suffix}-{i}"
        name = base
        n = 1
        while os.path.exists(os.path.join(outdir, f"{name}.{extension}")):
            name = f"{base}-{n}"
            n += 1

        infotext = processed.infotexts[i] if i < len(processed.infotexts) else None
        images.save_image(
            image, outdir, "",
            info=infotext,
            forced_filename=name,
            extension=extension,
            save_to_dirs=False,
            p=p,
        )

# ──────────────────────────────────────────────
# Core Processing Logic — mirrors txt2img_upscale_function
# ──────────────────────────────────────────────
def _process_single_image(img: Image.Image, geninfo: str | None, hires_params: dict, save_opts: dict):
    """
    Process one image through hires-fix. Mirrors the logic in
    modules/txt2img.py :: txt2img_upscale_function().

    Key difference from the ✨ button: we do NOT set p.txt2img_upscale = True,
    because that flag causes Forge Neo to skip model loading (it assumes the ✨
    button already has a loaded model). Since our extension runs independently,
    we need normal model loading to occur inside process_images().

    Runs on Forge's main thread (see batch_hires_fix_process). Never raises:
    returns (images, infotexts, error_traceback_or_None) so the full traceback
    reaches the status log instead of being swallowed by Gradio.
    """
    try:
        p = processing.StableDiffusionProcessingTxt2Img(
            outpath_samples=(
                getattr(shared.opts, "batch_hires_fix_output_dir", None)
                or shared.opts.outdir_samples
                or shared.opts.outdir_txt2img_samples
            ),
            outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids,
            prompt="",
            styles=[],
            negative_prompt="",
            batch_size=1,
            n_iter=1,
            cfg_scale=float(hires_params.get("cfg_scale", 7.0)),
            distilled_cfg_scale=float(hires_params.get("hr_distilled_cfg", 3.0)),
            width=img.size[0],
            height=img.size[1],
            enable_hr=True,
            denoising_strength=float(hires_params.get("denoising_strength", 0.6)),
            hr_scale=float(hires_params.get("hr_scale", 2.0)),
            hr_upscaler=hires_params.get("hr_upscaler"),
            hr_second_pass_steps=int(hires_params.get("hr_second_pass_steps", 0)),
            hr_resize_x=int(hires_params.get("hr_resize_x", 0)),
            hr_resize_y=int(hires_params.get("hr_resize_y", 0)),
            hr_checkpoint_name=None,
            hr_additional_modules=["Use same choices"],
            hr_sampler_name=(
                None if hires_params.get("hr_sampler_name") == "Use same sampler"
                else hires_params.get("hr_sampler_name")
            ),
            hr_scheduler=(
                None if hires_params.get("hr_scheduler") == "Use same scheduler"
                else hires_params.get("hr_scheduler")
            ),
            hr_prompt="",
            hr_negative_prompt="",
            hr_cfg=float(hires_params.get("hr_cfg", 6.0)),
            hr_distilled_cfg=float(hires_params.get("hr_distilled_cfg", 3.0)),
            override_settings={},
        )

        # Same pattern as txt2img_create_processing: assign scripts + real
        # default args and let the setters run setup_scripts() normally.
        p.scripts = scripts.scripts_txt2img
        p.script_args = _get_default_script_args().copy()

        if geninfo:
            _apply_source_image_parameters(p, geninfo)

        p.firstpass_image = img
        # Intentionally NOT setting p.txt2img_upscale — see docstring above.

        if shared.opts.txt2img_upscale_single_batch:
            p.batch_size = 1
            p.n_iter = 1

        p.override_settings["save_images_before_highres_fix"] = False

        if save_opts.get("use_original_name"):
            # We save manually afterwards with the original filename + suffix.
            p.do_not_save_samples = True

        with closing(p):
            processed = scripts.scripts_txt2img.run(p, *p.script_args)

            if processed is None:
                processed = processing.process_images(p)

        if save_opts.get("use_original_name"):
            _save_with_original_name(processed, p, save_opts["stem"], save_opts.get("suffix", ""))

        return processed.images, processed.infotexts, None
    except Exception:
        tb = traceback.format_exc()
        print(f"[Batch Hires-Fix] Error processing image:\n{tb}")
        return [], [], tb


def batch_hires_fix_process(
    files,
    denoising_strength,
    hr_scale,
    hr_upscaler,
    hr_second_pass_steps,
    hr_resize_x,
    hr_resize_y,
    hr_sampler_name,
    hr_scheduler,
    hr_cfg,
    use_original_name,
    filename_suffix,
):
    """
    Main batch processing function. Processes each image through hires-fix
    sequentially and collects all results.
    """
    if not files:
        yield [], "No images to process. Please drag and drop some images first."
        return

    max_images = shared.opts.batch_hires_fix_max_images
    skip_errors = shared.opts.batch_hires_fix_skip_errors

    if len(files) > max_images:
        yield [], f"Too many images ({len(files)}). Max is {max_images}."
        return

    hires_params = {
        "denoising_strength": float(denoising_strength),
        "hr_scale": float(hr_scale),
        "hr_upscaler": str(hr_upscaler) if hr_upscaler else None,
        "hr_second_pass_steps": int(hr_second_pass_steps),
        "hr_resize_x": int(hr_resize_x or 0),
        "hr_resize_y": int(hr_resize_y or 0),
        "hr_sampler_name": str(hr_sampler_name) if hr_sampler_name else None,
        "hr_scheduler": str(hr_scheduler) if hr_scheduler else None,
        "hr_cfg": float(hr_cfg),
    }

    total = len(files)
    all_results: list = []
    status_messages: list[str] = []
    failed_count = 0

    for idx, file_obj in enumerate(files):
        if isinstance(file_obj, str):
            image_path = file_obj
        elif hasattr(file_obj, "name"):
            image_path = file_obj.name
        else:
            status_messages.append(
                f"❌ [{idx + 1}/{total}] Unknown file object at index {idx}"
            )
            failed_count += 1
            continue

        name = os.path.basename(image_path)

        try:
            img = Image.open(image_path)
            # Read infotext BEFORE converting — convert() can drop PNG info.
            geninfo, _items = images.read_info_from_image(img)
            img = img.convert("RGB")
        except Exception as e:
            status_messages.append(f"❌ [{idx + 1}/{total}] Failed to load {name}: {e}")
            failed_count += 1
            continue

        if not geninfo:
            status_messages.append(
                f"⚠️ [{idx + 1}/{total}] {name}: no generation info found in image — "
                f"hires pass will run with an empty prompt."
            )

        shared.total_tqdm.clear()

        save_opts = {
            "use_original_name": bool(use_original_name),
            "stem": os.path.splitext(name)[0],
            "suffix": filename_suffix or "",
        }

        # state.begin() resets state.interrupted / stopping_generation, which
        # otherwise stay True forever after a UI reload (request_restart calls
        # interrupt()) and make process_images_inner return 0 images silently.
        # Real generations get this from the UI's wrap_gradio_gpu_call wrapper.
        shared.state.begin(job="batch_hires_fix")
        try:
            # GPU work must run on Forge's main thread, same as the ✨ button
            # (txt2img.py routes through main_thread.run_and_wait_result).
            result_images, _infotexts, error_tb = main_thread.run_and_wait_result(
                _process_single_image, img, geninfo, hires_params, save_opts
            )
        finally:
            shared.state.end()

        shared.total_tqdm.clear()

        if error_tb:
            status_messages.append(f"❌ [{idx + 1}/{total}] Error on {name}:\n{error_tb}")
            failed_count += 1
            if not skip_errors:
                break
            continue

        # Interrupt pressed during this image: report and stop the batch.
        # (Checked before the next begin() so the flag hasn't been reset yet.)
        if shared.state.interrupted or shared.state.stopping_generation:
            status_messages.append(f"⏹️ [{idx + 1}/{total}] Interrupted — stopping batch.")
            failed_count += 1
            break

        if not result_images:
            status_messages.append(f"⚠️ [{idx + 1}/{total}] No output for {name}")
            failed_count += 1
            continue

        all_results.extend(result_images)
        status_messages.append(f"✅ [{idx + 1}/{total}] Done: {name}")

        # Stream partial results into the gallery as each image finishes.
        yield all_results, f"Processing... {idx + 1}/{total} done.\n\n" + "\n".join(status_messages)

    status_text = (
        f"Batch complete — {len(all_results)} succeeded, "
        f"{failed_count} failed/skipped out of {total}.\n\n"
        + "\n".join(status_messages)
    )

    yield all_results, status_text

# ──────────────────────────────────────────────
# Gradio UI Tab
# ──────────────────────────────────────────────
def _build_ui_tab():
    from modules import sd_samplers, sd_schedulers

    upscaler_choices = list(shared.latent_upscale_modes.keys()) + [x.name for x in shared.sd_upscalers]

    default_upscaler = "4xUltrasharp_4xUltrasharpV10"
    if default_upscaler not in upscaler_choices:
        default_upscaler = "Latent"

    with gr.Blocks(analytics_enabled=False) as block:
        gr.Markdown(
            "# Batch Hires-Fix\n"
            "Drag and drop images generated via txt2img to run them through hires-fix in batch."
        )

        with gr.Row():
            # ── Left column: inputs & controls ──
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Drop images here (or click to browse)",
                    file_count="multiple",
                    type="filepath",
                )

                gr.Markdown("### Hires-Fix Settings")

                with gr.Row():
                    denoising_strength = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01,
                        value=0.3, label="Denoising Strength", scale=2,
                    )
                    hr_scale = gr.Slider(
                        minimum=1.0, maximum=8.0, step=0.05,
                        value=1.25, label="Upscale By", scale=2,
                    )

                hr_upscaler = gr.Dropdown(
                    choices=upscaler_choices,
                    value=default_upscaler,
                    label="Hires Upscaler",
                    allow_custom_value=True,
                )

                # hr_cfg=1.0 would make the pipeline drop the negative prompt
                # entirely, so keep this visible. The hires distilled CFG
                # (shift) is intentionally NOT exposed: it inherits each
                # image's own base shift from infotext.
                hr_cfg = gr.Slider(
                    minimum=1.0, maximum=24.0, step=0.5,
                    value=4.5, label="Hires CFG Scale",
                )

                with gr.Row():
                    hr_second_pass_steps = gr.Slider(
                        minimum=0, maximum=150, step=1,
                        value=0, label="Hires Steps (0 = same as image's steps)", scale=2,
                    )
                    hr_resize_x = gr.Number(
                        value=0, label="Resize to Width (0 = auto)",
                        min=0, precision=0, scale=2,
                    )

                hr_resize_y = gr.Number(
                    value=0, label="Resize to Height (0 = auto)",
                    min=0, precision=0,
                )

                # ── Sampler & Scheduler ──
                with gr.Row():
                    hr_sampler_name = gr.Dropdown(
                        choices=["Use same sampler"] + sd_samplers.visible_sampler_names(),
                        value="Use same sampler",
                        label="Hires sampling method",
                    )
                    hr_scheduler = gr.Dropdown(
                        choices=["Use same scheduler"] + [x.label for x in sd_schedulers.schedulers],
                        value="Use same scheduler",
                        label="Hires schedule type",
                    )

                # ── Output naming ──
                with gr.Row():
                    use_original_name = gr.Checkbox(
                        value=True,
                        label="Save as original filename + suffix",
                        scale=2,
                    )
                    filename_suffix = gr.Textbox(
                        value="-hires",
                        label="Filename suffix",
                        max_lines=1,
                        scale=1,
                    )

                process_btn = gr.Button("🚀 Run Batch Hires-Fix", variant="primary", size="lg")

            # ── Right column: output & status ──
            with gr.Column(scale=2):
                # elem_id must end in "_gallery" so Forge's lightbox modal
                # (javascript/imageviewer.js + ui.js all_gallery_buttons) picks
                # it up — that's what enables ←/→ arrow-key navigation in the
                # full-size preview. preview=True matches the txt2img gallery.
                output_gallery = gr.Gallery(
                    label="Results",
                    elem_id="batch_hires_fix_gallery",
                    columns=[4],
                    height="auto",
                    preview=True,
                )

                status_text = gr.TextArea(
                    label="Status / Log",
                    lines=10,
                    interactive=False,
                )

        process_btn.click(
            fn=batch_hires_fix_process,
            inputs=[
                file_input,
                denoising_strength,
                hr_scale,
                hr_upscaler,
                hr_second_pass_steps,
                hr_resize_x,
                hr_resize_y,
                hr_sampler_name,
                hr_scheduler,
                hr_cfg,
                use_original_name,
                filename_suffix,
            ],
            outputs=[output_gallery, status_text],
        )

    return block

def _on_ui_tabs():
    """Register the Batch Hires-Fix tab with Forge Neo's UI."""
    yield (_build_ui_tab(), "Batch Hires-Fix", "batch-hires-fix-tab")

# ──────────────────────────────────────────────
# Registration
# Runs unconditionally: Forge Neo loads each extension script once per launch.
# (The historical "double tab" came from a stale duplicate extension folder,
# not double import. Do NOT guard on shared.opts attribute existence — saved
# values in config.json make the attribute exist before registration, which
# would skip tab registration entirely.)
# ──────────────────────────────────────────────
_register_settings()
script_callbacks.on_ui_tabs(_on_ui_tabs)
