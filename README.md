# Comfy SynthID Remover

Local image-to-image reconstruction with [ComfyUI](https://github.com/Comfy-Org/ComfyUI). Includes ready-to-import workflows and a Python CLI for processing individual images or folders.

The pipeline runs an input image through a diffusion model and saves the reconstructed result. This can affect invisible watermarks, including SynthID, but **removal is not guaranteed**. The project does not extract watermarks or implement a SynthID detector. Image details can change during reconstruction.

[Документация на русском](README.ru.md)

## Quick start

1. Install and start [ComfyUI](https://github.com/Comfy-Org/ComfyUI).
2. Install a compatible diffusion model, text encoder, and VAE (see [Models](#models)).
3. Drag [`workflows/optimized.json`](workflows/optimized.json) onto the ComfyUI canvas.
4. Select the installed weights in the loader nodes and choose an input file in **Load Image**.
5. Click **Run**. Images are saved under `ComfyUI/output/` with the prefix `SynthidImg2Img`.

The optimized workflow uses standard ComfyUI nodes. It does not require an extension for image saving.

## Models

The workflow is configured for a Z-Image-style setup with a Qwen3-4B text encoder. Its default model filenames are:

| ComfyUI directory | Default filename |
| --- | --- |
| `models/diffusion_models/` | `moody-pro-signature-004_00001__clean_convrot_int8.safetensors` |
| `models/text_encoders/` | `qwen3-4b_int8_convrot_fp16emixed.safetensors` |
| `models/vae/` | `ae.safetensors` |

These filenames are presets, not bundled downloads. Replace them with compatible files available in your ComfyUI installation. In particular, the INT8 checkpoints may need support for their quantization format. The [Comfy-Org Z-Image Turbo collection](https://huggingface.co/Comfy-Org/z_image_turbo) is a source of standard weights; using other checkpoints may require adjusting the model and encoder settings.

The **CLIP Loader** is set to `lumina2`, and **ModelSamplingAuraFlow** uses `shift = 3` with `flow` sampling. Keep the loader configuration compatible with the chosen model.

## Workflows

| File | Purpose |
| --- | --- |
| [`workflows/optimized.json`](workflows/optimized.json) | Recommended starting point: six-step img2img using built-in nodes. |
| [`workflows/fast-4step-experimental.json`](workflows/fast-4step-experimental.json) | Four-step preset for testing lower latency. |
| [`workflows/original.json`](workflows/original.json) | Full reference graph, including the optional `SaveImageAdvanced` node. |
| [`api/prompt.json`](api/prompt.json) | API-format version used by the CLI. |

Default reconstruction settings: `steps = 6`, `cfg = 1`, `sampler = res_multistep`, `scheduler = karras`, `denoise = 0.6`, `shift = 3`, empty prompt.

The optimized workflow shares a single empty-prompt conditioning output between the sampler's positive and negative inputs. With `cfg = 1`, the negative branch does not contribute classifier-free guidance, so a second identical text-encoding node is unnecessary. The text encoder still needs to load for the first uncached execution.

## Command-line use

Requires Python 3.9+ and a running ComfyUI server. The CLI uses the Python standard library; no `pip install` is required.

Process one image:

```bash
python cli.py -i input.webp -o output.png
```

Process a folder (non-recursive):

```bash
python cli.py -i ./input -o ./output --steps 4 --denoise 0.6
```

Select different model filenames:

```bash
python cli.py -i input.webp -o output.png \
  --model z_image_turbo_bf16.safetensors \
  --clip qwen_3_4b.safetensors \
  --vae ae.safetensors \
  --scheduler simple --seed 42
```

Other useful options: `--server` (default `http://127.0.0.1:8188`), `--prompt`, `--shift`, `--timeout`, `--poll`, and `--api-prompt`. See `python cli.py --help`.

The runner uploads images through `/upload/image`, queues `api/prompt.json` using `/prompt`, waits for `/history/{prompt_id}`, and downloads the output using `/view`. Folder inputs run sequentially so ComfyUI can reuse loaded models. Run the server locally or restrict API access; the CLI does not provide authentication.

## Tuning

Lowering `steps` from 6 to 4 reduces the number of diffusion passes, but it may also change fine details or the result of a watermark check. `denoise` controls how far reconstruction can deviate from the input. Use a fixed `--seed` when comparing settings.

For a meaningful comparison, use the same image and evaluate both visual fidelity and detector results. A negative SynthID detection result **does not prove** that the watermark has been completely removed. No removal-rate benchmark is included.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover workflow links, API prompt structure, CLI overrides, and a mocked ComfyUI HTTP round trip. They do not run model inference or validate SynthID removal.

## License

No software license is included. The project can be viewed and forked on GitHub, but reuse and redistribution permissions have not been specified.
