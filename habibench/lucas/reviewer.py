"""Conservative, survey-label-conditioned context review of UK LUCAS photographs."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import urllib.error
import urllib.request
from importlib.metadata import version
from pathlib import Path

from PIL import Image, ImageOps


NOVEL_MODEL = "Qwen/Qwen3.8-27B"
NOVEL_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
LONG_EDGE = 1024
MAX_TOKENS = 180
RUBRIC_VERSION = "lucas-2018-context-anomaly-v2"
COVER_NAMES = {
    "A": "Artificial land", "B": "Cropland", "C": "Woodland", "D": "Shrubland",
    "E": "Grassland", "F": "Bareland", "G": "Water", "H": "Wetlands",
}
CLASSES = tuple(COVER_NAMES)
CONTEXTS = ("compatible", "ambiguous", "extreme_outlier")
OBSTRUCTIONS = ("none", "building", "vehicle", "interior", "person_equipment", "other")
STRENGTHS = ("high", "medium", "low")

# Eurostat LUCAS 2018 C1 (photograph protocol) and C3 (land cover categories).
RUBRIC = """You are screening UK LUCAS field photographs for extreme context failures.
The supplied broad survey label describes the survey POINT. Judge whether this individual
PHOTO is plainly unusable as a view of that landscape. Preserve difficult ecological examples.
You need not identify the habitat in the photo. A plausible alternative class, inability
to confirm a label, and uncertainty about management or hydrology all favour retention.

A compatible view can include nearby or secondary features. A wetland may look grassy,
heathy or dry and have no standing water; a field of grasses may be grassland or cropland.
Woodland, shrubland and grassland intergrade. Bare soil can occur in managed fields.
Natural close-ups can be relevant evidence. A fence, survey marker, distant house or vehicle
within a visible landscape is ordinary. For Artificial land, a facade or paved scene can
itself show the labelled land cover.

Choose extreme_outlier only for compelling visible evidence of a wrong-context capture.
For example, a building facade or parked vehicle fills virtually the whole usable frame
at a grassland point. A strip of sky or background treetops above a van does not make
this a usable grassland view. A vehicle beside a clearly visible field is a different case.
An interior or camera blocked by equipment may similarly eliminate the landscape.
Use ambiguous for ecological boundary cases and inconclusive views. Use compatible when
the scene or detail is consistent with the survey context.

Report context, the obstruction that dominates the frame (none when absent), landscape_visible
(true when enough survey-relevant or ecologically plausible context is visible), strength
of the visual evidence for your context verdict, and a short reason describing what is visible.
Do not infer what lies behind a blocked view. Use the supplied JSON schema."""

VIEW_TEXT = {
    "P": "Point photograph. A marker may show the survey location.",
    "C": "Close-up photograph. Detail is intentional; ecological detail can be useful by itself.",
    **{v: "Compass photograph. Its surroundings may legitimately differ from the survey point."
       for v in "NESW"},
}

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "context": {"type": "string", "enum": list(CONTEXTS)},
        "obstruction": {"type": "string", "enum": list(OBSTRUCTIONS)},
        "landscape_visible": {"type": "boolean"},
        "strength": {"type": "string", "enum": list(STRENGTHS)},
        "reason": {"type": "string"},
    },
    "required": ["context", "obstruction", "landscape_visible", "strength", "reason"],
}


def prompt_preview(view: str, broad_class: str) -> dict:
    if broad_class not in COVER_NAMES or view not in VIEW_TEXT:
        raise ValueError(f"Unknown broad class or view: {broad_class!r}, {view!r}")
    return {
        "system": RUBRIC,
        "user": (f"Survey point label: {broad_class} — {COVER_NAMES[broad_class]}. "
                 f"{VIEW_TEXT[view]} Assess this photo for an extreme context failure."),
        "schema": SCHEMA,
    }


def validate_observation(obs: dict) -> dict:
    if not isinstance(obs, dict) or set(obs) != set(SCHEMA["required"]):
        raise ValueError("Observation must have exactly the five context review fields")
    if obs["context"] not in CONTEXTS or obs["obstruction"] not in OBSTRUCTIONS:
        raise ValueError("Invalid context assessment or obstruction")
    if type(obs["landscape_visible"]) is not bool or obs["strength"] not in STRENGTHS:
        raise ValueError("Invalid landscape visibility or evidence strength")
    if not isinstance(obs["reason"], str) or not obs["reason"].strip():
        raise ValueError("Missing visual reason")
    return obs


def decision(observation: dict, broad_class: str) -> str:
    """Exclude only a high-confidence extreme anomaly without usable landscape context."""
    if broad_class not in COVER_NAMES:
        raise ValueError(f"Invalid survey class {broad_class!r}")
    obs = validate_observation(observation)
    if (obs["context"] == "extreme_outlier" and obs["strength"] == "high"
            and not obs["landscape_visible"] and obs["obstruction"] != "none"):
        return "exclude"
    return "keep"


def _parse(raw: str) -> dict:
    try:
        return {"raw": raw, "observation": validate_observation(json.loads(raw))}
    except (TypeError, ValueError) as exc:
        return {"raw": raw, "error": f"Invalid model response: {exc}"}


def _identity(backend: str, model: str) -> dict:
    prompts = [prompt_preview(v, c) for v in VIEW_TEXT for c in CLASSES]
    digest = hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest()
    return {
        "backend": backend, "model": model, "rubric_version": RUBRIC_VERSION,
        "rubric_sha256": digest, "long_edge": LONG_EDGE, "resize": "exif-rgb-lanczos-v1",
        "pillow_version": version("Pillow"), "max_tokens": MAX_TOKENS, "temperature": 0,
        "seed": 0,
    }


def _image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((LONG_EDGE, LONG_EDGE), Image.Resampling.LANCZOS)
    image.info.clear()  # Strip source EXIF, comments and GPS from the inference payload.
    return image


class OllamaBackend:
    url = "http://127.0.0.1:11434"

    def __init__(self, model: str):
        self.model = model
        tags = self._api("/api/tags")["models"]
        entry = next((m for m in tags if m["name"] == model), None)
        if entry is None:
            raise ValueError(f"Ollama model {model!r} is not installed; available: {[m['name'] for m in tags]}")
        self.disable_thinking = ("thinking" in entry.get("capabilities", [])
                                 or model.lower().split("/")[-1].startswith("qwen3.8"))
        # The installed Qwen3-VL thinking build can consume hundreds of tokens
        # internally even when the server accepts think=false.
        self.token_budget = 2048 if model.lower().startswith("qwen3-vl") else MAX_TOKENS
        self.identity = {
            **_identity("ollama", model), "model_digest": entry["digest"],
            "runtime_version": self._api("/api/version")["version"],
            "image_transport": "png", "thinking": False if self.disable_thinking else "model-default",
            "num_ctx": 8192, "max_tokens": self.token_budget,
        }

    def _api(self, route: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.url + route, data, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode(errors="replace")
            raise RuntimeError(f"Ollama {route}: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot reach Ollama at {self.url}. Start Ollama and retry. {exc.reason}") from exc

    def review_batch(self, items) -> list[dict]:
        results = []
        for path, view, broad_class in items:
            try:
                prompt = prompt_preview(view, broad_class)
                with io.BytesIO() as buffer:
                    _image(path).save(buffer, format="PNG")
                    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                payload = {
                    "model": self.model, "stream": False, "format": SCHEMA, "keep_alive": "15m",
                    "options": {"temperature": 0, "num_predict": self.token_budget, "num_ctx": 8192, "seed": 0},
                    "messages": [
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"], "images": [encoded]},
                    ],
                }
                if self.disable_thinking:
                    payload["think"] = False
                # Some local thinking builds exhaust their budget before emitting JSON.
                # Try more tokens, then change decoding to break a repetitive generation.
                attempts = [(0, self.token_budget), (0, min(6000, self.token_budget * 3)),
                            (0.3, self.token_budget), (0.6, self.token_budget)]
                history = []
                for temperature, budget in attempts:
                    payload["options"].update(temperature=temperature, num_predict=budget)
                    result = self._api("/api/chat", payload)
                    parsed = _parse(result["message"]["content"])
                    parsed.update(temperature=temperature, num_predict=budget, seed=0,
                                  done_reason=result.get("done_reason"), eval_count=result.get("eval_count"))
                    history.append(parsed)
                    if not parsed.get("error") or parsed["done_reason"] != "length":
                        break
                if len(history) > 1:
                    parsed["prior_attempts"] = history[:-1]
                results.append(parsed)
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                results.append({"raw": "", "error": str(exc)})
        return results


class VllmBackend:
    """Single-GPU, offline inference. Slurm launches one instance per GPU."""

    def __init__(self, model: str, model_path):
        if model != NOVEL_MODEL or not model_path:
            raise ValueError(f"Novel requires {NOVEL_MODEL} and its local snapshot path")
        snapshot = Path(model_path).resolve(strict=True)
        marker = snapshot / "HABIBENCH_REVISION"
        manifest = snapshot / "HABIBENCH_FILES.json"
        if not marker.is_file() or marker.read_text().strip() != NOVEL_REVISION or not manifest.is_file():
            raise ValueError("Stage the pinned weights with fetch-model before starting a GPU worker.")
        files = json.loads(manifest.read_text())
        from .download import REQUIRED
        if not REQUIRED <= files.keys() or not any(name.endswith(".safetensors") for name in files):
            raise ValueError("Incomplete model snapshot manifest")
        for name, info in files.items():
            if Path(name).name != name or (snapshot / name).stat().st_size != info["size"]:
                raise ValueError(f"Invalid staged file: {name}")

        os.environ["HF_HUB_OFFLINE"] = "1"
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        self.identity = {
            **_identity("vllm", model), "model_revision": NOVEL_REVISION,
            "model_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "runtime_version": version("vllm"), "transformers_version": version("transformers"),
            "dtype": "bfloat16", "thinking": False, "max_model_len": 8192,
            "max_num_seqs": 32, "gpu_memory_utilization": 0.90, "tensor_parallel_size": 1,
            "image_transport": "rgb-pixels",
            "oci_digest": os.environ.get("HABIBENCH_OCI_DIGEST"),
            "squashfs_sha256": os.environ.get("HABIBENCH_SQFS_SHA256"),
        }
        self.engine = LLM(
            model=str(snapshot), dtype="bfloat16", tensor_parallel_size=1,
            seed=0, max_model_len=8192, max_num_seqs=32, gpu_memory_utilization=0.90,
            limit_mm_per_prompt={"image": 1, "video": 0}, generation_config="vllm",
        )
        processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        self.prompts = {}
        for broad_class in CLASSES:
            for view in VIEW_TEXT:
                prompt = prompt_preview(view, broad_class)
                messages = [
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt["user"]},
                    ]},
                ]
                self.prompts[view, broad_class] = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
        self.params = SamplingParams(
            temperature=0, seed=0, max_tokens=MAX_TOKENS,
            structured_outputs=StructuredOutputsParams(json=SCHEMA),
        )

    def _generate(self, requests: list) -> list[dict]:
        try:
            outputs = self.engine.generate(requests, self.params, use_tqdm=False)
            if len(outputs) != len(requests):
                raise RuntimeError("vLLM returned the wrong number of outputs")
            return [_parse(output.outputs[0].text) for output in outputs]
        except (ValueError, RuntimeError) as exc:
            if "EngineDead" in type(exc).__name__:
                raise  # Allow Slurm to restart the worker from its durable ledger.
            if len(requests) == 1:
                return [{"raw": "", "error": str(exc)}]
            midpoint = len(requests) // 2
            return self._generate(requests[:midpoint]) + self._generate(requests[midpoint:])

    def review_batch(self, items) -> list[dict]:
        results = [{} for _ in items]
        indices, requests = [], []
        for i, (path, view, broad_class) in enumerate(items):
            try:
                requests.append({"prompt": self.prompts[view, broad_class], "multi_modal_data": {"image": _image(path)}})
                indices.append(i)
            except (OSError, ValueError, KeyError) as exc:
                results[i] = {"raw": "", "error": str(exc)}
        if requests:
            for index, result in zip(indices, self._generate(requests), strict=True):
                results[index] = result
        return results


def make_backend(profile: str, model: str, model_path=None):
    if profile == "local-small":
        return OllamaBackend(model)
    if profile == "novel":
        return VllmBackend(model, model_path)
    raise ValueError(f"Unknown profile {profile!r}")
