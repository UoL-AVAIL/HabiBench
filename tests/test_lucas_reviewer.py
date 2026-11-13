import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from habibench.lucas import reviewer


def observation(**changes):
    result = dict(context="extreme_outlier", obstruction="building", landscape_visible=False,
                  strength="high", reason="The house facade fills the frame.")
    result.update(changes)
    return result


class ReviewTests(unittest.TestCase):
    def test_only_extreme_high_confidence_context_failure_is_excluded(self):
        for changes, label, expected in [
            ({}, "E", "exclude"),
            ({"obstruction": "vehicle", "reason": "The parked van fills the frame."}, "E", "exclude"),
            ({"strength": "medium"}, "E", "keep"), ({"strength": "low"}, "E", "keep"),
            ({"landscape_visible": True}, "E", "keep"),
            ({"obstruction": "none", "reason": "The bank shows vegetation and sky with no open water."},
             "G", "keep"),
            ({"context": "ambiguous", "obstruction": "none", "landscape_visible": True,
              "reason": "Grassy heath could be wetland."}, "H", "keep"),
            ({"context": "compatible", "reason": "The building is artificial cover."}, "A", "keep"),
            ({"context": "ambiguous", "reason": "A close-up of leaves."}, "C", "keep"),
        ]:
            with self.subTest(changes=changes):
                self.assertEqual(reviewer.decision(observation(**changes), label), expected)

    def test_invalid_results_retain_raw_for_audit(self):
        for raw in [
            "missing json", json.dumps(observation(landscape_visible="true")),
            json.dumps(observation(context="not_a_context")), json.dumps(observation(reason=" ")),
            json.dumps(observation(obstruction="G")), json.dumps(observation(explanation="text")),
        ]:
            result = reviewer._parse(raw)
            self.assertIn("error", result)
            self.assertEqual(raw, result["raw"])

    def test_ollama_call_receives_only_pixels_view_broad_label_and_rubric(self):
        payloads = []

        def api(_self, route, payload=None):
            if route == "/api/tags":
                return {"models": [{"name": "qwen3-vl:8b", "digest": "test-digest"}]}
            if route == "/api/version":
                return {"version": "test-version"}
            payloads.append(payload)
            return {"message": {"content": json.dumps(observation())}}

        with tempfile.TemporaryDirectory() as tmp, patch.object(reviewer.OllamaBackend, "_api", api):
            folder = Path(tmp) / "G_water"
            folder.mkdir()
            path = folder / "340_638_34063818W.jpg"
            image = Image.new("RGB", (1600, 1200))
            exif = Image.Exif()
            exif[270] = "metadata-leak"
            image.save(path, exif=exif)
            engine = reviewer.make_backend("local-small", "qwen3-vl:8b")
            results = engine.review_batch([(path, "W", "G"), (path.with_suffix(".absent"), "P", "G")])
            rendered = reviewer._image(path)
            self.assertEqual(rendered.size, (1024, 768))
            self.assertFalse(rendered.getexif())
            self.assertEqual(results[0]["observation"], observation())
            self.assertIn("G — Water", payloads[0]["messages"][1]["content"])
            self.assertNotIn("G11", payloads[0]["messages"][1]["content"])
            self.assertIn("error", results[1])
            self.assertEqual(engine.identity["model_digest"], "test-digest")
            self.assertEqual(len(payloads), 1)
            self.assertNotIn("think", payloads[0])
            self.assertNotIn("34063818", str(payloads))
            self.assertNotIn(str(folder), str(payloads))

    def test_batch_failure_isolated_without_losing_order(self):
        engine = reviewer.VllmBackend.__new__(reviewer.VllmBackend)
        engine.params = None

        def generate(requests, _params, use_tqdm):
            if "bad" in requests:
                raise ValueError("invalid image")
            return [SimpleNamespace(outputs=[SimpleNamespace(text=json.dumps(observation(reason=req)))]) for req in requests]

        engine.engine = SimpleNamespace(generate=generate)
        results = engine._generate(["first", "bad", "last"])
        self.assertEqual(results[0]["observation"]["reason"], "first")
        self.assertIn("error", results[1])
        self.assertEqual(results[2]["observation"]["reason"], "last")

    def test_novel_selects_separate_prompt_for_same_view_and_different_label(self):
        engine = reviewer.VllmBackend.__new__(reviewer.VllmBackend)
        engine.prompts = {(view, label): reviewer.prompt_preview(view, label)["user"]
                          for view in ("P", "N") for label in ("A", "E")}
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "photo.jpg"
            Image.new("RGB", (60, 40)).save(image)
            with patch.object(engine, "_generate", return_value=[
                {"raw": "a", "observation": observation()},
                {"raw": "b", "observation": observation()},
            ]) as generate:
                engine.review_batch([(image, "N", "E"), (image, "N", "A")])
        first, second = generate.call_args.args[0]
        self.assertIn("E — Grassland", first["prompt"])
        self.assertIn("A — Artificial land", second["prompt"])

    def test_length_limited_thinking_build_retries_before_marking_image_as_failed(self):
        requests = []

        def api(_self, route, payload=None):
            if route == "/api/tags":
                return {"models": [{"name": "qwen3-vl:8b", "digest": "sha", "capabilities": ["thinking"]}]}
            if route == "/api/version":
                return {"version": "test"}
            requests.append((payload["think"], payload["options"]["num_predict"], payload["options"]["temperature"]))
            if len(requests) < 4:
                return {"message": {"content": ""}, "done_reason": "length", "eval_count": payload["options"]["num_predict"]}
            return {"message": {"content": json.dumps(observation())}, "done_reason": "stop"}

        with tempfile.TemporaryDirectory() as tmp, patch.object(reviewer.OllamaBackend, "_api", api):
            path = Path(tmp) / "view.jpg"
            Image.new("RGB", (100, 100)).save(path)
            backend = reviewer.make_backend("local-small", "qwen3-vl:8b")
            result, = backend.review_batch([(path, "C", "C")])
        self.assertEqual(requests, [(False, 2048, 0), (False, 6000, 0), (False, 2048, 0.3), (False, 2048, 0.6)])
        self.assertEqual(result["prior_attempts"][0]["done_reason"], "length")
        self.assertEqual(len(result["prior_attempts"]), 3)
        self.assertEqual(result["temperature"], 0.6)
        self.assertEqual(result["observation"], observation())
