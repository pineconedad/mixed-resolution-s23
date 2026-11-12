import torch

from mixres.study_runner import StudyRunner


def test_sweep_cache_reuses_exact_preprocessing_by_view_and_entry(monkeypatch):
    calls = []

    def load(path, view, entry):
        calls.append(entry)
        return torch.zeros(3, 28, 28)

    monkeypatch.setattr("mixres.study_runner.load_view", load)
    runner = StudyRunner.__new__(StudyRunner)
    view = {"sharp": False, "image": "sample.png", "hw": [28, 28], "hi_hw": [56, 56]}
    first = runner.cached_view("sample.png", view, "pixel")
    assert first is runner.cached_view("sample.png", view, "pixel")
    runner.cached_view("sample.png", view, "small")
    sharp = {**view, "sharp": True}
    assert runner.cached_view("sample.png", sharp, "pixel") is runner.cached_view("sample.png", sharp, "small")
    assert calls == ["pixel", "small", "pixel"]


def test_cache_evicts_cpu_images_without_changing_values(monkeypatch):
    monkeypatch.setattr("mixres.study_runner.load_view", lambda *args: torch.ones(3, 2, 2))
    runner = StudyRunner.__new__(StudyRunner)
    runner.image_cache_limit_bytes = 48
    runner.cached_view("one", {"sharp": False, "image": "one"}, "small")
    pixels = runner.cached_view("two", {"sharp": False, "image": "two"}, "small")
    assert len(runner.image_cache) == 1 and runner.image_cache_bytes == 48
    assert torch.equal(pixels, torch.ones(3, 2, 2))
