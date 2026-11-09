import numpy as np


def paired_bootstrap(differences, resamples=10000, seed=42):
    differences = np.asarray(differences, dtype=float)
    if not len(differences) or not np.isfinite(differences).all():
        raise ValueError("No finite paired differences")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(resamples, len(differences)))
    draws = differences[indices].mean(1)
    return {"mean": float(differences.mean()), "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "two_se": float(2 * differences.std(ddof=1) / np.sqrt(len(differences))) if len(differences) > 1 else None,
            "n": len(differences)}


def recovery(a2, a3, a4, resamples=10000, seed=42):
    a2, a3, a4 = (np.asarray(a, dtype=float) for a in (a2, a3, a4))
    if not (a2.shape == a3.shape == a4.shape) or a2.ndim != 1:
        raise ValueError("Recovery requires identically paired scenes")
    denominator = paired_bootstrap(a3 - a2, resamples, seed)
    output = {"denominator": denominator, "gain": paired_bootstrap(a4 - a2, resamples, seed),
              "R": None, "ci95": None}
    if denominator["ci95"][0] <= 0:
        output["reason"] = "Upsampling gain is not significantly positive; report raw differences"
        return output
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a2), size=(resamples, len(a2)))
    den = (a3 - a2)[idx].mean(1)
    if np.any(den <= 0):
        output["reason"] = "Bootstrap denominator crosses zero; ratio interval is unstable"
        return output
    draws = (a4 - a2)[idx].mean(1) / den
    output.update(R=float((a4 - a2).mean() / (a3 - a2).mean()),
                  ci95=np.quantile(draws, [0.025, 0.975]).tolist())
    return output


def holm(pvalues, family_size=None):
    p = np.asarray(pvalues, dtype=float)
    m = len(p) if family_size is None else family_size
    if m < len(p) or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid p-values/family size")
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate((m - np.arange(len(p))) * p[order]))
    return adjusted
