import numpy as np


def paired_bootstrap(differences, resamples=10000, seed=42):
    differences = np.asarray(differences, dtype=float)
    if not len(differences) or not np.isfinite(differences).all():
        raise ValueError("No finite paired differences")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(resamples, len(differences)))
    draws = differences[indices].mean(1)
    return {"mean": float(differences.mean()), "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "p_two_sided": float(min(1.0, 2 * min(np.mean(draws <= 0), np.mean(draws >= 0)))),
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


def paired_sign_flip(differences, resamples=10000, seed=42, exact_max_n=16):
    """Two-sided randomization test under exchangeable signs of paired differences."""
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 1 or not differences.size or not np.isfinite(differences).all():
        raise ValueError("No finite paired differences")
    if not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    # Zero differences do not change any randomization statistic.
    nonzero = differences[differences != 0]
    n = len(nonzero)
    exact = n <= exact_max_n
    count = 2 ** n if exact else resamples
    rng = np.random.default_rng(seed)
    observed = abs(nonzero.sum())
    tolerance = 1e-12 * max(1.0, float(np.abs(nonzero).sum()))
    extreme = 0
    for start in range(0, count, 1024):
        size = min(1024, count - start)
        if exact:
            bits = (np.arange(start, start + size, dtype=np.uint64)[:, None]
                    >> np.arange(n, dtype=np.uint64)) & 1
            signs = 2 * bits.astype(float) - 1
        else:
            signs = 2 * rng.integers(0, 2, size=(size, n)) - 1
        extreme += int(np.count_nonzero(np.abs(signs @ nonzero) >= observed - tolerance))
    return {"p_two_sided": float(extreme / count if exact else (extreme + 1) / (count + 1)),
            "p_method": "sign_flip_exact" if exact else "sign_flip_monte_carlo"}


def stratified_recovery(strata, resamples=10000, seed=42):
    """Bootstrap paired triples within datasets, then average dataset means equally.

    Each stratum is an n-scenes by 3 array: baseline, target, recovered.
    Dataset membership and each dataset's scene count stay fixed in every draw.
    """
    arrays = [np.asarray(rows, dtype=float) for rows in strata]
    if (not arrays or any(a.ndim != 2 or a.shape[1] != 3 or not len(a)
                          or not np.isfinite(a).all() for a in arrays)):
        raise ValueError("Recovery requires nonempty finite paired dataset triples")
    if not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    rng = np.random.default_rng(seed)
    draws = np.zeros((resamples, 2))
    means = []
    for array in arrays:
        differences = array[:, 1:] - array[:, :1]
        means.append(differences.mean(0))
        # Chunking avoids a resamples x scenes x metrics allocation for large datasets.
        for start in range(0, resamples, 1024):
            idx = rng.integers(0, len(array), size=(min(1024, resamples - start), len(array)))
            draws[start:start + len(idx)] += differences[idx].mean(1) / len(arrays)
    denominator, gain = np.mean(means, axis=0)
    intervals = np.quantile(draws, [0.025, 0.975], axis=0)
    output = {"denominator": {"mean": float(denominator), "ci95": intervals[:, 0].tolist()},
              "gain": {"mean": float(gain), "ci95": intervals[:, 1].tolist()},
              "R": None, "ci95": None}
    if intervals[0, 0] <= 0:
        output["reason"] = "Upsampling gain is not significantly positive; report raw differences"
    elif np.any(draws[:, 0] <= 0):
        output["reason"] = "Bootstrap denominator crosses zero; ratio interval is unstable"
    else:
        output.update(R=float(gain / denominator),
                      ci95=np.quantile(draws[:, 1] / draws[:, 0], [0.025, 0.975]).tolist())
    return output


def holm(pvalues, family_size=None):
    p = np.asarray(pvalues, dtype=float)
    m = len(p) if family_size is None else family_size
    if p.ndim != 1 or not np.isfinite(p).all() or m < len(p) or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid p-values/family size")
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate((m - np.arange(len(p))) * p[order]))
    return adjusted
