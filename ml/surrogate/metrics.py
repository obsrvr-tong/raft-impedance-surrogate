"""Error metric for impedance curves.

For one case and one term, the error is the largest deviation over the frequency band, divided
by the largest |K| of the true curve over all frequencies, in percent:

    e = 100 * max_f |K_pred(f) - K_true(f)| / max_f |K_true(f)|

Dividing by max |K| rather than by K(f) keeps the metric finite where Re K crosses zero (Kxx and
Kyy at 26-33 Hz on the larger footprints, notebook 01). It is the measure used by
automation/tools/compare_impe.py.

Summaries over a test set (summarize):
  err_<term>  95th percentile over the test cases: "95 % of new cases are within this"
  err_worst   the largest err_<term>: the bottleneck term
  err_mean    mean over all cases and terms: the typical error, the most stable for comparisons
"""
import numpy as np

from .data import TargetSpec, targets_to_K


def curve_errors(K_true, K_pred, freqs=None, band=None):
    """Errors in % for complex curves of shape (n, n_freq, n_terms) -> (n, n_terms).

    band = (f_lo, f_hi) restricts where deviations are measured (needs freqs); the normaliser
    max |K_true| is always taken over the full curve, so band errors add up consistently.
    """
    K_true, K_pred = np.asarray(K_true), np.asarray(K_pred)
    dev = np.abs(K_pred - K_true)
    if band is not None:
        f = np.asarray(freqs)
        dev = dev[:, (f >= band[0] - 1e-9) & (f <= band[1] + 1e-9)]
    return 100 * dev.max(axis=1) / np.abs(K_true).max(axis=1)


def target_errors(Y_true, Y_pred, spec: TargetSpec, freqs=None, band=None):
    """The same, for target rows (n, n_columns) as produced by make_targets."""
    return curve_errors(targets_to_K(Y_true, spec), targets_to_K(Y_pred, spec), freqs, band)


def summarize(errors, terms):
    """{'err_Ux': p95, ..., 'err_worst': max of the p95s, 'err_mean': mean} from (n, n_terms) errors."""
    p95 = np.percentile(errors, 95, axis=0)
    out = {f"err_{t}": float(v) for t, v in zip(terms, p95)}
    out["err_worst"] = float(p95.max())
    out["err_mean"] = float(np.mean(errors))
    return out


def error_spectrum(Y_true, Y_pred, spec: TargetSpec):
    """Deviation at every frequency, in % of each curve's max |K|: (n, n_freq, n_terms)."""
    K_true, K_pred = targets_to_K(Y_true, spec), targets_to_K(Y_pred, spec)
    return 100 * np.abs(K_pred - K_true) / np.abs(K_true).max(axis=1, keepdims=True)
