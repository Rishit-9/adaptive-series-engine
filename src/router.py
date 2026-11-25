import numpy as np

class AdaptiveBasisRouter:
    """
    Analyzes input function data to dynamically route to the optimal
    symbolic basis (Fourier, Taylor, or Chebyshev).
    """
    def __init__(self, variance_threshold=2.0, curvature_threshold=5.0, periodicity_tolerance=0.35):
        self.var_thresh = variance_threshold
        self.curv_thresh = curvature_threshold
        self.per_tol = periodicity_tolerance

    def calculate_smoothness(self, y_vals, x_vals=None):
        """Approximates the true second derivative d²y/dx² to measure curvature and sharpness."""
        if x_vals is not None:
            dy = np.gradient(y_vals, x_vals)
            d2y = np.gradient(dy, x_vals)
        else:
            d2y = np.gradient(np.gradient(y_vals))
        return float(np.max(np.abs(d2y)))

    def has_discontinuity(self, y_vals):
        """Detects sharp jump discontinuities (e.g., square or sawtooth waves)."""
        y_range = float(np.ptp(y_vals))
        if y_range < 1e-8:
            return False
        max_jump = float(np.max(np.abs(np.diff(y_vals))))
        return max_jump > 0.25 * y_range

    def detect_periodicity(self, y_vals, x_vals=None):
        """Uses centered rFFT power spectrum and boundary matching to detect true periodicity."""
        y_centered = y_vals - np.mean(y_vals)
        y_range = float(np.ptp(y_vals))
        if y_range < 1e-8:
            return False

        fft_power = np.abs(np.fft.rfft(y_centered))[1:] ** 2
        total_power = float(np.sum(fft_power))
        if total_power < 1e-12:
            return False

        dominant_idx = int(np.argmax(fft_power)) + 1  # Harmonic index k >= 1
        dominant_ratio = float(fft_power[dominant_idx - 1] / total_power)

        # Multi-cycle periodic wave (e.g., sin(3x) + 0.5*cos(7x), multi-cycle sawtooth)
        if dominant_idx >= 2 and dominant_ratio > self.per_tol:
            return True

        # Single-cycle periodic wave (e.g., sin(x) or cos(x) on [-pi, pi])
        if dominant_idx == 1 and dominant_ratio > 0.75:
            dy = np.gradient(y_vals, x_vals) if x_vals is not None else np.gradient(y_vals)
            max_slope = float(np.max(np.abs(dy))) + 1e-8
            endpoints_match = abs(float(y_vals[0] - y_vals[-1])) < 0.2 * y_range
            slopes_match = abs(float(dy[0] - dy[-1])) < 0.25 * max_slope
            return endpoints_match and slopes_match

        return False

    def route(self, x_vals, y_vals):
        """
        Decision Matrix for Basis Routing:
        - Periodic or discontinuous signal -> Fourier
        - Smooth, low-curvature analytic function -> Taylor
        - High-curvature bounded function (Runge-prone) -> Chebyshev
        """
        is_periodic = self.detect_periodicity(y_vals, x_vals)
        is_discontinuous = self.has_discontinuity(y_vals)
        smoothness = self.calculate_smoothness(y_vals, x_vals)
        variance = float(np.var(y_vals))

        if is_periodic or is_discontinuous:
            return "fourier"
        elif variance < self.var_thresh and smoothness < self.curv_thresh:
            return "taylor"
        else:
            return "chebyshev"
