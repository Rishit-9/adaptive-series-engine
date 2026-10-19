import numpy as np

class AdaptiveBasisRouter:
    """
    Analyzes input function data to dynamically route to the optimal
    symbolic basis (Fourier, Taylor, or Chebyshev).
    """
    def __init__(self, variance_threshold=2.0, periodicity_tolerance=0.1):
        self.var_thresh = variance_threshold
        self.per_tol = periodicity_tolerance

    def calculate_smoothness(self, y_vals):
        """Approximates the second derivative to check for sharp jumps."""
        d2y = np.gradient(np.gradient(y_vals))
        return np.max(np.abs(d2y))

    def detect_periodicity(self, y_vals):
        """Uses FFT to detect dominant repeating frequencies."""
        fft_vals = np.abs(np.fft.fft(y_vals))
        dominant_freq = np.max(fft_vals[1:]) # Ignore DC offset
        return dominant_freq > (np.mean(fft_vals) * 3)

    def route(self, x_vals, y_vals):
        """
        Decision Matrix for Basis Routing:
        - Highly periodic -> Fourier
        - Smooth, centralized -> Taylor
        - Bounded with edge variance (Runge-prone) -> Chebyshev
        """
        is_periodic = self.detect_periodicity(y_vals)
        smoothness = self.calculate_smoothness(y_vals)
        variance = np.var(y_vals)

        if is_periodic and smoothness > self.var_thresh:
            return "fourier"
        elif variance < self.var_thresh and smoothness < 1.0:
            return "taylor"
        else:
            # Default to Chebyshev for bounded, high-variance domains 
            # to mitigate Runge's phenomenon
            return "chebyshev"