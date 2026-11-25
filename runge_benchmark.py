import numpy as np
from src.symbolic_engine import SymbolicExpansionEngine
from src.neural_optimizer import NeuralCoefficientOptimizer
from src.router import AdaptiveBasisRouter


def run_integration_benchmark():
    print("Integration & Runge's Phenomenon Benchmark ---")

    # Runge's function: f(x) = 1 / (1 + 25x^2)
    x_vals = np.linspace(-1, 1, 100, dtype=np.float32)
    y_true = 1.0 / (1.0 + 25.0 * x_vals**2)
    runge_func = lambda x: 1.0 / (1.0 + 25.0 * x**2)

    # 1. Adaptive Routing
    router = AdaptiveBasisRouter()
    selected_basis = router.route(x_vals, y_true).lower()
    print(f"[Router] Selected Basis: {selected_basis.upper()}")

    # 2. Symbolic Generation
    engine = SymbolicExpansionEngine()
    runge_sym = 1 / (1 + 25 * engine.x**2)

    if selected_basis == "chebyshev":
        c_base, phi_matrix = engine.chebyshev_expansion(runge_func, x_vals, n_terms=11)
    elif selected_basis == "fourier":
        c_base, phi_matrix = engine.fourier_expansion(runge_func, x_vals, n_terms=11)
    elif selected_basis == "taylor":
        c_base, phi_matrix = engine.taylor_expansion(runge_sym, x_vals, n_terms=11)
    else:
        raise ValueError(f"Unsupported basis returned by router: {selected_basis}")

    y_static = phi_matrix @ c_base

    # 3. Neural Optimization
    optimizer = NeuralCoefficientOptimizer(c_base, learning_rate=0.01, alpha_peak=0.3)
    c_refined, y_hybrid = optimizer.fit(phi_matrix, y_true, epochs=400)

    static_mse = np.mean((y_true - y_static) ** 2)
    hybrid_mse = np.mean((y_true - y_hybrid) ** 2)
    improvement = ((static_mse - hybrid_mse) / static_mse) * 100 if static_mse > 0 else 0.0

    print(f"Static MSE ({selected_basis.upper()}): {static_mse:.6f}")
    print(f"Hybrid MSE (Mitigated): {hybrid_mse:.6f}")
    print(f"MSE Improvement: {improvement:.2f}%")


if __name__ == "__main__":
    run_integration_benchmark()
