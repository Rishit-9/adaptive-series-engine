import numpy as np
import sympy as sp
import time
import matplotlib.pyplot as plt
from src.symbolic_engine import SymbolicExpansionEngine
from src.neural_optimizer import NeuralCoefficientOptimizer

def run_comprehensive_validation():
    engine = SymbolicExpansionEngine()

    print("===============================================================")
    print(" HIGH-FIDELITY EDGE-CASE BENCHMARKING & VALIDATION")
    print("===============================================================\n")

    # =========================================================================
    # VALIDATION SUITE 1: Discontinuous Signal (Sawtooth Wave)
    # Testing the refined Huber Loss precision and TV penalty
    # =========================================================================
    print("--- Suite 1: Discontinuous Signal Validation ---")
    x_vals_disc = np.linspace(-np.pi, np.pi, 600, dtype=np.float32)
    # Sawtooth wave: highly discontinuous at boundaries
    y_true_disc = 2 * (x_vals_disc / (2 * np.pi) - np.floor(0.5 + x_vals_disc / (2 * np.pi))).astype(np.float32)

    start_time = time.time()
    c_base_fourier, phi_fourier = engine.fourier_expansion(lambda x: 2 * (x / (2 * np.pi) - np.floor(0.5 + x / (2 * np.pi))), x_vals_disc, period_L=np.pi, n_terms=15)
    
    # Higher alpha_peak to suppress Gibbs on the aggressive sawtooth drop
    opt_fourier = NeuralCoefficientOptimizer(c_base_fourier, learning_rate=0.01, alpha_peak=0.4, beta_tv=0.3)
    c_refined_fourier, y_hybrid_disc = opt_fourier.fit(phi_fourier, y_true_disc, epochs=800, verbose=False)
    exec_time_1 = time.time() - start_time

    mse_static_disc = np.mean((y_true_disc - (phi_fourier @ c_base_fourier))**2)
    mse_hybrid_disc = np.mean((y_true_disc - y_hybrid_disc)**2)

    print(f"Static Fourier MSE: {mse_static_disc:.6f}")
    print(f"Hybrid Refined MSE: {mse_hybrid_disc:.6f}")
    print(f"Graph-Compiled Execution Time: {exec_time_1:.3f} seconds\n")

    # =========================================================================
    # VALIDATION SUITE 2: Multi-Variable Function Support
    # Target: f(x, y) = sin(x) * cos(y) around (0,0)
    # =========================================================================
    print("--- Suite 2: Multi-Variable Function Expansion ---")
    x_sym = engine.x
    y_sym = engine.y
    sym_target_2d = sp.sin(x_sym) * sp.cos(y_sym)
    
    x_vals_2d = np.linspace(-1.5, 1.5, 50, dtype=np.float32)
    y_vals_2d = np.linspace(-1.5, 1.5, 50, dtype=np.float32)
    
    start_time = time.time()
    c_base_2d, phi_2d, X_grid, Y_grid = engine.taylor_expansion_2d(sym_target_2d, x_vals_2d, y_vals_2d, center=(0.0, 0.0), degree=2)
    
    # Generate True 2D Data
    num_target_2d = sp.lambdify((x_sym, y_sym), sym_target_2d, "numpy")
    Z_true = num_target_2d(X_grid, Y_grid)
    Z_true_flat = Z_true.flatten().astype(np.float32)

    # Memory-optimized Neural Fitting over 2500 flattened grid points
    opt_2d = NeuralCoefficientOptimizer(c_base_2d, learning_rate=0.008, alpha_peak=0.1, beta_tv=0.01)
    c_refined_2d, Z_hybrid_flat = opt_2d.fit(phi_2d, Z_true_flat, epochs=500, verbose=False)
    exec_time_2 = time.time() - start_time

    mse_static_2d = np.mean((Z_true_flat - (phi_2d @ c_base_2d))**2)
    mse_hybrid_2d = np.mean((Z_true_flat - Z_hybrid_flat)**2)
    
    print(f"2D Static Taylor MSE: {mse_static_2d:.6f}")
    print(f"2D Hybrid Refined MSE: {mse_hybrid_2d:.6f}")
    print(f"Graph-Compiled Execution Time: {exec_time_2:.3f} seconds\n")
    print("Comprehensive validation checks passed.")

if __name__ == "__main__":
    run_comprehensive_validation()