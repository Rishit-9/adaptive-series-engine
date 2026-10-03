import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from src.symbolic_engine import SymbolicExpansionEngine
from src.neural_optimizer import NeuralCoefficientOptimizer

def evaluate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def run_prototype():
    engine = SymbolicExpansionEngine()

    # =========================================================================
    # TEST CASE 1: Single-Variable Taylor Expansion vs. Neural Refinement
    # Target: f(x) = cos(x) * exp(-0.1 * x^2) over [-3.2, 3.2]
    # =========================================================================
    print("--- Running Prototype Test 1: Taylor Expansion Refinement ---")
    x_sym = engine.x
    sym_target = sp.cos(x_sym) * sp.exp(-0.1 * x_sym**2)
    num_target_1 = sp.lambdify(x_sym, sym_target, "numpy")

    x_vals_1 = np.linspace(-3.2, 3.2, 400, dtype=np.float32)
    y_true_1 = num_target_1(x_vals_1).astype(np.float32)

    # Step 1: Symbolic Base Generation (Degree 6 -> 7 terms)
    c_base_taylor, phi_taylor = engine.taylor_expansion(sym_target, x_vals_1, center=0.0, n_terms=7)
    y_static_taylor = phi_taylor @ c_base_taylor

    # Step 2: Neural Coefficient Optimization
    opt_taylor = NeuralCoefficientOptimizer(c_base_taylor, learning_rate=0.005, alpha_peak=0.1, beta_tv=0.02)
    c_refined_taylor, y_hybrid_taylor = opt_taylor.fit(phi_taylor, y_true_1, epochs=600)

    mse_static_1 = evaluate_mse(y_true_1, y_static_taylor)
    mse_hybrid_1 = evaluate_mse(y_true_1, y_hybrid_taylor)
    reduction_1 = ((mse_static_1 - mse_hybrid_1) / mse_static_1) * 100

    print(f"Taylor Static MSE: {mse_static_1:.6f}")
    print(f"Taylor Hybrid MSE: {mse_hybrid_1:.6f} ({reduction_1:.2f}% Error Reduction)\n")

    # =========================================================================
    # TEST CASE 2: Single-Variable Fourier Series vs. Neural Refinement (Gibbs)
    # Target: Discontinuous Square Wave f(x) = sign(x) over [-pi, pi]
    # =========================================================================
    print("--- Running Prototype Test 2: Fourier Expansion Refinement (Gibbs) ---")
    x_vals_2 = np.linspace(-np.pi, np.pi, 500, dtype=np.float32)
    num_target_2 = lambda x: np.sign(x)
    y_true_2 = num_target_2(x_vals_2).astype(np.float32)

    # Step 1: Symbolic Base Generation (9 harmonics)
    c_base_fourier, phi_fourier = engine.fourier_expansion(num_target_2, x_vals_2, period_L=np.pi, n_terms=9)
    y_static_fourier = phi_fourier @ c_base_fourier

    # Step 2: Neural Coefficient Optimization (Higher peak & TV penalty to damp Gibbs overshoot)
    opt_fourier = NeuralCoefficientOptimizer(c_base_fourier, learning_rate=0.01, alpha_peak=0.35, beta_tv=0.20)
    c_refined_fourier, y_hybrid_fourier = opt_fourier.fit(phi_fourier, y_true_2, epochs=600)

    mse_static_2 = evaluate_mse(y_true_2, y_static_fourier)
    mse_hybrid_2 = evaluate_mse(y_true_2, y_hybrid_fourier)
    overshoot_static = np.max(np.abs(y_static_fourier)) - 1.0
    overshoot_hybrid = np.max(np.abs(y_hybrid_fourier)) - 1.0

    print(f"Fourier Static Overshoot: {overshoot_static*100:.2f}% | Hybrid Overshoot: {overshoot_hybrid*100:.2f}%")
    print(f"Fourier Static MSE: {mse_static_2:.6f} | Hybrid MSE: {mse_hybrid_2:.6f}\n")

    # =========================================================================
    # VISUALIZATION DASHBOARD
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Plot 1: Taylor vs Hybrid
    axes[0].plot(x_vals_1, y_true_1, 'k-', linewidth=2.2, label='Ground Truth f(x)')
    axes[0].plot(x_vals_1, y_static_taylor, 'r--', linewidth=1.5, label=f'Static Taylor (MSE: {mse_static_1:.3f})')
    axes[0].plot(x_vals_1, y_hybrid_taylor, 'b-', linewidth=2.0, label=f'Hybrid Refined (MSE: {mse_hybrid_1:.3f})')
    axes[0].set_ylim(-1.5, 1.5)
    axes[0].set_title(f'Taylor Boundary Stabilization ({reduction_1:.1f}% Error Drop)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Fourier vs Hybrid (Gibbs Damping)
    axes[1].plot(x_vals_2, y_true_2, 'k-', linewidth=2.2, label='Square Wave')
    axes[1].plot(x_vals_2, y_static_fourier, 'r--', alpha=0.7, label=f'Static Fourier (Overshoot: {overshoot_static*100:.1f}%)')
    axes[1].plot(x_vals_2, y_hybrid_fourier, 'g-', linewidth=2.0, label=f'Hybrid Refined (Overshoot: {overshoot_hybrid*100:.1f}%)')
    axes[1].set_title('Gibbs Phenomenon Suppression via Custom Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('milestone_2_prototype_results.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    run_prototype()