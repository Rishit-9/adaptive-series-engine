import sympy as sp
import numpy as np
from scipy.integrate import quad

class SymbolicExpansionEngine:
    """
    Generates static series expansions (Taylor, Fourier, Chebyshev)
    and extracts Base Coefficients (C_base) and Basis Matrices (Phi)
    for neural hand-off.
    """
    def __init__(self):
        self.x = sp.Symbol('x')

    def taylor_expansion(self, sym_func, x_vals, center=0.0, n_terms=8):
        """
        Computes exact symbolic derivatives at `center` to extract Taylor C_base.
        Basis: phi_n(x) = (x - a)^n
        """
        c_base = []
        basis_matrix = np.zeros((len(x_vals), n_terms), dtype=np.float32)

        current_deriv = sym_func
        for n in range(n_terms):
            if n > 0:
                current_deriv = sp.diff(current_deriv, self.x)
            
            # Evaluate n-th derivative at expansion center 'a'
            deriv_at_a = float(current_deriv.subs(self.x, center).evalf())
            coeff = deriv_at_a / float(sp.factorial(n))
            c_base.append(coeff)

            # Populate basis column: (x - center)^n
            basis_matrix[:, n] = (x_vals - center) ** n

        return np.array(c_base, dtype=np.float32), basis_matrix

    def fourier_expansion(self, num_func, x_vals, period_L=np.pi, n_terms=10):
        """
        Computes Fourier coefficients [a_0/2, a_1, b_1, ..., a_n, b_n].
        Basis: [1, cos(pi*x/L), sin(pi*x/L), ..., cos(n*pi*x/L), sin(n*pi*x/L)]
        """
        total_coeffs = 1 + 2 * n_terms
        c_base = np.zeros(total_coeffs, dtype=np.float32)
        basis_matrix = np.zeros((len(x_vals), total_coeffs), dtype=np.float32)

        # a_0 / 2 term (Constant DC offset)
        a0, _ = quad(lambda x: num_func(x), -period_L, period_L, limit=100)
        c_base[0] = a0 / (2.0 * period_L)
        basis_matrix[:, 0] = 1.0

        # Harmonic terms (a_n and b_n)
        col_idx = 1
        for n in range(1, n_terms + 1):
            freq = (n * np.pi) / period_L
            
            an, _ = quad(lambda x: num_func(x) * np.cos(freq * x), -period_L, period_L, limit=100)
            bn, _ = quad(lambda x: num_func(x) * np.sin(freq * x), -period_L, period_L, limit=100)

            c_base[col_idx] = an / period_L
            c_base[col_idx + 1] = bn / period_L

            basis_matrix[:, col_idx] = np.cos(freq * x_vals)
            basis_matrix[:, col_idx + 1] = np.sin(freq * x_vals)
            col_idx += 2

        return c_base, basis_matrix

    def chebyshev_expansion(self, num_func, x_vals, n_terms=10):
        """
        Fits Chebyshev polynomials of the first kind T_k(x) over [-1, 1]
        using Chebyshev nodes to extract C_base.
        """
        # Generate Chebyshev nodes: cos((2k - 1)*pi / 2n)
        k = np.arange(1, n_terms + 1)
        cheb_nodes = np.cos((2 * k - 1) * np.pi / (2 * n_terms))
        y_nodes = num_func(cheb_nodes)

        # Fit Chebyshev series to nodes
        cheb_poly = np.polynomial.chebyshev.Chebyshev.fit(
            cheb_nodes, y_nodes, deg=n_terms - 1, domain=[-1, 1]
        )
        c_base = cheb_poly.coef.astype(np.float32)

        # Evaluate Chebyshev basis polynomials T_k(x) at x_vals
        basis_matrix = np.zeros((len(x_vals), n_terms), dtype=np.float32)
        for i in range(n_terms):
            coeffs = np.zeros(i + 1)
            coeffs[i] = 1.0
            basis_matrix[:, i] = np.polynomial.chebyshev.chebval(x_vals, coeffs)

        return c_base, basis_matrix
