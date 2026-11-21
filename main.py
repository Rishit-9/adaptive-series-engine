from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import sympy as sp
from src.symbolic_engine import SymbolicExpansionEngine
from src.neural_optimizer import NeuralCoefficientOptimizer
from src.router import AdaptiveBasisRouter

app = FastAPI(
    title="Adaptive Function Approximation Engine (AFAE)",
    description="Neuro-Symbolic Series Expansion API with Dynamic Basis Routing",
    version="1.0.0"
)

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


# Pydantic models for request validation
class ApproximationRequest(BaseModel):
    expression: str
    x_min: float = -3.14159
    x_max: float = 3.14159
    num_points: int = 400
    n_terms: int = 7
    epochs: int = 300

@app.post("/api/v1/optimize/auto")
async def optimize_auto(req: ApproximationRequest):
    """
    The core AFAE endpoint. Evaluates the function, dynamically routes it 
    to the optimal basis expansion, and executes the neural refinement.
    """
    try:
        engine = SymbolicExpansionEngine()
        x_sym = sp.sympify(req.expression)
        num_target = sp.lambdify(engine.x, x_sym, "numpy")
        
        x_vals = np.linspace(req.x_min, req.x_max, req.num_points, dtype=np.float32)
        y_true = num_target(x_vals).astype(np.float32)
        
        # 1. Pass data to the Router for intelligent basis selection
        router = AdaptiveBasisRouter()
        best_basis = router.route(x_vals, y_true)
        
        # 2. Execute Symbolic Generation based on routing decision
        if best_basis == "taylor":
            c_base, phi_matrix = engine.taylor_expansion(
                x_sym, x_vals, center=(req.x_max + req.x_min)/2, n_terms=req.n_terms
            )
        elif best_basis == "fourier":
            period = (req.x_max - req.x_min) / 2.0
            c_base, phi_matrix = engine.fourier_expansion(
                num_target, x_vals, period_L=period, n_terms=req.n_terms
            )
        else: # chebyshev
            # Chebyshev mathematically requires the [-1, 1] domain for its nodes
            x_vals_cheb = np.linspace(-1.0, 1.0, req.num_points, dtype=np.float32)
            y_true_cheb = num_target(x_vals_cheb).astype(np.float32)
            c_base, phi_matrix = engine.chebyshev_expansion(
                num_target, x_vals_cheb, n_terms=req.n_terms
            )
            y_true = y_true_cheb # Align true values for MSE check
            
        # 3. Neural Optimization
        optimizer = NeuralCoefficientOptimizer(c_base)
        c_refined, y_hybrid = optimizer.fit(phi_matrix, y_true, epochs=req.epochs, verbose=False)
        
        mse_static = float(np.mean((y_true - (phi_matrix @ c_base))**2))
        mse_hybrid = float(np.mean((y_true - y_hybrid)**2))
        
        return {
            "status": "success",
            "routed_basis": best_basis,
            "mse_static": mse_static,
            "mse_hybrid": mse_hybrid,
            "c_base": c_base.tolist(),
            "c_refined": c_refined.tolist(),
            "improvement_pct": round(((mse_static - mse_hybrid) / mse_static) * 100, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/optimize/taylor")
async def optimize_taylor(req: ApproximationRequest):
    """Manual override to force a Taylor series expansion."""
    try:
        engine = SymbolicExpansionEngine()
        x_sym = sp.sympify(req.expression)
        num_target = sp.lambdify(engine.x, x_sym, "numpy")
        
        x_vals = np.linspace(req.x_min, req.x_max, req.num_points, dtype=np.float32)
        y_true = num_target(x_vals).astype(np.float32)
        
        c_base, phi_matrix = engine.taylor_expansion(
            x_sym, x_vals, center=(req.x_max + req.x_min)/2, n_terms=req.n_terms
        )
        
        optimizer = NeuralCoefficientOptimizer(c_base)
        c_refined, y_hybrid = optimizer.fit(phi_matrix, y_true, epochs=req.epochs, verbose=False)
        
        mse_static = float(np.mean((y_true - (phi_matrix @ c_base))**2))
        mse_hybrid = float(np.mean((y_true - y_hybrid)**2))
        
        return {
            "status": "success",
            "basis": "taylor",
            "mse_static": mse_static,
            "mse_hybrid": mse_hybrid,
            "c_base": c_base.tolist(),
            "c_refined": c_refined.tolist(),
            "improvement_pct": round(((mse_static - mse_hybrid) / mse_static) * 100, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/optimize/fourier")
async def optimize_fourier(req: ApproximationRequest):
    """Manual override to force a Fourier series expansion."""
    try:
        engine = SymbolicExpansionEngine()
        x_sym = sp.sympify(req.expression)
        num_target = sp.lambdify(engine.x, x_sym, "numpy")
        
        x_vals = np.linspace(req.x_min, req.x_max, req.num_points, dtype=np.float32)
        y_true = num_target(x_vals).astype(np.float32)
        
        period = (req.x_max - req.x_min) / 2.0
        
        c_base, phi_matrix = engine.fourier_expansion(
            num_target, x_vals, period_L=period, n_terms=req.n_terms
        )
        
        optimizer = NeuralCoefficientOptimizer(c_base)
        c_refined, y_hybrid = optimizer.fit(phi_matrix, y_true, epochs=req.epochs, verbose=False)
        
        mse_static = float(np.mean((y_true - (phi_matrix @ c_base))**2))
        mse_hybrid = float(np.mean((y_true - y_hybrid)**2))
        
        return {
            "status": "success",
            "basis": "fourier",
            "mse_static": mse_static,
            "mse_hybrid": mse_hybrid,
            "c_base": c_base.tolist(),
            "c_refined": c_refined.tolist(),
            "improvement_pct": round(((mse_static - mse_hybrid) / mse_static) * 100, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/optimize/chebyshev")
async def optimize_chebyshev(req: ApproximationRequest):
    """Manual override to force a Chebyshev polynomial expansion."""
    try:
        engine = SymbolicExpansionEngine()
        x_sym = sp.sympify(req.expression)
        num_target = sp.lambdify(engine.x, x_sym, "numpy")
        
        x_vals = np.linspace(-1.0, 1.0, req.num_points, dtype=np.float32)
        y_true = num_target(x_vals).astype(np.float32)
        
        c_base, phi_matrix = engine.chebyshev_expansion(
            num_target, x_vals, n_terms=req.n_terms
        )
        
        optimizer = NeuralCoefficientOptimizer(c_base)
        c_refined, y_hybrid = optimizer.fit(phi_matrix, y_true, epochs=req.epochs, verbose=False)
        
        mse_static = float(np.mean((y_true - (phi_matrix @ c_base))**2))
        mse_hybrid = float(np.mean((y_true - y_hybrid)**2))
        
        return {
            "status": "success",
            "basis": "chebyshev",
            "mse_static": mse_static,
            "mse_hybrid": mse_hybrid,
            "c_base": c_base.tolist(),
            "c_refined": c_refined.tolist(),
            "improvement_pct": round(((mse_static - mse_hybrid) / mse_static) * 100, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/route")
async def auto_route_basis(req: ApproximationRequest):
    """Standalone endpoint to evaluate a function and return the optimal basis recommendation."""
    try:
        engine = SymbolicExpansionEngine()
        x_sym = sp.sympify(req.expression)
        num_target = sp.lambdify(engine.x, x_sym, "numpy")
        
        x_vals = np.linspace(req.x_min, req.x_max, req.num_points, dtype=np.float32)
        y_true = num_target(x_vals).astype(np.float32)
        
        router = AdaptiveBasisRouter()
        best_basis = router.route(x_vals, y_true)
        
        return {"expression": req.expression, "recommended_basis": best_basis}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))