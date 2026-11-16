from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import sympy as sp
from src.symbolic_engine import SymbolicExpansionEngine
from src.neural_optimizer import NeuralCoefficientOptimizer
from src.router import AdaptiveBasisRouter

app = FastAPI(
    title="Adaptive Function Approximation Engine (AFAE)",
    description="Neuro-Symbolic Series Expansion API",
    version="1.0.0"
)

class ApproximationRequest(BaseModel):
    expression: str
    x_min: float = -3.14159
    x_max: float = 3.14159
    num_points: int = 400
    n_terms: int = 7
    epochs: int = 300

@app.post("/api/v1/optimize/taylor")
async def optimize_taylor(req: ApproximationRequest):
    try:
        engine = SymbolicExpansionEngine()
        x_sym = engine.x
        
        sym_target = sp.sympify(req.expression)
        num_target = sp.lambdify(x_sym, sym_target, "numpy")
        
        x_vals = np.linspace(req.x_min, req.x_max, req.num_points, dtype=np.float32)
        y_true = num_target(x_vals).astype(np.float32)
        
        c_base, phi_matrix = engine.taylor_expansion(
            sym_target, x_vals, center=(req.x_max + req.x_min)/2, n_terms=req.n_terms
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

@app.post("/api/v1/route")
async def auto_route_basis(req: ApproximationRequest):
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