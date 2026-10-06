"""
Module: optimizer.py
Project: Diesel Backpressure and Emission Optimization Toolkit
Author: Billel Anisse Bessekek
Affiliation: Moscow Automobile and Road Construction State Technical University (MADI)
Date: 2026
"""

import math
from typing import Dict, List

class DieselBackpressureAnalyzer:
    """
    Simulates the internal thermodynamic and fluid-dynamic impact of exhaust 
    backpressure variations on medium-duty diesel engine cycles.
    """
    def __init__(self, displacement_l: float = 4.75, rated_power_kw: float = 110.0):
        self.displacement = displacement_l
        self.rated_power = rated_power_kw
        self.base_gamma_r = 0.035  # Baseline residual gas fraction

    def compute_residual_gas_fraction(self, delta_p_kpa: float) -> float:
        """
        Calculates residual gas fraction (gamma_r) based on exhaust tract flow resistance.
        """
        k_sensitivity = 0.00185
        gamma_r = self.base_gamma_r + (k_sensitivity * delta_p_kpa)
        return min(gamma_r, 0.12)

    def evaluate_cycle_metrics(self, delta_p_kpa: float) -> Dict[str, float]:
        """
        Returns relative combustion, thermal efficiency, and emission factors.
        """
        gamma_r = self.compute_residual_gas_fraction(delta_p_kpa)
        excess_gas = max(0.0, gamma_r - self.base_gamma_r)

        # Kinetic and thermodynamic correlations based on Diesel-RK calibration
        rel_nox = max(0.65, 1.0 - (1.18 * excess_gas))
        rel_soot = 1.0 + (13.2 * (excess_gas ** 1.35))
        bsfc_penalty_pct = round(0.315 * delta_p_kpa, 2)
        indicated_eff_drop_pct = round(0.18 * delta_p_kpa, 2)

        return {
            "backpressure_kpa": delta_p_kpa,
            "gamma_r": round(gamma_r, 4),
            "nox_index": round(rel_nox, 4),
            "soot_index": round(rel_soot, 4),
            "bsfc_increase_pct": bsfc_penalty_pct,
            "indicated_efficiency_drop_pct": indicated_eff_drop_pct
        }

    def generate_parametric_sweep(self, max_delta_p: float = 35.0, steps: int = 8) -> List[Dict[str, float]]:
        """
        Runs parametric calculations over a range of backpressure levels.
        """
        step_size = max_delta_p / steps
        results = []
        for i in range(steps + 1):
            p = round(i * step_size, 1)
            results.append(self.evaluate_cycle_metrics(p))
        return results


if __name__ == "__main__":
    analyzer = DieselBackpressureAnalyzer(displacement_l=4.75, rated_power_kw=110.0)
    print("=== MADI Scientific Simulation Module: Diesel Backpressure Optimization ===")
    print("Author: Bessekek Billel Anisse | Institution: MADI\n")
    sweep = analyzer.generate_parametric_sweep()
    for row in sweep:
        print(f"ΔP: {row['backpressure_kpa']:4.1f} kPa | γ_r: {row['gamma_r']:.4f} | "
              f"NOx Index: {row['nox_index']:.3f} | Soot Index: {row['soot_index']:.3f} | "
              f"+BSFC: {row['bsfc_increase_pct']}%")
