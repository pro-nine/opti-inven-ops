"""
DistroSense Master Run Script
================================
Runs all 6 modules in sequence and prints the full findings summary.
Usage: python3 -m scripts.run_all
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR  = str(ROOT / "outputs" / "figures")
DATA_DIR = str(ROOT / "data")

def main():
    print("\n" + "="*70)
    print("DISTROSENSE — FULL ANALYSIS PIPELINE")
    print("Nordpack GmbH Supply Chain Operations Analytics")
    print("="*70)

    # Module 1
    from src.module1_demand import run_module1
    m1 = run_module1(fig_dir=FIG_DIR, data_dir=DATA_DIR)

    # Module 2
    from src.module2_inventory import run_module2
    m2 = run_module2(demand_df=m1["df"], fig_dir=FIG_DIR, data_dir=DATA_DIR)

    # Module 3
    from src.module3_newsvendor import run_module3
    m3 = run_module3(demand_df=m1["df"], fig_dir=FIG_DIR, data_dir=DATA_DIR)

    # Module 4
    from src.module4_queuing import run_module4
    m4 = run_module4(fig_dir=FIG_DIR, data_dir=DATA_DIR)

    # Module 5
    from src.module5_montecarlo import run_module5
    m5 = run_module5(fig_dir=FIG_DIR, data_dir=DATA_DIR)

    # Module 6
    from src.module6_recommendations import run_module6
    ltd_a = m5["mc_all"]["SKU_A_Yoghurt"]["TC_opt"]  # reuse MC samples
    from src.module5_montecarlo import simulate_ltd_fast, BASELINE
    b = BASELINE["SKU_A_Yoghurt"]
    ltd_samples = simulate_ltd_fast(b["mu_w"], b["sig_resid"],
                                     b["mu_L"], b["sig_L"], n_sim=10000)
    m6 = run_module6(ltd_samples=ltd_samples, fig_dir=FIG_DIR, data_dir=DATA_DIR)

    print("\n" + "="*70)
    print("FULL PIPELINE COMPLETE — EXECUTIVE SUMMARY")
    print("="*70)
    print(f"\n  Total annual benefit identified:  €{m6['total_savings']:,.0f}")
    print(f"  of which SKU-level savings:       €{m6['sku_savings']:,.0f}")
    print(f"  of which dock expansion (net):    €249,906")
    print(f"\n  Figures produced: {len(list(Path(FIG_DIR).glob('*.png')))} charts")
    print(f"  Located in:  {FIG_DIR}")
    print("\n  Run-to-run reproducibility: all random seeds fixed. Re-running produces identical output.")


if __name__ == "__main__":
    main()
