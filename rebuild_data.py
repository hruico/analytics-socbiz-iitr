"""Script to rebuild analytical base with all fixes."""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.real_data_pipeline import build_real_data_pipeline

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "=" * 60)
    print("REBUILDING ANALYTICAL BASE WITH FIXES")
    print("=" * 60)
    print("\nFixes applied:")
    print("✓ FIX 1: Per-zone utilization (not system-wide)")
    print("✓ FIX 2: Real ACN + UrbanEV data (no synthetic)")
    print("✓ FIX 3: Temporal feature alignment (no time_step join)")
    print("✓ FIX 4: Data-driven peak hours (not hardcoded)")
    print("✓ FIX 5: Baseline constant (dynamic prices tracked separately)")
    print("✓ FIX 6: Soft confidence weighting + reward decomposition")
    print("\n" + "=" * 60 + "\n")
    
    unified = build_real_data_pipeline()
    
    print("\n" + "=" * 60)
    print("✓ DATA REBUILD COMPLETE")
    print("=" * 60)
    print(f"\nRows: {len(unified)}")
    print(f"Utilization range: {unified['urban_mean_utilization'].min():.2%} - {unified['urban_mean_utilization'].max():.2%}")
    print(f"Peak hours: {sorted(unified[unified['is_peak_hour']==1]['hour_of_day'].unique())}")
    print("\nNext step: Run python run_agentic.py")
