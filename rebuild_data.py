"""Build the unified analytical base from raw ACN and UrbanEV datasets."""
import sys
import logging
from pathlib import Path
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.real_data_pipeline import build_real_data_pipeline


def convert_excel_to_json():
    """Convert ACN sessions Excel to JSON if not already done."""
    excel_file = "data/raw/acndata_sessions.json.xlsx"
    json_file  = "data/raw/acndata_sessions.json"

    if Path(json_file).exists():
        print(f"✓ {json_file} already exists")
        return

    print(f"Converting {excel_file} → {json_file}")
    df = pd.read_excel(excel_file)
    df_sessions = df[df["sessionID"].notna()].copy()
    print(f"  {len(df_sessions)} sessions found")

    sessions_list = []
    for _, row in df_sessions.iterrows():
        def to_iso(val):
            if val is None or (hasattr(val, "__class__") and val.__class__.__name__ == "float"):
                return None
            if isinstance(val, str):
                try:
                    return pd.to_datetime(val, utc=True).isoformat()
                except Exception:
                    return None
            return val.isoformat() if hasattr(val, "isoformat") else None

        sessions_list.append({
            "sessionID":      str(row["sessionID"]) if pd.notna(row["sessionID"]) else None,
            "stationID":      str(row["stationID"]) if pd.notna(row["stationID"]) else "unknown",
            "connectionTime": to_iso(row["connectionTime"] if pd.notna(row["connectionTime"]) else None),
            "disconnectTime": to_iso(row["disconnectTime"] if pd.notna(row["disconnectTime"]) else None),
            "kWhDelivered":   float(row["kWhDelivered"]) if pd.notna(row["kWhDelivered"]) else 0.0,
        })

    with open(json_file, "w") as f:
        json.dump(sessions_list, f, indent=2)

    print(f"✓ Saved {len(sessions_list)} sessions to {json_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    convert_excel_to_json()

    unified = build_real_data_pipeline()

    acn_peak = sorted(unified[unified["is_peak_hour"] == 1]["hour_of_day"].unique())
    util_by_hour = unified.groupby("hour_of_day")["urban_mean_utilization"].mean()
    urbanev_peak = sorted(util_by_hour[util_by_hour >= util_by_hour.quantile(0.75)].index)

    print(f"\nRows: {len(unified)}")
    print(f"Utilization range: {unified['urban_mean_utilization'].min():.2%} – {unified['urban_mean_utilization'].max():.2%}")
    print(f"ACN peak hours:    {acn_peak}")
    print(f"UrbanEV peak hours (>P75): {urbanev_peak}")
    print("\nNext: python run_eda.py")
