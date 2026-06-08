"""
Fetch additional ERA5 upper-air timesteps from the public ARCO-ERA5 zarr
(Google Cloud, anonymous) and write them as upper_YYYYMMDD.nc files matching
the format already used in this repo:

  dims:  (time, level=13, latitude=721 [90..-90], longitude=1440 [0..359.75])
  level ascending: 50,100,150,200,250,300,400,500,600,700,850,925,1000
  vars:  z, q, t, u, v   (geopotential, specific_humidity, temperature, u, v)

Surface for July 2019 is already present (surface_201907.nc), so we only extend
the upper-air data which originally shipped with just 2019-07-01/02.
"""
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr

ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
RENAME = {
    "geopotential": "z",
    "specific_humidity": "q",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
}
OUT_DIR = "/opt/dlami/nvme/upper"

# Days/hours to fetch. 07-01/02 already have all 24 hours.
# Enrich training days 07-03..07-05 with 06/12/18 (00 already present);
# keep 07-06..07-10 at 00:00 for held-out test / rollout.
DATES = ["20190703", "20190704", "20190705", "20190706",
         "20190707", "20190708", "20190709", "20190710"]
HOURS_BY_DATE = {
    "20190703": ["00", "06", "12", "18"],
    "20190704": ["00", "06", "12", "18"],
    "20190705": ["00", "06", "12", "18"],
    "20190706": ["00"],
    "20190707": ["00"],
    "20190708": ["00"],
    "20190709": ["00"],
    "20190710": ["00"],
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("opening ARCO-ERA5 zarr ...", flush=True)
    ds = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})
    ds = ds[list(RENAME.keys())].sel(level=LEVELS).rename(RENAME)

    for d in DATES:
        hours = HOURS_BY_DATE.get(d, ["00"])
        out = os.path.join(OUT_DIR, f"upper_{d}.nc")
        # Re-fetch if file missing OR has fewer timesteps than requested.
        if os.path.exists(out):
            try:
                existing = xr.open_dataset(out)
                if existing.time.size >= len(hours):
                    existing.close()
                    print(f"skip {out} (has {existing.time.size} steps)", flush=True)
                    continue
                existing.close()
            except Exception:
                pass
        times = [pd.to_datetime(f"{d}{h}", format="%Y%m%d%H") for h in hours]
        sub = ds.sel(time=times)
        sub = sub.sortby("level")
        if sub.latitude.values[0] < sub.latitude.values[-1]:
            sub = sub.sortby("latitude", ascending=False)
        for v in ["z", "q", "t", "u", "v"]:
            sub[v] = sub[v].astype("float32")
        print(f"writing {out} ({len(hours)} steps) ...", flush=True)
        sub.to_netcdf(out)
        print(f"  done {out}  size={os.path.getsize(out)/1e6:.0f} MB", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
