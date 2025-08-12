#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDCD Conformal k-NN anomaly detector for univariate streams
-----------------------------------------------------------
Implements the "Lazy Drifting Conformal Detector (LDCD)" from
Ishimtsev et al. (2017), using k-NN average distance as the NCM.

Usage (examples):
  python ldcd_knn.py --csv my_series.csv --col value -l 19 -k 27 -n 600 -m 600 --metric mahalanobis --prune --threshold 0.99 --out results.csv
  python ldcd_knn.py --csv my_series.csv --col value -l 1  -k 1  -n 300 -m 300 --metric euclidean  --threshold 0.99 --plot

Notes:
- Embedding window length l creates vectors of l consecutive points.
- Training size n and calibration size m slide over embedded time.
- p-value at time t uses the last m scores plus the current one (LDCD).
- Anomaly probability reported is 1 - p-value.

Author: (you)
"""




# ---------------------------- Core math ---------------------------- #













# ---------------------------- CLI helpers ---------------------------- #

def load_series(csv_path: str, col: Optional[str]) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
    if pd is None:
        # Fallback to numpy if pandas not available; expects single-column numeric file
        series = np.loadtxt(csv_path, delimiter=",")
        if series.ndim > 1:
            series = series[:, 0]
        return series.astype(float), None
    df = pd.read_csv(csv_path)
    if col is None:
        # Heuristic: prefer a column named 'value', else first numeric
        if 'value' in df.columns:
            col = 'value'
        else:
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) == 0:
                raise ValueError("No numeric column found; specify --col")
            col = num_cols[0]
    x = df[col].to_numpy(dtype=float)
    return x, df


def main():
    p = argparse.ArgumentParser(description="LDCD Conformal k-NN anomaly detector for univariate data")
    p.add_argument("--csv", type=str, help="Path to input CSV with the series", required=True)
    p.add_argument("--col", type=str, help="Column name for the series (defaults to 'value' or first numeric)")
    p.add_argument("-l", "--embed", type=int, default=19, help="Embedding length l")
    p.add_argument("-k", "--neighbors", type=int, default=27, help="k nearest neighbors")
    p.add_argument("-n", "--train", type=int, default=600, help="Training window size n (embedded)")
    p.add_argument("-m", "--calib", type=int, default=600, help="Calibration window size m (scores)")
    p.add_argument("--metric", choices=["mahalanobis", "euclidean"], default="mahalanobis",
                   help="Distance metric for k-NN")
    p.add_argument("--reg", type=float, default=1e-6, help="Covariance regularization for Mahalanobis")
    p.add_argument("--prune", action="store_true", help="Enable alarm-thinning (n/5 cooldown at >99.5%)")
    p.add_argument("--prune_q", type=float, default=0.995, help="Prune quantile threshold (default 0.995)")
    p.add_argument("--cooldown_frac", type=float, default=0.2, help="Cooldown length as fraction of n (default 0.2)")
    p.add_argument("--threshold", type=float, default=0.99, help="Anomaly probability threshold to flag detections")
    p.add_argument("--out", type=str, help="Path to write CSV with results")
    p.add_argument("--plot", action="store_true", help="Plot series and anomaly prob (requires matplotlib)")
    args = p.parse_args()

    # Load series
    x, df = load_series(args.csv, args.col)

    cfg = LDCDConfig(
        l=args.embed, k=args.neighbors, n=args.train, m=args.calib,
        metric=args.metric, reg=args.reg,
        prune=args.prune, prune_quantile=args.prune_q, cooldown_frac=args.cooldown_frac
    )

    res = ldcd_conformal_knn(x, cfg)

    # Prepare output table (if pandas available)
    if pd is not None:
        out = pd.DataFrame({
            "t": np.arange(len(x)),
            "value": x,
            "anomaly_prob": res["anomaly_prob"],
            "p_value": res["p_value"]
        })
        if args.out:
            out.to_csv(args.out, index=False)
            print(f"Wrote results to: {args.out}")
        else:
            # Print a short preview
            print(out.head(10).to_string(index=False))
            print("...")
            print(out.tail(10).to_string(index=False))
    else:
        # Minimal stdout output
        print("First valid index (raw):", res["valid_from"])
        print("Example anomaly_prob (last 10):", res["anomaly_prob"][-10:])

    # Simple detection report
    thr = args.threshold
    idx = np.where((~np.isnan(res["anomaly_prob"])) & (res["anomaly_prob"] >= thr))[0]
    print(f"Detections at threshold {thr}: {len(idx)} hits")
    if len(idx) > 0:
        print("First few detections (raw indices):", idx[:20].tolist())

    # Optional plot
    if args.plot:
        if not HAS_PLOT:
            print("matplotlib not available; cannot plot.", file=sys.stderr)
        else:
            t = np.arange(len(x))
            fig, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(t, x, label="value")
            ax1.set_xlabel("t")
            ax1.set_ylabel("value")
            ax2 = ax1.twinx()
            ax2.plot(t, res["anomaly_prob"], label="anomaly_prob (1-p)", alpha=0.8)
            ax2.axhline(thr, linestyle="--", alpha=0.5)
            ax2.set_ylabel("anomaly probability")
            fig.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
