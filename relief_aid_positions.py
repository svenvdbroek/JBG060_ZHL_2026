import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

YEAR, LEAD = 2025, 1
K = 3                          # number of relief aid positions (depots)
K_SWEEP = range(1, 7)          # for the K sweep at the end
SWEEP_AVOID = 1.01             # exclusion for the K sweep: 0.5 = on, 1.01 = off

# AVOID_FLOODED: no depot in counties with p_flood >= this (1.01 = no restriction)
RUNS = [
    ("raw, exclusion on",  "priority",      "raw",  0.5),
    ("raw, exclusion off", "priority",      "raw",  1.01),
    ("sqrt, exclusion on", "priority",      "sqrt", 0.5),
    ("sqrt, exclusion off", "priority",     "sqrt", 1.01),
    ("ORACLE (IOM), excl on",  "displaced_iom", "raw", 0.5),
    ("ORACLE (IOM), excl off", "displaced_iom", "raw", 1.01),
]


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def dist_matrix(a, b):
    """km between every row of a and every row of b (both need lat, lon)."""
    return haversine(a["lat"].values[:, None], a["lon"].values[:, None],
                     b["lat"].values[None, :], b["lon"].values[None, :])


def solve_pmedian(D, w, k):
    """Exact p-median. D: (n demand x c candidates) km, w: demand weights.
    Returns indices of the k chosen candidates."""
    n, c = D.shape
    nv = c + n * c
    obj = np.concatenate([np.zeros(c), (w[:, None] * D).ravel()])
    A = lil_matrix((n + n * c + 1, nv))
    lo, hi = [], []
    r = 0
    for i in range(n):
        A[r, c + i * c: c + (i + 1) * c] = 1
        lo.append(1); hi.append(1); r += 1
    for i in range(n):
        for j in range(c):
            A[r, c + i * c + j] = 1
            A[r, j] = -1
            lo.append(-np.inf); hi.append(0); r += 1
    A[r, :c] = 1                            # exactly k depots
    lo.append(k); hi.append(k)
    res = milp(obj, constraints=LinearConstraint(A.tocsr(), lo, hi),
               integrality=np.concatenate([np.ones(c), np.zeros(n * c)]),
               bounds=Bounds(0, 1))
    if res.x is None:
        raise RuntimeError(f"MILP failed: {res.message}")
    return np.where(res.x[:c] > 0.5)[0]


def evaluate(depots, ev):
    d = dist_matrix(ev, depots).min(axis=1)
    return {
        "fc_km": round(float((ev["priority"] * d).sum() / ev["priority"].sum()), 1),
        "iom_km": round(float((ev["displaced_iom"] * d).sum() / ev["displaced_iom"].sum()), 1),
        "max_km": round(float(d.max()), 1),
    }


def main():
    from processing_data.loading_impact_data import load_admin_boundaries

    df = pd.read_csv("processing_data/floods/where_to_help.csv")
    df = df[(df["year"] == YEAR) & (df["lead"] == LEAD)].copy()
    df["priority"] = df["priority"].fillna(0)
    df["displaced_iom"] = df["displaced_iom"].fillna(0)

    _, admin2 = load_admin_boundaries()
    cent = admin2.to_crs(32636).centroid.to_crs(4326)
    xy = pd.DataFrame({"adm2_pcode": admin2["adm2_pcode"], "adm2_name": admin2["adm2_name"],
                       "lon": cent.x, "lat": cent.y})

    df = df.merge(xy[["adm2_pcode", "lon", "lat"]], on="adm2_pcode")
    ev = df[(df["priority"] > 0) | (df["displaced_iom"] > 0)].reset_index(drop=True)
    print(f"{YEAR}, lead {LEAD}: {len(ev)} counties in the evaluation set")
    print(f"total priority {ev['priority'].sum():.0f}, total IOM displaced {ev['displaced_iom'].sum():.0f}\n")

    def run_one(weight_col, transform, avoid, k):
        demand = df[df[weight_col] > 0].reset_index(drop=True)
        w = demand[weight_col].values.astype(float)
        if transform == "sqrt":
            w = np.sqrt(w)
        flooded = set(df.loc[df["p_flood"] >= avoid, "adm2_pcode"])
        cand = xy[~xy["adm2_pcode"].isin(flooded)].reset_index(drop=True)
        idx = solve_pmedian(dist_matrix(demand, cand), w, k)
        return cand.iloc[idx]

    rows = []
    for label, weight_col, transform, avoid in RUNS:
        depots = run_one(weight_col, transform, avoid, K)
        rows.append({"run": label, "depots": ", ".join(depots["adm2_name"]),
                     **evaluate(depots, ev)})
    print(f"K = {K}")
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\nK sweep (raw priority weights, exclusion {'off' if SWEEP_AVOID > 1 else 'on'}):")
    rows = []
    for k in K_SWEEP:
        depots = run_one("priority", "raw", SWEEP_AVOID, k)
        rows.append({"K": k, "depots": ", ".join(depots["adm2_name"]), **evaluate(depots, ev)})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()