import math
import numpy as np
import pandas as pd

# ─── PITCH CLASSIFICATION PIPELINE (from pitcher_sheets.py) ──────────────────
# Full pitch names used throughout (no short codes until display layer)

RHP_AVERAGES = {
    "Fastball":  {"velo": 90.6, "ivb": 16.3, "hb":  7.7, "spin": 2232},
    "Sinker":    {"velo": 90.2, "ivb":  7.6,  "hb": 15.7, "spin": 2155},
    "Cutter":    {"velo": 87.7, "ivb":  8.3,  "hb":  0.0, "spin": 2315},
    "Slider":    {"velo": 81.9, "ivb":  1.4,  "hb": -4.5, "spin": 2362},
    "Curveball": {"velo": 77.2, "ivb": -10.4, "hb": -9.4, "spin": 2333},
    "ChangeUp":  {"velo": 83.1, "ivb":  6.4,  "hb": 14.6, "spin": 1843},
    "Splitter":  {"velo": 81.2, "ivb":  2.6,  "hb": 11.7, "spin": 1016},
}
LHP_AVERAGES = {
    "Fastball":  {"velo": 89.2, "ivb": 16.3, "hb":  8.1, "spin": 2232},
    "Sinker":    {"velo": 89.1, "ivb":  7.4,  "hb": 15.9, "spin": 2155},
    "Cutter":    {"velo": 86.7, "ivb":  8.0,  "hb": -0.1, "spin": 2315},
    "Slider":    {"velo": 80.2, "ivb":  1.3,  "hb": -4.6, "spin": 2362},
    "Curveball": {"velo": 75.4, "ivb": -9.9,  "hb": -8.3, "spin": 2333},
    "ChangeUp":  {"velo": 82.4, "ivb":  6.8,  "hb": 14.4, "spin": 1843},
    "Splitter":  {"velo": 83.6, "ivb":  2.6,  "hb": 11.7, "spin": 1016},
}

# Map raw Trackman tags to canonical full names
RAW_TAG_MAP = {
    "FourSeamFastBall": "Fastball", "TwoSeamFastBall": "Sinker",
    "OneSeamFastBall": "Fastball",
    "Four-Seam": "Fastball", "4-Seam": "Fastball",
    "FB": "Fastball", "Two-Seam": "Sinker",
    "Curve Ball": "Curveball", "CurveBall": "Curveball", "CB": "Curveball",
    "KnuckleCurve": "Curveball",
    "SL": "Slider", "Sweeper": "Slider",
    "CH": "ChangeUp", "Changeup": "ChangeUp",
    "CT": "Cutter",
    "SP": "Splitter",
    "Undefined": "Unknown",
    "Other":     "Unknown",
    "undefined": "Unknown",
    "other":     "Unknown",
}


def relabel_split_fastballs(pitcher_df):
    """Step 1: Normalize raw Trackman tags."""
    df = pitcher_df.copy()
    df["TaggedPitchType"] = df["TaggedPitchType"].astype(str).str.strip()
    df["TaggedPitchType"] = df["TaggedPitchType"].replace(RAW_TAG_MAP)

    pitcher_hand = df["PitcherThrows"].iloc[0]
    total_pitches = len(df)
    min_pitches = math.ceil(total_pitches * 0.05)
    counts = df["TaggedPitchType"].value_counts()
    valid = counts[(counts >= min_pitches) & (~counts.index.isin(["Unknown"]))].index.tolist()

    pitch_centers = (
        df[df["TaggedPitchType"].isin(valid)]
        .groupby("TaggedPitchType")[["HorzBreak", "InducedVertBreak"]].mean()
    )
    velo_ranges = (
        df[df["TaggedPitchType"].isin(valid)]
        .groupby("TaggedPitchType")["RelSpeed"].quantile([0.25, 0.75]).unstack()
    )

    fb_hb = 8.9 if pitcher_hand == "Right" else -8.9
    pitch_centers.loc["Fastball"] = {"HorzBreak": fb_hb, "InducedVertBreak": 17.9}
    velo_ranges.loc["Fastball"] = {0.25: 88, 0.75: 93}
    if "Fastball" not in valid:
        valid.append("Fastball")

    def classify_row(row):
        pitch = row["TaggedPitchType"]
        if pitch not in velo_ranges.index:
            return pitch
        if pitch not in ["Fastball", "FourSeamFastBall", "TwoSeamFastBall", "Unknown"]:
            return pitch
        if pd.isna(row["HorzBreak"]) or pd.isna(row["InducedVertBreak"]):
            return pitch
        hb, ivb, velo = row["HorzBreak"], row["InducedVertBreak"], row["RelSpeed"]
        best_pitch, best_dist = "Fastball", float("inf")
        for p in valid:
            v25 = velo_ranges.loc[p, 0.25]
            v75 = velo_ranges.loc[p, 0.75]
            penalty = 0
            if v25 != v75:
                if velo < v25:
                    penalty = ((v25 - velo) / (v75 - v25)) * 0.5
                elif velo > v75:
                    penalty = ((velo - v75) / (v75 - v25)) * 0.5
            center = pitch_centers.loc[p]
            dist = np.sqrt((hb - center["HorzBreak"])**2 +
                           (ivb - center["InducedVertBreak"])**2) + penalty
            if dist < best_dist:
                best_dist = dist
                best_pitch = p
        return best_pitch

    df["TaggedPitchType"] = df.apply(classify_row, axis=1)
    df.loc[df["TaggedPitchType"] == "Unknown", "TaggedPitchType"] = "Fastball"
    return df


def resolve_similar_groups(df, pitcher_hand):
    """Step 2: Merge pitch clusters that are too similar to be distinct."""
    averages = RHP_AVERAGES if pitcher_hand == "Right" else LHP_AVERAGES
    df = df.copy()
    df["NormHB"] = df["HorzBreak"].apply(lambda hb: hb if pitcher_hand == "Right" else -hb)
    NEVER_MERGE = {frozenset(["Slider","Curveball"]), frozenset(["Cutter","Fastball"])}

    pitch_centers = (
        df.groupby("TaggedPitchType")[["NormHB","InducedVertBreak","RelSpeed","SpinRate"]]
        .mean().drop(index="Unknown", errors="ignore").dropna()
    )
    # Force obvious curveballs first
    for pt in list(pitch_centers.index):
        if pitch_centers.loc[pt, "InducedVertBreak"] < -8:
            df.loc[df["TaggedPitchType"] == pt, "TaggedPitchType"] = "Curveball"
    pitch_centers = (
        df.groupby("TaggedPitchType")[["NormHB","InducedVertBreak","RelSpeed","SpinRate"]]
        .mean().drop(index="Unknown", errors="ignore").dropna()
    )

    pitch_types = list(pitch_centers.index)
    for i in range(len(pitch_types)):
        for j in range(i+1, len(pitch_types)):
            p1, p2 = pitch_types[i], pitch_types[j]
            if frozenset([p1, p2]) in NEVER_MERGE:
                continue
            if p1 not in df["TaggedPitchType"].values or p2 not in df["TaggedPitchType"].values:
                continue
            c1, c2 = pitch_centers.loc[p1], pitch_centers.loc[p2]
            dist = np.sqrt(
                ((c1["NormHB"]-c2["NormHB"])/2.5)**2 +
                ((c1["InducedVertBreak"]-c2["InducedVertBreak"])/2.5)**2 +
                ((c1["RelSpeed"]-c2["RelSpeed"])/3.5)**2
            )
            if dist > 2:
                continue
            combined = df[df["TaggedPitchType"].isin([p1,p2])][["NormHB","InducedVertBreak","RelSpeed"]].mean()
            best_pitch, best_dist = None, float("inf")
            for name, avg in averages.items():
                d = np.sqrt(
                    ((combined["NormHB"]-avg["hb"])/2.5)**2 +
                    ((combined["InducedVertBreak"]-avg["ivb"])/2.5)**2 +
                    ((combined["RelSpeed"]-avg["velo"])/3.5)**2
                )
                if d < best_dist:
                    best_dist = d; best_pitch = name
            df.loc[df["TaggedPitchType"].isin([p1,p2]), "TaggedPitchType"] = best_pitch
            pitch_centers = (
                df.groupby("TaggedPitchType")[["NormHB","InducedVertBreak","RelSpeed","SpinRate"]]
                .mean().drop(index="Unknown", errors="ignore").dropna()
            )
    df = df.drop(columns=["NormHB"])
    return df


def fix_pitch_outliers(pitcher_df):
    """Step 3/6: Reassign outlier pitches to nearest cluster center."""
    df = pitcher_df.copy()
    pitcher_hand = df["PitcherThrows"].iloc[0]
    df["NormHB"] = df["HorzBreak"].apply(lambda hb: hb if pitcher_hand == "Right" else -hb)
    pitch_centers = df.groupby("TaggedPitchType")[["NormHB","InducedVertBreak","RelSpeed"]].mean()
    df["c_hb"]   = df["TaggedPitchType"].map(pitch_centers["NormHB"])
    df["c_ivb"]  = df["TaggedPitchType"].map(pitch_centers["InducedVertBreak"])
    df["c_velo"] = df["TaggedPitchType"].map(pitch_centers["RelSpeed"])
    df["dist"]   = np.sqrt(
        ((df["NormHB"]-df["c_hb"])/2.5)**2 +
        ((df["InducedVertBreak"]-df["c_ivb"])/2.5)**2 +
        ((df["RelSpeed"]-df["c_velo"])/2.3)**2
    )
    thresholds = df.groupby("TaggedPitchType")["dist"].quantile(0.90)
    df["thresh"] = df["TaggedPitchType"].map(thresholds)
    for idx in df[df["dist"] > df["thresh"]].index:
        row = df.loc[idx]
        best_pitch, best_dist = row["TaggedPitchType"], float("inf")
        for pitch in pitch_centers.index:
            c = pitch_centers.loc[pitch]
            d = np.sqrt(
                ((row["NormHB"]-c["NormHB"])/2.5)**2 +
                ((row["InducedVertBreak"]-c["InducedVertBreak"])/2.5)**2 +
                ((row["RelSpeed"]-c["RelSpeed"])/2.3)**2
            )
            if d < best_dist:
                best_dist = d; best_pitch = pitch
        df.loc[idx, "TaggedPitchType"] = best_pitch
    df = df.drop(columns=["NormHB","c_hb","c_ivb","c_velo","dist","thresh"])
    return df


def recategorize_rare_pitches(pitcher_df):
    """Step 4/7: Reassign rare pitch types against major clusters + league average gate."""
    df = pitcher_df.copy()
    pitcher_hand = df["PitcherThrows"].iloc[0]
    league_averages = RHP_AVERAGES if pitcher_hand == "Right" else LHP_AVERAGES
    df["NormHB"] = df["HorzBreak"].apply(lambda hb: hb if pitcher_hand == "Right" else -hb)
    counts = df["TaggedPitchType"].value_counts()
    counts = counts[counts.index != "Unknown"]
    total = len(df)
    rare_thresh = max(5, math.ceil(total * 0.03))
    rare_pitches = counts[counts < rare_thresh].index
    major_pitches = counts[counts > 5].index
    if len(major_pitches) == 0:
        major_pitches = counts.index
        rare_pitches = pd.Index([])

    pitcher_centers = (
        df[df["TaggedPitchType"].isin(major_pitches)]
        .groupby("TaggedPitchType")[["NormHB","InducedVertBreak","RelSpeed"]].mean()
    )

    def reclassify(row):
        if row["TaggedPitchType"] not in rare_pitches:
            return row["TaggedPitchType"]
        if pd.isna(row["NormHB"]) or pd.isna(row["InducedVertBreak"]):
            return row["TaggedPitchType"]
        hb, ivb, velo = row["NormHB"], row["InducedVertBreak"], row["RelSpeed"]
        best_pitch, best_dist = row["TaggedPitchType"], float("inf")
        for pitch in major_pitches:
            if pitch == "Curveball" and ivb > -4: continue
            if pitch == "Slider" and ivb < -9: continue
            if pitch not in pitcher_centers.index: continue
            if pitch == row["TaggedPitchType"]: continue
            if pitch in league_averages:
                lg = league_averages[pitch]
                lg_d = np.sqrt(
                    ((hb-lg["hb"])/2.5)**2 + ((ivb-lg["ivb"])/2.5)**2 + ((velo-lg["velo"])/3.5)**2
                )
                if lg_d > 4.0: continue
            c = pitcher_centers.loc[pitch]
            d = np.sqrt(
                ((hb-c["NormHB"])/2.5)**2 + ((ivb-c["InducedVertBreak"])/2.5)**2 + ((velo-c["RelSpeed"])/3.5)**2
            )
            if d < best_dist:
                best_dist = d; best_pitch = pitch
        return best_pitch

    df["TaggedPitchType"] = df.apply(reclassify, axis=1)
    df = df.drop(columns=["NormHB"])
    return df


def compare_and_reassign(df, pitcher_hand):
    """Step 5: Main classifier — pitcher model then league fallback."""
    league_averages = RHP_AVERAGES if pitcher_hand == "Right" else LHP_AVERAGES
    norm = lambda hb: hb if pitcher_hand == "Right" else -hb
    df = df.copy()
    df["NormHB"] = df["HorzBreak"].apply(norm)
    pitcher_avgs = (
        df.groupby("TaggedPitchType")
        .agg({"NormHB":"mean","InducedVertBreak":"mean","RelSpeed":"mean","SpinRate":"mean"})
        .to_dict("index")
    )
    pitch_counts = df["TaggedPitchType"].value_counts()
    allowed = {p for p, c in pitch_counts.items() if c >= 5}
    GLOVE = {"Cutter","Slider","Curveball"}

    def cdist(hb, ivb, velo, spin, avg, pname):
        hs = 2.0 if pname in GLOVE else 2.5
        vs = 4.0 if pname in GLOVE else 2.3
        return np.sqrt(((hb-avg["NormHB"])/hs)**2 + ((ivb-avg["InducedVertBreak"])/2.5)**2 +
                       ((velo-avg["RelSpeed"])/vs)**2 + ((spin-avg["SpinRate"])/200)**2)

    def ldist(hb, ivb, velo, spin, avg, pname):
        hs = 2.0 if pname in GLOVE else 2.5
        is_ = 1.8 if pname in GLOVE else 2.5
        vs = 4.0 if pname in GLOVE else 2.3
        return np.sqrt(((hb-avg["hb"])/hs)**2 + ((ivb-avg["ivb"])/is_)**2 +
                       ((velo-avg["velo"])/vs)**2 + ((spin-avg["spin"])/200)**2)

    def reclassify(row):
        if any(pd.isna(row.get(c)) for c in ["HorzBreak","InducedVertBreak","RelSpeed","SpinRate"]):
            return row["TaggedPitchType"]
        cur = row["TaggedPitchType"]
        if cur == "Unknown": cur = "Fastball"
        if cur not in pitcher_avgs: return cur
        hb = norm(row["HorzBreak"]); ivb = row["InducedVertBreak"]
        velo = row["RelSpeed"]; spin = row["SpinRate"]
        cur_d = cdist(hb, ivb, velo, spin, pitcher_avgs[cur], cur)
        best, best_d, second = cur, cur_d, float("inf")
        for pname, avg in pitcher_avgs.items():
            if pname == cur or pname not in allowed: continue
            d = cdist(hb, ivb, velo, spin, avg, pname)
            if d < best_d: second = best_d; best_d = d; best = pname
            elif d < second: second = d
        if best != cur and (cur_d-best_d) > 0.8 and best_d < 3.0 and (best_d+0.5) < second:
            return best
        if cur not in league_averages: return cur
        cur_ld = ldist(hb, ivb, velo, spin, league_averages[cur], cur)
        best, best_ld = cur, cur_ld
        for pname, avg in league_averages.items():
            if pname == cur or pname not in allowed: continue
            d = ldist(hb, ivb, velo, spin, avg, pname)
            if d < best_ld: best_ld = d; best = pname
        if best != cur and (cur_ld-best_ld) > 1.2 and best_ld < 3.5:
            return best
        return cur

    df["TaggedPitchType"] = df.apply(reclassify, axis=1)

    def force_unknown(row):
        if row["TaggedPitchType"] != "Unknown": return row["TaggedPitchType"]
        if pd.isna(row["HorzBreak"]) or pd.isna(row["InducedVertBreak"]): return "Fastball"
        hb = norm(row["HorzBreak"]); ivb = row["InducedVertBreak"]
        velo = row["RelSpeed"] if not pd.isna(row.get("RelSpeed")) else 85
        best, best_d = None, float("inf")
        for pname, avg in league_averages.items():
            d = np.sqrt(((hb-avg["hb"])/2.5)**2+((ivb-avg["ivb"])/2.5)**2+((velo-avg["velo"])/3.5)**2)
            if d < best_d: best_d = d; best = pname
        return best
    df["TaggedPitchType"] = df.apply(force_unknown, axis=1)
    df = df.drop(columns=["NormHB"])
    return df


def fix_curveballs(df, pitcher_hand):
    """Step 8: Reassign curveballs with IVB > -3 (not real curves)."""
    curveballs = df[df["TaggedPitchType"] == "Curveball"]
    if curveballs.empty: return df
    if curveballs["InducedVertBreak"].mean() > -3:
        averages = {k: v for k,v in (RHP_AVERAGES if pitcher_hand=="Right" else LHP_AVERAGES).items() if k != "Curveball"}
        def reassign(row):
            if row["TaggedPitchType"] != "Curveball": return row["TaggedPitchType"]
            hb = row["HorzBreak"] if pitcher_hand=="Right" else -row["HorzBreak"]
            ivb, velo, spin = row["InducedVertBreak"], row["RelSpeed"], row.get("SpinRate", 2000)
            best, best_d = None, float("inf")
            for pn, avg in averages.items():
                d = np.sqrt(((hb-avg["hb"])/2.5)**2+((ivb-avg["ivb"])/2.5)**2+
                            ((velo-avg["velo"])/2.3)**2+((spin-avg["spin"])/200)**2)
                if d < best_d: best_d = d; best = pn
            return best or row["TaggedPitchType"]
        df["TaggedPitchType"] = df.apply(reassign, axis=1)
    return df


def fix_curveball_outliers(df, pitcher_hand):
    """Step 9: Fix individual curveball pitches that belong to other clusters."""
    df = df.copy()
    df["NormHB"] = df["HorzBreak"].apply(lambda hb: hb if pitcher_hand=="Right" else -hb)
    curveballs = df[df["TaggedPitchType"] == "Curveball"]
    if curveballs.empty:
        df = df.drop(columns=["NormHB"]); return df
    pitch_centers = df.groupby("TaggedPitchType")[["NormHB","InducedVertBreak","RelSpeed"]].mean()
    curve_center = pitch_centers.loc["Curveball"]
    curve_spread = np.sqrt(
        ((curveballs["NormHB"]-curve_center["NormHB"])/2.5)**2 +
        ((curveballs["InducedVertBreak"]-curve_center["InducedVertBreak"])/2.5)**2 +
        ((curveballs["RelSpeed"]-curve_center["RelSpeed"])/2.3)**2
    )
    thresh = curve_spread.median() * 2

    def check_cb(row):
        if row["TaggedPitchType"] != "Curveball": return row["TaggedPitchType"]
        hb, ivb, velo = row["NormHB"], row["InducedVertBreak"], row["RelSpeed"]
        d_curve = np.sqrt(((hb-curve_center["NormHB"])/2.5)**2+((ivb-curve_center["InducedVertBreak"])/2.5)**2+((velo-curve_center["RelSpeed"])/2.3)**2)
        if d_curve <= thresh: return "Curveball"
        best, best_d = "Curveball", d_curve
        for pitch in pitch_centers.index:
            if pitch == "Curveball": continue
            c = pitch_centers.loc[pitch]
            d = np.sqrt(((hb-c["NormHB"])/2.5)**2+((ivb-c["InducedVertBreak"])/2.5)**2+((velo-c["RelSpeed"])/2.3)**2)
            if d < best_d: best_d = d; best = pitch
        return best
    df["TaggedPitchType"] = df.apply(check_cb, axis=1)
    df = df.drop(columns=["NormHB"])
    return df


def run_pitch_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full 9-step pitch classification pipeline per pitcher, matching pitcher_sheets.py."""
    import math
    # Ensure SpinRate exists (needed by compare_and_reassign)
    if "SpinRate" not in df.columns:
        df["SpinRate"] = np.nan

    # Replace Undefined with Unknown
    df["TaggedPitchType"] = df["TaggedPitchType"].replace({"Undefined": "Unknown"})

    result_parts = []
    for pitcher_name in df["Pitcher"].unique():
        mask = df["Pitcher"] == pitcher_name
        pdf = df[mask].copy()
        if pdf.empty:
            result_parts.append(pdf)
            continue

        pitcher_hand = pdf["PitcherThrows"].iloc[0] if "PitcherThrows" in pdf.columns else "Right"

        # Step 1: normalize raw tags
        tags = pdf["TaggedPitchType"].unique()
        if any(t in tags for t in ["FourSeamFastBall","TwoSeamFastBall","OneSeamFastBall","Unknown"]) \
           or any(t in RAW_TAG_MAP for t in tags):
            pdf = relabel_split_fastballs(pdf)

        # Step 2: merge similar clusters
        pdf = resolve_similar_groups(pdf, pitcher_hand)

        # Step 3: first outlier pass
        pdf = fix_pitch_outliers(pdf)

        # Step 4: reclassify rare pitches
        if len(pdf) > 19:
            pdf = recategorize_rare_pitches(pdf)

        # Step 5: main classifier
        pdf = compare_and_reassign(pdf, pitcher_hand)

        # Step 6: second outlier pass
        pdf = fix_pitch_outliers(pdf)

        # Step 7: final rare cleanup
        if len(pdf) > 19:
            pdf = recategorize_rare_pitches(pdf)

        # Step 8: curveball IVB sanity
        pdf = fix_curveballs(pdf, pitcher_hand)

        # Step 9: curveball outliers
        pdf = fix_curveball_outliers(pdf, pitcher_hand)

        result_parts.append(pdf)

    if not result_parts:
        return df
    return pd.concat(result_parts, ignore_index=True)
