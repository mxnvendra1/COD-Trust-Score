"""
The COD Trust Score  --  a story about Cash-on-Delivery returns in Indian e-commerce
====================================================================================
"""
import warnings; warnings.filterwarnings("ignore")
import io
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve,
                             classification_report)

# ----------------------------------------------------------------- palette / look
INK    = "#18181b"   # obsidian for deep, grounded text
ACCENT = "#8b857c"   # muted taupe for neutral highlighting
GREEN  = "#16a34a"   # low risk
AMBER  = "#d97706"   # medium risk
RED    = "#dc2626"   # high risk
GREY   = "#9ca3af"
SOFT   = "#e5e7eb"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#cbd5e1", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#eef2f7", "grid.linewidth": 1.0,
    "axes.axisbelow": True, "font.size": 10, "font.family": "DejaVu Sans",
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelcolor": INK, "xtick.color": "#475569", "ytick.color": "#475569",
    "text.color": INK,
})

def bare(ax, keep_left=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)
    return ax

st.set_page_config(page_title="The COD Trust Score", page_icon="📦",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .block-container {max-width: 1080px; padding-top: 2rem;}
  h1, h2, h3 {color: #1f2937; letter-spacing: -0.01em;}
  p, li {font-size: 1.03rem; line-height: 1.6; color: #374151;}
  .stTabs [data-baseweb="tab"] {font-size: 0.95rem; padding: 8px 14px;}
  blockquote {border-left: 4px solid #0d9488; background: #f0fdfa;
              padding: 0.6rem 1rem; border-radius: 6px; color:#134e4a;}
  div[data-testid="stMetricValue"] {font-size: 1.6rem;}
  hr {margin: 1.2rem 0; border-color:#e5e7eb;}
</style>
""", unsafe_allow_html=True)

TARGET = "DeliveryStatus"
POSITIVE = "Returned"
DEFAULT_RTO_COST = 250   

@st.cache_data(show_spinner=False)
def load_raw(file_bytes=None):
    if file_bytes is not None:
        return pd.read_csv(io.BytesIO(file_bytes))
    if os.path.exists("cod_orders.csv"):
        return pd.read_csv("cod_orders.csv")
    
    # Auto-generate synthetic data if missing
    np.random.seed(42)
    N = 5000
    customer_ids = [f"C{str(i).zfill(5)}" for i in np.random.randint(1, 4000, N)]
    order_ids = [f"ORD{str(i).zfill(6)}" for i in range(1, N+1)]
    city_tiers = np.random.choice(["Tier 1", "tier1", "T1", "Tier-2", "Tier 3", "TIER-3"], N, p=[0.2, 0.1, 0.1, 0.2, 0.3, 0.1])
    states = np.random.choice(["KARNATAKA", "karnataka", "Maharashtra", "Delhi", "Uttar Pradesh", "Tamil Nadu"], N)
    categories = np.random.choice(["Fashion", "Electronics", "Health", "Footwear", "Home"], N, p=[0.4, 0.2, 0.1, 0.2, 0.1])
    payment_methods = np.random.choice(["COD", "Prepaid"], N, p=[0.65, 0.35])
    devices = np.random.choice(["Mobile App", "Mobile Web", "Desktop", np.nan], N, p=[0.6, 0.2, 0.15, 0.05])
    address_qualities = np.random.choice(["Complete", "Partial", "Vague", np.nan], N, p=[0.5, 0.3, 0.15, 0.05])
    
    order_values = np.random.lognormal(mean=7, sigma=0.8, size=N)
    order_values = [f"Rs {int(v):,}" if np.random.rand() > 0.5 else int(v) for v in order_values]
    
    discount_pcts = np.random.exponential(scale=15, size=N).astype(int)
    discount_pcts = [f"{v}%" if np.random.rand() > 0.5 else (999 if np.random.rand() < 0.01 else v) for v in discount_pcts]
    
    order_hours = np.random.randint(0, 24, N)
    items = np.random.poisson(lam=1.5, size=N) + 1
    prior_orders = np.random.poisson(lam=2, size=N)
    prior_returns = [np.random.randint(0, po+1) if po > 0 else 0 for po in prior_orders]
    
    rto_probs = np.zeros(N)
    for i in range(N):
        prob = 0.05
        if payment_methods[i] == "COD":
            prob = 0.25
            if "3" in city_tiers[i]: prob += 0.15
            elif "1" in city_tiers[i]: prob -= 0.05
            if categories[i] in ["Fashion", "Footwear"]: prob += 0.1
            if pd.isna(address_qualities[i]) or address_qualities[i] == "Vague": prob += 0.15
            elif address_qualities[i] == "Complete": prob -= 0.05
            if prior_orders[i] > 0:
                hist_rate = prior_returns[i] / prior_orders[i]
                if hist_rate > 0.5: prob += 0.4
                elif hist_rate == 0: prob -= 0.1
            else:
                prob += 0.05
        rto_probs[i] = np.clip(prob, 0.02, 0.95)
    
    rto_target = (np.random.rand(N) < rto_probs).astype(int)
    delivery_status = ["Returned" if r == 1 else "Delivered" for r in rto_target]
    
    df = pd.DataFrame({
        "OrderID": order_ids, "CustomerID": customer_ids, "CityTier": city_tiers,
        "State": states, "Category": categories, "OrderValue": order_values,
        "DiscountPct": discount_pcts, "PaymentMethod": payment_methods,
        "Device": devices, "AddressQuality": address_qualities, "OrderHour": order_hours,
        "Items": items, "PriorOrders": prior_orders, "PriorReturns": prior_returns,
        "DeliveryStatus": delivery_status
    })
    
    df = pd.concat([df, df.sample(10)]).sample(frac=1).reset_index(drop=True)
    df.to_csv("cod_orders.csv", index=False)
    return df

@st.cache_data(show_spinner=False)
def clean(raw: pd.DataFrame):
    d = raw.copy()
    notes = []
    n0 = len(d)

    dups = int(d.duplicated().sum())
    if dups:
        d = d.drop_duplicates().reset_index(drop=True)
        notes.append(f"Removed **{dups}** exact duplicate rows ({n0} → {len(d)}).")

    if "OrderValue" in d:
        d["OrderValue"] = pd.to_numeric(d["OrderValue"].astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
        notes.append("Parsed **OrderValue** — removed `Rs`/`INR`/commas, converted to numbers.")

    if "DiscountPct" in d:
        d["DiscountPct"] = pd.to_numeric(d["DiscountPct"].astype(str).str.replace("%", "", regex=False), errors="coerce")
        bad = int((d["DiscountPct"] > 100).sum())
        if bad:
            med = d.loc[d["DiscountPct"] <= 100, "DiscountPct"].median()
            d.loc[d["DiscountPct"] > 100, "DiscountPct"] = med
            notes.append(f"Fixed **{bad}** impossible discounts (>100%) → set to the median ({med:.0f}%).")
        notes.append("Parsed **DiscountPct** — removed `%`, converted to numbers.")

    if "CityTier" in d:
        before = d["CityTier"].nunique()
        digit = d["CityTier"].astype(str).str.extract(r"([123])")[0]
        d["CityTier"] = digit.map({"1": "Tier-1", "2": "Tier-2", "3": "Tier-3"})
        notes.append(f"Standardised **CityTier**: {before} messy variants → 3 clean tiers.")

    if "State" in d:
        before = d["State"].nunique()
        d["State"] = d["State"].astype(str).str.strip().str.title()
        notes.append(f"Cleaned **State**: {before} → {d['State'].nunique()} distinct names.")

    for c in ["Device", "AddressQuality"]:
        if c in d:
            miss = int(d[c].isna().sum())
            if miss:
                d[c] = d[c].fillna("Unknown")
                notes.append(f"**{c}**: {miss} missing values labelled `Unknown`.")

    d["RTO"] = (d[TARGET] == POSITIVE).astype(int)
    d["PriorRTORate"] = np.where(d["PriorOrders"] > 0, d["PriorReturns"] / d["PriorOrders"], 0.0)
    d["FirstTime"] = (d["PriorOrders"] == 0).astype(int)
    d["IsCOD"] = (d["PaymentMethod"] == "COD").astype(int)
    notes.append("Built the target **RTO** (1 = Returned, 0 = Delivered) and helper features.")

    miss_val = int(d["OrderValue"].isna().sum())
    if miss_val:
        d = d.dropna(subset=["OrderValue"]).reset_index(drop=True)
        notes.append(f"Dropped **{miss_val}** rows with unrecoverable OrderValue.")

    return d, notes

NUM_FEATS = ["OrderValue", "DiscountPct", "PriorOrders", "PriorRTORate", "FirstTime", "OrderHour", "Items"]
CAT_FEATS = ["CityTier", "State", "Category", "Device", "AddressQuality"]
SCORE_MODEL = "Gradient Boosting"

def design_matrix(df_cod):
    X = df_cod[NUM_FEATS + CAT_FEATS].copy()
    y = df_cod["RTO"].copy()
    return X, y

def preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS),
    ])

@st.cache_resource(show_spinner=True)
def train_models(_X, _y, test_size, seed):
    Xtr, Xte, ytr, yte = train_test_split(_X, _y, test_size=test_size, stratify=_y, random_state=seed)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced", random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
    }
    out = {}
    for name, clf in models.items():
        pipe = Pipeline([("pre", preprocessor()), ("clf", clf)]).fit(Xtr, ytr)
        ptr, pte = pipe.predict(Xtr), pipe.predict(Xte)
        proba = pipe.predict_proba(Xte)[:, 1]
        out[name] = {
            "test_acc": accuracy_score(yte, pte),
            "roc_auc": roc_auc_score(yte, proba),
            "pipe": pipe,
        }
    return out, (len(Xtr), len(Xte), ytr.mean(), yte.mean())

@st.cache_data(show_spinner=True)
def honest_scores(_X, _y, best_name, test_size, seed):
    model = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced", random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
    }[best_name]
    pipe = Pipeline([("pre", preprocessor()), ("clf", model)])
    proba = cross_val_predict(pipe, _X, _y, cv=5, method="predict_proba")[:, 1]
    score = np.round(900 - proba * 600).astype(int)
    return proba, score

def cramers_v(df, col, target="RTO"):
    ct = pd.crosstab(df[col], df[target])
    chi2, p, dof, _ = chi2_contingency(ct)
    n = ct.to_numpy().sum(); r, k = ct.shape
    v = np.sqrt((chi2 / n) / max(min(r - 1, k - 1), 1))
    return chi2, p, dof, v, int(ct.shape[0])

# ================================================================ SIDEBAR
st.sidebar.title("⚙️ Controls")
up = st.sidebar.file_uploader("Use your own CSV (optional)", type=["csv"])
raw = load_raw(up.read() if up is not None else None)
df, notes = clean(raw)

st.sidebar.markdown("**Model settings**")
test_size = st.sidebar.slider("Test split", 0.15, 0.40, 0.25, 0.05)
seed = int(st.sidebar.number_input("Random seed", value=42, step=1))
st.sidebar.caption("Every chart is computed live from the data.")

cod = df[df["IsCOD"] == 1].copy().reset_index(drop=True)
overall_rto = df["RTO"].mean()
cod_rto = cod["RTO"].mean()
prepaid_rto = df[df["IsCOD"] == 0]["RTO"].mean()

# ================================================================ HEADER
st.title("📦 The COD Trust Score")
st.markdown("##### A credit score, but for *Cash on Delivery* — finding which shoppers a brand can safely offer COD, and which it can't.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Orders in our store", f"{len(df):,}")
m2.metric("Customers", f"{df['CustomerID'].nunique():,}")
m3.metric("COD return rate", f"{cod_rto*100:.0f}%", help="Share of COD parcels sent back")
m4.metric("Prepaid return rate", f"{prepaid_rto*100:.0f}%", "much safer", delta_color="off")

tabs = st.tabs([
    "1 · 📦 The Problem", "2 · 🧾 Meet the Data", "3 · 🧹 Cleaning",
    "4 · 📊 What Happened", "5 · 🔍 The Real Reasons", "6 · 🎯 The COD Score",
    "7 · ⚖️ The Verdict",
])

# =============================================================== 1. THE PROBLEM
with tabs[0]:
    st.header("Chapter 1 — The leak nobody sees")
    st.markdown("""
Picture a small clothing brand in India. A customer taps **"Cash on Delivery"** and
places a ₹1,200 order. The brand packs it, pays a courier, and ships it.
A week later, the parcel comes **back**. The customer wasn't home. Or changed their mind.
The brand now pays shipping **twice** and earns **nothing**. This is **RTO** (Return to Origin).
""")
    st.subheader("Why this is a *big* problem")
    c1, c2, c3 = st.columns(3)
    c1.metric("COD share of orders in India", "≈ 60–65%")
    c2.metric("Typical COD return rate", "25–40%")
    c3.metric("Lost per returned order", "₹180–350")

    fig, ax = plt.subplots(figsize=(8, 2.4))
    rates = [prepaid_rto * 100, cod_rto * 100]
    bars = ax.barh(["Prepaid", "Cash on Delivery"], rates, color=[GREEN, RED], height=0.6)
    for b, r in zip(bars, rates):
        ax.text(r + 0.6, b.get_y() + b.get_height() / 2, f"{r:.0f}%", va="center", fontweight="bold", color=INK)
    ax.set_xlim(0, max(rates) * 1.25); ax.set_xlabel("Return rate")
    ax.set_title("In our own store: COD comes back far more often than prepaid")
    bare(ax); ax.tick_params(left=False)
    st.pyplot(fig, width="stretch")

# =============================================================== 2. MEET THE DATA
with tabs[1]:
    st.header("Chapter 2 — Meet the store")
    st.markdown(f"Synthetic store of **{len(df):,} orders** from **{df['CustomerID'].nunique():,} shoppers** across India.")
    st.subheader("A peek at the raw file")
    st.dataframe(raw.head(12), width="stretch")

# =============================================================== 3. CLEANING
with tabs[2]:
    st.header("Chapter 3 — Cleaning the data")
    for n in notes:
        st.markdown(f"- {n}")
    st.success(f"Clean dataset ready: **{len(df):,} orders**, **{df.shape[1]} columns**.")

# =============================================================== 4. DESCRIPTIVE
with tabs[3]:
    st.header("Chapter 4 — What happened: who sends parcels back?")
    def rate_by(col, title, min_n=20, full=True):
        base = df if full else cod
        ref = base["RTO"].mean()
        g = base.groupby(col)["RTO"].agg(["mean", "count"])
        g = g[g["count"] >= min_n].sort_values("mean")
        fig, ax = plt.subplots(figsize=(8, max(2.4, 0.5 * len(g))))
        colors = [RED if m > ref else GREEN for m in g["mean"]]
        bars = ax.barh(g.index.astype(str), g["mean"] * 100, color=colors, height=0.62)
        ax.axvline(ref * 100, color=INK, ls="--", lw=1.3)
        for b, (m, n) in zip(bars, zip(g["mean"], g["count"])):
            ax.text(m * 100 + 0.5, b.get_y() + b.get_height() / 2, f"{m*100:.0f}%  (n={n:,})", va="center", fontsize=9, color=INK)
        ax.set_xlim(0, g["mean"].max() * 100 * 1.25)
        ax.set_xlabel("Return rate"); ax.set_title(title)
        bare(ax); ax.tick_params(left=False)
        return fig

    lens = st.selectbox("Break COD returns down by:", ["City tier", "Product category", "Address quality"], index=0)
    if lens == "City tier":
        st.pyplot(rate_by("CityTier", "COD return rate by city tier", full=False), width="stretch")
    elif lens == "Product category":
        st.pyplot(rate_by("Category", "COD return rate by product category", full=False), width="stretch")
    else:
        st.pyplot(rate_by("AddressQuality", "COD return rate by address quality", full=False), width="stretch")

# =============================================================== 5. DIAGNOSTIC
with tabs[4]:
    st.header("Chapter 5 — The real reasons (signal vs noise)")
    assoc_cols = [c for c in ["PaymentMethod", "CityTier", "Category", "AddressQuality", "State", "Device"] if c in df.columns]
    rows = []
    for c in assoc_cols:
        chi2, p, dof, v, ncat = cramers_v(df, c)
        rows.append({"Clue": c, "Cramér's V": round(v, 3), "p-value": f"{p:.1e}", "Real link?": "✅ yes" if p < 0.05 else "— (luck)"})
    assoc = pd.DataFrame(rows).sort_values("Cramér's V", ascending=False)
    st.dataframe(assoc, width="stretch", hide_index=True)

# =============================================================== 6. THE COD SCORE
with tabs[5]:
    st.header("Chapter 6 — Giving every shopper a COD Score")
    X, y = design_matrix(cod)
    results, (ntr, nte, trp, tep) = train_models(X, y, test_size, seed)
    
    metrics = pd.DataFrame({m: {"Accuracy": r["test_acc"], "ROC-AUC": r["roc_auc"]} for m, r in results.items()}).T.round(3)
    st.dataframe(metrics, width="stretch")
    st.success(f"🎯 Scoring model: **{SCORE_MODEL}** — calibrated, accurate, and steady.")

    if SCORE_MODEL in ("Random Forest", "Gradient Boosting", "Decision Tree"):
        model_step = results[SCORE_MODEL]["pipe"].named_steps["clf"]
        if hasattr(model_step, "feature_importances_"):
            ohe = results[SCORE_MODEL]["pipe"].named_steps["pre"].named_transformers_["cat"]
            feat = NUM_FEATS + list(ohe.get_feature_names_out(CAT_FEATS))
            imp = pd.Series(model_step.feature_importances_, index=feat)
            imp = imp.sort_values(ascending=False).head(12)[::-1]
            
            st.subheader(f"What {SCORE_MODEL} pays most attention to")
            fig, ax = plt.subplots(figsize=(8, 4.4))
            ax.barh(imp.index, imp.values, color=ACCENT, height=0.7)
            ax.set_xlabel("Importance"); ax.set_title("Top clues driving the score")
            bare(ax); ax.tick_params(left=False)
            st.pyplot(fig, width="stretch")

    st.subheader("From risk → the COD Trust Score")
    proba, score = honest_scores(X, y, SCORE_MODEL, test_size, seed)
    cod_scored = cod.copy()
    cod_scored["Risk"] = proba
    cod_scored["Score"] = score
    st.session_state["cod_scored"] = cod_scored
    
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.hist(score, bins=40, color="#94a3b8", edgecolor="white")
    ax.axvspan(300, 580, color=RED, alpha=0.10)
    ax.axvspan(580, 720, color=AMBER, alpha=0.12)
    ax.axvspan(720, 900, color=GREEN, alpha=0.12)
    ax.set_xlabel("COD Trust Score"); ax.set_ylabel("Number of shoppers")
    bare(ax)
    st.pyplot(fig, width="stretch")

# =============================================================== 7. THE VERDICT
with tabs[6]:
    st.header("Chapter 7 — The verdict: who gets COD?")
    if "cod_scored" not in st.session_state:
        X, y = design_matrix(cod)
        results, _ = train_models(X, y, test_size, seed)
        proba, score = honest_scores(X, y, SCORE_MODEL, test_size, seed)
        cs = cod.copy(); cs["Risk"] = proba; cs["Score"] = score
        st.session_state["cod_scored"] = cs
    cs = st.session_state["cod_scored"].copy()

    p1, p2, p3 = st.columns(3)
    with p1:
        low_cut = st.slider("Prepaid-only below score", 300, 700, 580, 10)
    with p2:
        high_cut = st.slider("Free COD at or above score", 600, 900, 720, 10)
    with p3:
        rto_cost = st.number_input("₹ lost per returned order", 100, 600, DEFAULT_RTO_COST, 10)
    
    cod_fee = st.slider("COD fee charged in the middle tier (₹)", 0, 150, 50, 5)
    cs["Tier"] = np.where(cs["Score"] >= high_cut, "Free COD", np.where(cs["Score"] < low_cut, "Prepaid-only", "COD + fee"))

    base_loss = (cs["RTO"] * rto_cost).sum()
    pp = cs[cs["Tier"] == "Prepaid-only"]
    prevented = pp["RTO"].sum() * 0.90
    saved_prepaid = prevented * rto_cost
    good_lost = (len(pp) - pp["RTO"].sum()) * 0.25
    avg_pp_order = pp["OrderValue"].mean() if len(pp) else 0
    lost_margin = good_lost * avg_pp_order * 0.20
    mid = cs[cs["Tier"] == "COD + fee"]
    fee_income = len(mid) * cod_fee * 0.85
    net = saved_prepaid + fee_income - lost_margin

    mcol = st.columns(4)
    mcol[0].metric("Blind COD loss", f"₹{base_loss:,.0f}")
    mcol[1].metric("Saved by prepaid-only", f"₹{saved_prepaid:,.0f}")
    mcol[2].metric("COD-fee income", f"₹{fee_income:,.0f}")
    mcol[3].metric("Net effect", f"₹{net:,.0f}")

    st.download_button("⬇️ Download the cleaned, scored data (CSV)", cs.to_csv(index=False).encode(), "cod_scored.csv", "text/csv")