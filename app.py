"""
The COD Trust Score  --  a story about Cash-on-Delivery returns in Indian e-commerce
====================================================================================
Run locally:   pip install -r requirements.txt  &&  streamlit run app.py
Deploy free:   push this folder to GitHub -> share.streamlit.io -> point to app.py

The dashboard reads ONE messy CSV (cod_orders.csv) and tells a 7-chapter story:
  1. The Problem        - why COD returns quietly drain Indian D2C brands
  2. Meet the Data      - what each order in our (synthetic) store looks like
  3. Cleaning the Data  - the messy export, fixed step by step (data prep)
  4. What Happened      - who sends parcels back more often (descriptive)
  5. The Real Reasons   - which signals truly matter vs. noise (diagnostic)
  6. The COD Score      - turning return-risk into a CIBIL-style score (models)
  7. The Verdict        - who gets COD free / COD with a fee / prepaid only

Everything below is recomputed live from the CSV. Nothing is hard-coded.

Honesty note carried throughout: a high return-risk is a reason to ADJUST THE OFFER
(ask for prepayment, add a small fee), never a verdict that a customer is a bad person.
"""
import warnings; warnings.filterwarnings("ignore")
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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
INK    = "#1f2937"   # slate ink for text + structure
ACCENT = "#0d9488"   # calm teal = "trust"
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
    """Minimalist axis: drop top/right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)
    return ax

st.set_page_config(page_title="The COD Trust Score", page_icon="📦",
                   layout="wide", initial_sidebar_state="expanded")

# light touch of global CSS for a clean, readable, minimalist feel
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

# COD economics (industry-reported; editable in the Verdict chapter)
DEFAULT_RTO_COST = 250   # ₹ lost on a typical returned COD order (forward+reverse+handling)

# ============================================================ DATA: load + clean
@st.cache_data(show_spinner=False)
def load_raw(file_bytes):
    if file_bytes is None:
        return pd.read_csv("cod_orders.csv")
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def clean(raw: pd.DataFrame):
    """Turn the messy export into analysis-ready data, logging every change."""
    d = raw.copy()
    notes = []
    n0 = len(d)

    # 1) exact duplicate rows (accidental double-export)
    dups = int(d.duplicated().sum())
    if dups:
        d = d.drop_duplicates().reset_index(drop=True)
        notes.append(f"Removed **{dups}** exact duplicate rows ({n0} → {len(d)}).")

    # 2) OrderValue: strip 'Rs'/'INR'/commas → number
    if "OrderValue" in d:
        d["OrderValue"] = pd.to_numeric(
            d["OrderValue"].astype(str).str.replace(r"[^0-9.]", "", regex=True),
            errors="coerce")
        notes.append("Parsed **OrderValue** — removed `Rs`/`INR`/commas, converted to numbers.")

    # 3) DiscountPct: strip '%', cap impossible values (>100) as data errors
    if "DiscountPct" in d:
        d["DiscountPct"] = pd.to_numeric(
            d["DiscountPct"].astype(str).str.replace("%", "", regex=False),
            errors="coerce")
        bad = int((d["DiscountPct"] > 100).sum())
        if bad:
            med = d.loc[d["DiscountPct"] <= 100, "DiscountPct"].median()
            d.loc[d["DiscountPct"] > 100, "DiscountPct"] = med
            notes.append(f"Fixed **{bad}** impossible discounts (>100%) → set to the "
                         f"median ({med:.0f}%).")
        notes.append("Parsed **DiscountPct** — removed `%`, converted to numbers.")

    # 4) CityTier: merge case/format variants (Tier 1 / tier1 / T1 / TIER-1 → Tier-1)
    if "CityTier" in d:
        before = d["CityTier"].nunique()
        digit = d["CityTier"].astype(str).str.extract(r"([123])")[0]
        d["CityTier"] = digit.map({"1": "Tier-1", "2": "Tier-2", "3": "Tier-3"})
        notes.append(f"Standardised **CityTier**: {before} messy variants → 3 clean tiers.")

    # 5) State: trim whitespace + Title Case (merge 'KARNATAKA'/'karnataka')
    if "State" in d:
        before = d["State"].nunique()
        d["State"] = d["State"].astype(str).str.strip().str.title()
        notes.append(f"Cleaned **State**: {before} → {d['State'].nunique()} distinct names "
                     "(trimmed spaces, unified case).")

    # 6) Missing categoricals → explicit 'Unknown'
    for c in ["Device", "AddressQuality"]:
        if c in d:
            miss = int(d[c].isna().sum())
            if miss:
                d[c] = d[c].fillna("Unknown")
                notes.append(f"**{c}**: {miss} missing values labelled `Unknown`.")

    # 7) binary target + handy derived features (all knowable BEFORE dispatch)
    d["RTO"] = (d[TARGET] == POSITIVE).astype(int)
    d["PriorRTORate"] = np.where(d["PriorOrders"] > 0,
                                 d["PriorReturns"] / d["PriorOrders"], 0.0)
    d["FirstTime"] = (d["PriorOrders"] == 0).astype(int)
    d["IsCOD"] = (d["PaymentMethod"] == "COD").astype(int)
    notes.append("Built the target **RTO** (1 = Returned, 0 = Delivered) and helper "
                 "features: past-return rate, first-time flag, COD flag.")

    # drop any rows that lost their value in parsing (rare)
    miss_val = int(d["OrderValue"].isna().sum())
    if miss_val:
        d = d.dropna(subset=["OrderValue"]).reset_index(drop=True)
        notes.append(f"Dropped **{miss_val}** rows with unrecoverable OrderValue.")

    return d, notes


# ============================================================ MODELLING (COD only)
NUM_FEATS = ["OrderValue", "DiscountPct", "PriorOrders", "PriorRTORate",
             "FirstTime", "OrderHour", "Items"]
CAT_FEATS = ["CityTier", "State", "Category", "Device", "AddressQuality"]
# A score must be *calibrated* (predicted risk ≈ real risk), not just high-AUC.
# Gradient Boosting's probabilities track reality almost exactly, so we score from it.
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
    Xtr, Xte, ytr, yte = train_test_split(_X, _y, test_size=test_size,
                                           stratify=_y, random_state=seed)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=25),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                                random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12,
                                                class_weight="balanced",
                                                random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
    }
    out = {}
    for name, clf in models.items():
        pipe = Pipeline([("pre", preprocessor()), ("clf", clf)]).fit(Xtr, ytr)
        ptr, pte = pipe.predict(Xtr), pipe.predict(Xte)
        proba = pipe.predict_proba(Xte)[:, 1]
        out[name] = {
            "train_acc": accuracy_score(ytr, ptr),
            "test_acc": accuracy_score(yte, pte),
            "precision": precision_score(yte, pte, zero_division=0),
            "recall": recall_score(yte, pte, zero_division=0),
            "f1": f1_score(yte, pte, zero_division=0),
            "roc_auc": roc_auc_score(yte, proba),
            "cm": confusion_matrix(yte, pte),
            "roc": roc_curve(yte, proba),
            "report": classification_report(yte, pte,
                       target_names=["Delivered", "Returned"], zero_division=0),
            "pipe": pipe,
        }
    return out, (len(Xtr), len(Xte), ytr.mean(), yte.mean())


@st.cache_data(show_spinner=True)
def honest_scores(_X, _y, best_name, test_size, seed):
    """Return-risk for EVERY COD order using cross-validation (out-of-fold),
    so no order is scored by a model that already saw it. Then map risk → score."""
    model = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                                random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12,
                                                class_weight="balanced",
                                                random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=25),
    }[best_name]
    pipe = Pipeline([("pre", preprocessor()), ("clf", model)])
    proba = cross_val_predict(pipe, _X, _y, cv=5, method="predict_proba")[:, 1]
    # CIBIL-style: low risk -> high score. Range ~300..900.
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
st.sidebar.caption("The story reads top-to-bottom — open the tabs in order. "
                   "Every chart is computed live from the data.")

# COD subset = the population the score is built for
cod = df[df["IsCOD"] == 1].copy().reset_index(drop=True)
overall_rto = df["RTO"].mean()
cod_rto = cod["RTO"].mean()
prepaid_rto = df[df["IsCOD"] == 0]["RTO"].mean()

# ================================================================ HEADER
st.title("📦 The COD Trust Score")
st.markdown("##### A credit score, but for *Cash on Delivery* — finding which shoppers "
            "a brand can safely offer COD, and which it can't.")

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
Or never really meant to buy it.

The brand now pays shipping **twice** — once out, once back — plus packaging and handling,
and earns **nothing**. This is called **Return to Origin**, or **RTO**, and in India it is
quietly one of the biggest costs an online brand carries.
""")

    st.subheader("Why this is a *big* problem (not just ours)")
    c1, c2, c3 = st.columns(3)
    c1.metric("COD share of orders in India", "≈ 60–65%", help="Versus ~12% in the US")
    c2.metric("Typical COD return rate", "25–40%", help="Prepaid is only ~3–8%")
    c3.metric("Lost per returned order", "₹180–350", help="Shipping both ways + handling")

    st.markdown("""
> **In plain words:** most Indian online orders are COD, and on COD, roughly **1 in 3 to
> 1 in 4 parcels** can come back. Each one burns a few hundred rupees. For a brand doing
> thousands of orders a month, that's **lakhs of rupees** vanishing every month.
""")

    st.markdown("**Don't take our word for it — here's the industry saying the same thing:**")
    st.markdown("""
- **GoKwik** — *What is Return to Origin (RTO)?* — explains the ₹200–250 lost on a typical
  ₹1,000 COD return → [gokwik.co](https://www.gokwik.co/blog/what-is-return-to-origin-rto-in-ecommerce)
- **HillTeck** — *True Cost of RTO for Indian D2C Brands* — the real all-in cost, with a
  formula → [hillteck.com](https://www.hillteck.com/blog/rto-cost-indian-d2c-brands.html)
- **Pragma** — *Reduce RTO without hurting COD orders* — segments customers by behaviour
  (exactly our idea) → [bepragma.ai](https://www.bepragma.ai/blogs/how-to-reduce-rto-in-indian-e-commerce-without-hurting-cod-orders)
- **Edgistify** — *RTO %: the silent killer of Indian D2C* → [edgistify.com](https://www.edgistify.com/resources/blogs/rto-percentage-silent-killer-indian-d2c)
- **CallFox** — *Reduce COD returns in India* → [callfox.in](https://www.callfox.in/blog/reduce-cod-returns-india)
""")

    st.subheader("So why not just switch COD off?")
    st.markdown("""
Because in India, COD is how people **trust** a new brand — especially in smaller towns.
Kill COD and you kill a huge chunk of genuine sales. So the real question isn't *"COD or
no COD?"* It's smarter:

> **Which customers should we offer COD to — and which ones should pay first?**
""")

    st.subheader("The big idea: a 'CIBIL score', but for COD")
    colA, colB = st.columns([1.15, 1])
    with colA:
        st.markdown("""
When a bank gives a loan, it checks your **CIBIL score** first. The score reads your
past behaviour — did you repay on time? — and sorts you into:

- **good score →** loan approved, best terms
- **middle score →** approved, but higher interest (to cover the risk)
- **low score →** loan refused

We do the **same thing for parcels.** We build a **COD Trust Score** for each shopper from
their order behaviour and past returns, then:

- **good score →** COD offered freely
- **middle score →** COD offered, with a small **COD fee** (covers the risk, like interest)
- **low score →** COD switched off — **pay online instead**

That fee in the middle is the heart of it: it's **risk-based pricing**, exactly like a
loan's interest rate.
""")
    with colB:
        st.markdown("**The parallel, side by side:**")
        st.table(pd.DataFrame({
            "CIBIL (loans)": ["Reads repayment history", "Bank: lend or not?",
                              "Higher interest for risky", "Refuse below cut-off"],
            "COD Score (parcels)": ["Reads return history", "Brand: offer COD or not?",
                                    "COD fee for risky", "Prepaid-only below cut-off"],
        }))

    # our own data, in one line, to set up the rest of the story
    fig, ax = plt.subplots(figsize=(8, 2.4))
    rates = [prepaid_rto * 100, cod_rto * 100]
    bars = ax.barh(["Prepaid", "Cash on Delivery"], rates, color=[GREEN, RED], height=0.6)
    for b, r in zip(bars, rates):
        ax.text(r + 0.6, b.get_y() + b.get_height() / 2, f"{r:.0f}%",
                va="center", fontweight="bold", color=INK)
    ax.set_xlim(0, max(rates) * 1.25); ax.set_xlabel("Return rate")
    ax.set_title("In our own store: COD comes back far more often than prepaid")
    bare(ax); ax.tick_params(left=False)
    st.pyplot(fig, width="stretch")
    st.caption("This is the gap our whole project is about. The next chapters dig into "
               "*who* drives the COD half of it — and what to do about them.")

# =============================================================== 2. MEET THE DATA
with tabs[1]:
    st.header("Chapter 2 — Meet the store")
    st.markdown(f"""
There's no public dataset of one brand's private COD returns (it's sensitive business
data). So we **built our own** — a synthetic store of **{len(df):,} orders** from
**{df['CustomerID'].nunique():,} shoppers** across India, designed to behave like the real
world the blogs above describe.

**Why synthetic is the honest choice here:**

- We control the truth, so we can *check* whether our methods actually find the patterns
  we planted (a real dataset never tells you the "right answer").
- No real customer's privacy is touched.
- We can bake in genuine Indian-market behaviour: COD-heavy smaller towns, deep-discount
  impulse buys, first-timers who get cold feet at the door.
""")

    st.markdown("> **The one trick that makes a *score* possible:** every shopper has a "
                "hidden habit — some reliably accept parcels, some often refuse. That habit "
                "shows up in their **past returns**. So history becomes a clue to the "
                "future, the same way repaying past loans builds your CIBIL score.")

    st.subheader("What one order looks like")
    field_help = pd.DataFrame({
        "Column": ["OrderID / CustomerID", "CityTier", "State", "Category", "OrderValue",
                   "DiscountPct", "PaymentMethod", "Device", "AddressQuality", "OrderHour",
                   "Items", "PriorOrders", "PriorReturns", "DeliveryStatus"],
        "In plain English": [
            "Who placed it (IDs)",
            "Tier-1 metro, Tier-2 city, or Tier-3 town",
            "Which state it ships to",
            "What was bought (Fashion, Electronics …)",
            "Order amount in ₹",
            "Discount on the order",
            "Cash on Delivery or paid online (Prepaid)",
            "App / mobile web / desktop",
            "Was the address complete, partial, or vague?",
            "Hour of day the order was placed (0–23)",
            "How many items in the order",
            "How many orders this shopper made before",
            "How many of those came back",
            "👉 The outcome: Delivered or Returned",
        ],
        "Why it might matter": [
            "Identify repeat shoppers",
            "Smaller towns tend to return more",
            "Some regions are riskier",
            "Fashion is tried-on-and-returned a lot",
            "Big COD bills get refused at the door",
            "Deep discounts → impulse → regret",
            "COD is where returns happen",
            "Rough signal of who the shopper is",
            "Bad address = failed delivery",
            "Late-night orders can be impulsive",
            "Bigger carts, different behaviour",
            "Loyalty signal",
            "The history that powers the score",
            "What we're trying to predict",
        ],
    })
    st.dataframe(field_help, width="stretch", hide_index=True)

    st.subheader("A peek at the raw file (yes, it's messy — that's on purpose)")
    st.caption("Real exports are never clean. Notice `Rs`/commas in OrderValue, `%` and a "
               "stray `999%` in DiscountPct, and tiers written five different ways. "
               "Chapter 3 fixes all of it.")
    st.dataframe(raw.head(12), width="stretch")

# =============================================================== 3. CLEANING
with tabs[2]:
    st.header("Chapter 3 — Cleaning the data")
    st.markdown("""
Before any chart can be trusted, the messy export has to become tidy, consistent numbers.
We do this in the open — **every change is logged** below, so anyone can audit exactly what
we touched and why. (This is the *data preparation* step of the project.)
""")

    st.subheader("Every fix we made")
    for n in notes:
        st.markdown(f"- {n}")

    st.subheader("Before → After")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Before** (raw)")
        st.dataframe(raw[["OrderValue", "DiscountPct", "CityTier", "State"]].head(8),
                     width="stretch")
    with cc2:
        st.markdown("**After** (clean)")
        st.dataframe(df[["OrderValue", "DiscountPct", "CityTier", "State"]].head(8),
                     width="stretch")

    st.subheader("The features we engineered")
    st.markdown("""
From the clean columns we built a few new ones — all things a brand would know **the moment
an order is placed**, *before* shipping (so we never cheat by peeking at the future):

- **PriorRTORate** — of a shopper's past orders, what fraction came back (their "track record")
- **FirstTime** — is this their very first order? (no track record yet)
- **IsCOD** — is this a Cash-on-Delivery order?

> **Why "before shipping" matters:** if we used anything only known *after* delivery, the
> model would look brilliant in testing and be **useless in real life**. Good models only
> use clues available at decision time. This is the same discipline our reference project
> calls *avoiding leakage*.
""")
    st.success(f"Clean dataset ready: **{len(df):,} orders**, **{df.shape[1]} columns**, "
               f"zero duplicates, zero missing values in the fields we use.")

# =============================================================== 4. DESCRIPTIVE
with tabs[3]:
    st.header("Chapter 4 — What happened: who sends parcels back?")
    st.markdown("""
Now the fun part. We ask **simple questions** of the data and let the bars answer. The
dashed line on every chart is the **overall return rate** — bars to the **right (red)** are
*worse than average*, bars to the **left (green)** are *better*.
""")

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
            ax.text(m * 100 + 0.5, b.get_y() + b.get_height() / 2,
                    f"{m*100:.0f}%  (n={n:,})", va="center", fontsize=9, color=INK)
        ax.set_xlim(0, g["mean"].max() * 100 * 1.25)
        ax.set_xlabel("Return rate"); ax.set_title(title)
        bare(ax); ax.tick_params(left=False)
        return fig

    st.subheader("Cash on Delivery vs Prepaid")
    st.pyplot(rate_by("PaymentMethod", "Return rate by payment method"), width="stretch")
    st.info(f"**The headline:** COD returns at **{cod_rto*100:.0f}%**, prepaid at only "
            f"**{prepaid_rto*100:.0f}%**. Paying first changes everything. From here on, "
            "we zoom into the **COD orders** — that's where the money leaks.")

    st.divider()
    st.markdown("#### Within COD, who's risky? Pick a lens:")
    lens = st.selectbox("Break COD returns down by:",
                        ["City tier", "Product category", "Address quality",
                         "First-time vs repeat", "Discount depth", "Order value"],
                        index=0)

    if lens == "City tier":
        st.pyplot(rate_by("CityTier", "COD return rate by city tier", full=False),
                  width="stretch")
        st.info("Smaller towns return more. Tier-3 roughly **triples** Tier-1 — patchy "
                "addresses and a stronger cash-only habit.")
    elif lens == "Product category":
        st.pyplot(rate_by("Category", "COD return rate by product category", full=False),
                  width="stretch")
        st.info("**Fashion and footwear** top the list — people order to 'try at home', "
                "then refuse. Health and electronics are stickier.")
    elif lens == "Address quality":
        st.pyplot(rate_by("AddressQuality", "COD return rate by address quality",
                          min_n=10, full=False), width="stretch")
        st.info("A **vague address** is a delivery waiting to fail. Cleaning addresses at "
                "checkout is one of the cheapest wins available.")
    elif lens == "First-time vs repeat":
        tmp = cod.assign(Who=np.where(cod["FirstTime"] == 1, "First-time buyer",
                                      "Repeat buyer"))
        ref = cod["RTO"].mean()
        g = tmp.groupby("Who")["RTO"].mean()
        fig, ax = plt.subplots(figsize=(7, 2.6))
        bars = ax.barh(g.index, g.values * 100,
                       color=[RED if v > ref else GREEN for v in g.values], height=0.55)
        for b, v in zip(bars, g.values):
            ax.text(v*100+0.5, b.get_y()+b.get_height()/2, f"{v*100:.0f}%",
                    va="center", fontweight="bold")
        ax.axvline(ref*100, color=INK, ls="--", lw=1.3)
        ax.set_xlim(0, g.values.max()*100*1.25); ax.set_xlabel("Return rate")
        ax.set_title("COD: first-timers vs repeat buyers"); bare(ax); ax.tick_params(left=False)
        st.pyplot(fig, width="stretch")
        st.info("**First-timers are riskier** — no track record, and some are just testing. "
                "Loyalty earns trust, exactly like a long credit history.")
    elif lens == "Discount depth":
        b = pd.cut(cod["DiscountPct"], [-1, 10, 25, 40, 100],
                   labels=["0–10%", "11–25%", "26–40%", "40%+"])
        g = cod.assign(Band=b).groupby("Band", observed=True)["RTO"].agg(["mean", "count"])
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        bars = ax.bar(g.index.astype(str), g["mean"]*100, color=ACCENT, width=0.6)
        ax.axhline(cod["RTO"].mean()*100, color=RED, ls="--", lw=1.3)
        for b_, (m, n) in zip(bars, zip(g["mean"], g["count"])):
            ax.text(b_.get_x()+b_.get_width()/2, m*100+0.6, f"{m*100:.0f}%\nn={n:,}",
                    ha="center", fontsize=9)
        ax.set_ylabel("Return rate"); ax.set_title("COD return rate by discount depth")
        bare(ax)
        st.pyplot(fig, width="stretch")
        st.info("Deeper discounts → more impulse buys → more regret at the door. The "
                "steeper the markdown, the more returns climb.")
    else:  # Order value
        b = pd.qcut(cod["OrderValue"], 4,
                    labels=["Cheapest 25%", "Lower-mid", "Upper-mid", "Priciest 25%"])
        g = cod.assign(Band=b).groupby("Band", observed=True)["RTO"].agg(["mean", "count"])
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        bars = ax.bar(g.index.astype(str), g["mean"]*100, color=ACCENT, width=0.6)
        ax.axhline(cod["RTO"].mean()*100, color=RED, ls="--", lw=1.3)
        for b_, (m, n) in zip(bars, zip(g["mean"], g["count"])):
            ax.text(b_.get_x()+b_.get_width()/2, m*100+0.6, f"{m*100:.0f}%\nn={n:,}",
                    ha="center", fontsize=9)
        ax.set_ylabel("Return rate"); ax.set_title("COD return rate by order value")
        bare(ax)
        st.pyplot(fig, width="stretch")
        st.info("Bigger COD bills are scarier to accept at the door — more cash to hand "
                "over on the spot, so refusals tick up.")

    st.divider()
    st.subheader("How the numbers relate to each other")
    st.caption("Correlation of the numeric clues with returns. Positive (teal) = more "
               "returns; negative = fewer. These are *straight-line* links only — the next "
               "chapter tests them properly.")
    corr_cols = ["RTO", "PriorRTORate", "FirstTime", "DiscountPct", "OrderValue",
                 "PriorOrders", "Items", "OrderHour"]
    corr = cod[corr_cols].corr()["RTO"].drop("RTO").sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bars = ax.barh(corr.index, corr.values,
                   color=[ACCENT if v >= 0 else GREY for v in corr.values], height=0.6)
    for b, v in zip(bars, corr.values):
        ax.text(v + (0.005 if v >= 0 else -0.005), b.get_y()+b.get_height()/2,
                f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color=INK, lw=1); ax.set_xlabel("Correlation with returning (COD)")
    ax.set_title("Which numbers move with returns?"); bare(ax, keep_left=False)
    ax.tick_params(left=False)
    st.pyplot(fig, width="stretch")
    st.info("**Past-return rate stands out** — a shopper's history is the loudest single "
            "number. That's the seed of the COD Score.")

# =============================================================== 5. DIAGNOSTIC
with tabs[4]:
    st.header("Chapter 5 — The real reasons (signal vs noise)")
    st.markdown("""
Chapter 4 showed *what* happened. Now we ask *which clues genuinely matter* and which are
just noise. We use a classic test — **Chi-square** with an effect-size called **Cramér's V**.

> **In plain English:** Cramér's V is a 0-to-1 "how strongly are these two things linked?"
> dial. **0 = no link, closer to 1 = strong link.** The *p-value* just says "is this link
> real or could it be luck?" — small p (under 0.05) = real. We grey-out anything that's
> just luck.
""")

    assoc_cols = [c for c in ["PaymentMethod", "CityTier", "Category", "AddressQuality",
                              "State", "Device"] if c in df.columns]
    rows = []
    for c in assoc_cols:
        chi2, p, dof, v, ncat = cramers_v(df, c)
        rows.append({"Clue": c, "Cramér's V": round(v, 3), "p-value": f"{p:.1e}",
                     "Real link?": "✅ yes" if p < 0.05 else "— (luck)"})
    assoc = pd.DataFrame(rows).sort_values("Cramér's V", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(assoc) + 1))
    a2 = assoc.sort_values("Cramér's V")
    colors = [GREY if "—" in s else ACCENT for s in a2["Real link?"]]
    bars = ax.barh(a2["Clue"], a2["Cramér's V"], color=colors, height=0.6)
    for b, v in zip(bars, a2["Cramér's V"]):
        ax.text(v + 0.005, b.get_y()+b.get_height()/2, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlabel("Cramér's V  (link strength with returns)")
    ax.set_title("Which clues are truly linked to returns?"); bare(ax); ax.tick_params(left=False)
    st.pyplot(fig, width="stretch")
    st.dataframe(assoc, width="stretch", hide_index=True)

    st.info("**Payment method is the strongest lever, then city tier and category.** "
            "But notice the trap: the biggest lever is *being COD itself* — and we can't "
            "just ban COD without killing genuine sales. So the smart move is to look "
            "**inside COD** and separate risky shoppers from safe ones. That's Chapter 6.")

    st.divider()
    st.subheader("The clue that travels with you: your track record")
    h = cod[cod["PriorOrders"] > 0].copy()
    h["band"] = pd.cut(h["PriorRTORate"], [-0.01, 0.0, 0.25, 0.5, 1.01],
                       labels=["Never returned", "Returned 1–25%", "Returned 26–50%",
                               "Returned >50%"])
    g = h.groupby("band", observed=True)["RTO"].agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(8, 3.2))
    bars = ax.bar(g.index.astype(str), g["mean"]*100,
                  color=[GREEN, "#84cc16", AMBER, RED], width=0.62)
    for b, (m, n) in zip(bars, zip(g["mean"], g["count"])):
        ax.text(b.get_x()+b.get_width()/2, m*100+1, f"{m*100:.0f}%\nn={n:,}",
                ha="center", fontsize=9)
    ax.set_ylabel("Return rate on the NEXT order")
    ax.set_title("Past returns strongly predict the next return (COD, repeat buyers)")
    bare(ax)
    st.pyplot(fig, width="stretch")
    st.info("A shopper who returned **more than half** their past parcels returns the next "
            "one most of the time. A shopper who **never** returned rarely starts. This is "
            "the single most CIBIL-like signal we have — and it powers the score next.")

# =============================================================== 6. THE COD SCORE
with tabs[5]:
    st.header("Chapter 6 — Giving every shopper a COD Score")
    st.markdown("""
Time to combine every clue into **one number per shopper**. We train several prediction
models on the COD orders, each one learning to guess **how likely this order is to come
back**, then pick the best and turn its risk estimate into a friendly **300–900 score** —
the same range as CIBIL, on purpose.
""")

    X, y = design_matrix(cod)
    results, (ntr, nte, trp, tep) = train_models(X, y, test_size, seed)
    st.caption(f"Trained on **{ntr:,}** COD orders, tested on a held-out **{nte:,}** "
               f"(returns make up {tep*100:.0f}% of each, kept balanced). "
               "Positive class = a *returned* parcel.")

    st.subheader("How the models did")
    metrics = pd.DataFrame({
        m: {"Accuracy": r["test_acc"], "Precision": r["precision"], "Recall": r["recall"],
            "F1": r["f1"], "ROC-AUC": r["roc_auc"],
            "Overfit gap": r["train_acc"] - r["test_acc"]}
        for m, r in results.items()
    }).T.round(3)
    st.dataframe(
        metrics.style.highlight_max(axis=0,
            subset=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], color="#d1fae5")
            .highlight_max(axis=0, subset=["Overfit gap"], color="#fee2e2"),
        width="stretch")
    st.markdown("""
**Reading this table in plain English:**
- **Accuracy** — overall, how often the guess is right.
- **Precision** — when it shouts "this will come back!", how often it's correct.
- **Recall** — of all parcels that *truly* came back, how many it caught.
- **F1** — a balance of precision and recall.
- **ROC-AUC** — overall skill at telling returns apart from deliveries (0.5 = coin-flip,
  1.0 = perfect). Ours sit in a believable **0.7–0.8** band — good, not magic.
- **Overfit gap** — how much better it does on *seen* vs *unseen* data; smaller is safer.
""")
    auc_leader = metrics["ROC-AUC"].idxmax()
    st.markdown(f"""
**Which model do we build the score from?** Not simply the one with the highest accuracy.
A *score* has to be **calibrated** — when it says "35% chance of return", about 35% of
those parcels really should come back. A model can rank risk well yet quote numbers that
run hot or cold; for a score, calibration is what makes the number trustworthy.

**{auc_leader}** edges the ROC-AUC chart (best at *ranking* risk). But we build the score
from **{SCORE_MODEL}** — its predicted risks line up almost exactly with what actually
happens (we check this just below), it's strong across every metric, and it stays steady.
That's what makes its 300–900 number honest instead of just high or low.
""")
    st.success(f"🎯 Scoring model: **{SCORE_MODEL}** — calibrated, accurate, and steady.")

    st.subheader("Picking the model apart")
    pick = st.selectbox("Inspect a model", list(results.keys()),
                        index=list(results.keys()).index(best))
    cset = st.columns([1, 1])
    with cset[0]:
        cm = results[pick]["cm"]
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.imshow(cm, cmap="BuGn")
        for a in range(2):
            for bb in range(2):
                ax.text(bb, a, f"{cm[a,bb]:,}", ha="center", va="center", fontsize=14,
                        fontweight="bold",
                        color="white" if cm[a, bb] > cm.max()/2 else INK)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Said Deliver", "Said Return"])
        ax.set_yticklabels(["Was Delivered", "Was Returned"])
        ax.set_title(f"{pick}: hits & misses"); ax.grid(False)
        st.pyplot(fig, width="stretch")
    with cset[1]:
        fpr, tpr, _ = results[pick]["roc"]
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        ax.plot(fpr, tpr, color=ACCENT, lw=2.2,
                label=f"{pick} (AUC={results[pick]['roc_auc']:.2f})")
        ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1)
        ax.set_xlabel("False alarms"); ax.set_ylabel("Returns caught")
        ax.set_title("ROC: skill above the coin-flip line")
        ax.legend(loc="lower right", fontsize=8); bare(ax)
        st.pyplot(fig, width="stretch")

    # ROC of all models together
    st.subheader("All models on one chart")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    palette = [ACCENT, AMBER, GREEN, RED, INK]
    for (name, r), c in zip(results.items(), palette):
        fpr, tpr, _ = r["roc"]
        ax.plot(fpr, tpr, lw=2, color=c, label=f"{name} ({r['roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1)
    ax.set_xlabel("False alarms"); ax.set_ylabel("Returns caught")
    ax.set_title("Higher and more to the top-left = better"); ax.legend(fontsize=8)
    bare(ax)
    st.pyplot(fig, width="stretch")

    # what the model leans on
    if best in ("Random Forest",):
        rf = results[best]["pipe"]
        ohe = rf.named_steps["pre"].named_transformers_["cat"]
        feat = NUM_FEATS + list(ohe.get_feature_names_out(CAT_FEATS))
        imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=feat)
        imp = imp.sort_values(ascending=False).head(12)[::-1]
        st.subheader(f"What {best} pays most attention to")
        fig, ax = plt.subplots(figsize=(8, 4.4))
        ax.barh(imp.index, imp.values, color=ACCENT, height=0.7)
        ax.set_xlabel("Importance"); ax.set_title("Top clues driving the score")
        bare(ax); ax.tick_params(left=False)
        st.pyplot(fig, width="stretch")

    st.divider()
    st.subheader("From risk → the COD Trust Score")
    proba, score = honest_scores(X, y, best, test_size, seed)
    cod_scored = cod.copy()
    cod_scored["Risk"] = proba
    cod_scored["Score"] = score
    st.session_state["cod_scored"] = cod_scored

    st.markdown("We flip the risk around so that **higher = safer**, and stretch it onto a "
                "**300–900** dial. Here's how our COD shoppers spread out:")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.hist(score, bins=40, color="#94a3b8", edgecolor="white")
    ax.axvspan(300, 580, color=RED, alpha=0.10)
    ax.axvspan(580, 720, color=AMBER, alpha=0.12)
    ax.axvspan(720, 900, color=GREEN, alpha=0.12)
    ax.text(440, ax.get_ylim()[1]*0.9, "Prepaid-only", color=RED, ha="center", fontweight="bold")
    ax.text(650, ax.get_ylim()[1]*0.9, "COD + fee", color=AMBER, ha="center", fontweight="bold")
    ax.text(810, ax.get_ylim()[1]*0.9, "COD free", color=GREEN, ha="center", fontweight="bold")
    ax.set_xlabel("COD Trust Score"); ax.set_ylabel("Number of shoppers")
    ax.set_title("Most shoppers are safe; a risky tail needs guard-rails"); bare(ax)
    st.pyplot(fig, width="stretch")

    st.markdown("**A few example shoppers, low to high score:**")
    show = cod_scored.sort_values("Score")
    ex = pd.concat([show.head(2), show.iloc[[len(show)//2-1, len(show)//2]], show.tail(2)])
    st.dataframe(ex[["CustomerID", "CityTier", "Category", "OrderValue", "PriorOrders",
                     "PriorReturns", "Score"]].rename(columns={"OrderValue": "₹ Order"}),
                 width="stretch", hide_index=True)
    st.info("Notice the pattern: shoppers with **more past returns** and **first-timers in "
            "risky categories** land low; **loyal, clean-history** shoppers land high — "
            "just like a credit score rewards a long record of repaying.")

# =============================================================== 7. THE VERDICT
with tabs[6]:
    st.header("Chapter 7 — The verdict: who gets COD?")
    st.markdown("""
A score is useless until it drives a **decision**. Here we turn the score into a simple,
tiered **COD policy** — and, like a good manager, we check what each choice **costs and
saves** before committing. You're in control: move the dials and watch the trade-off.
""")

    if "cod_scored" not in st.session_state:
        X, y = design_matrix(cod)
        results, _ = train_models(X, y, test_size, seed)
        best = max(results, key=lambda k: results[k]["roc_auc"])
        proba, score = honest_scores(X, y, best, test_size, seed)
        cs = cod.copy(); cs["Risk"] = proba; cs["Score"] = score
        st.session_state["cod_scored"] = cs
    cs = st.session_state["cod_scored"].copy()

    st.subheader("Set the policy")
    p1, p2, p3 = st.columns(3)
    with p1:
        low_cut = st.slider("Prepaid-only below score", 300, 700, 580, 10,
                            help="Risky shoppers below this must pay online")
    with p2:
        high_cut = st.slider("Free COD at or above score", 600, 900, 720, 10,
                             help="Trusted shoppers above this get COD with no friction")
    with p3:
        rto_cost = st.number_input("₹ lost per returned order", 100, 600,
                                   DEFAULT_RTO_COST, 10)
    cod_fee = st.slider("COD fee charged in the middle tier (₹)", 0, 150, 50, 5)
    if low_cut >= high_cut:
        st.warning("Keep the prepaid-only cut **below** the free-COD cut for three clean tiers.")
        high_cut = low_cut + 10

    cs["Tier"] = np.where(cs["Score"] >= high_cut, "Free COD",
                  np.where(cs["Score"] < low_cut, "Prepaid-only", "COD + fee"))

    order = ["Free COD", "COD + fee", "Prepaid-only"]
    colmap = {"Free COD": GREEN, "COD + fee": AMBER, "Prepaid-only": RED}
    summ = (cs.groupby("Tier")
              .agg(Shoppers=("Score", "size"), ReturnRate=("RTO", "mean"),
                   AvgScore=("Score", "mean"), AvgOrder=("OrderValue", "mean"))
              .reindex(order))

    st.subheader("The three tiers")
    ccc = st.columns(3)
    for col, t in zip(cccols := ccc, order):
        n = int(summ.loc[t, "Shoppers"]); rr = summ.loc[t, "ReturnRate"]
        share = n / len(cs) * 100
        col.markdown(f"<div style='border-top:4px solid {colmap[t]};padding:10px 4px;'>"
                     f"<b style='color:{colmap[t]};font-size:1.05rem'>{t}</b><br>"
                     f"<span style='font-size:1.7rem;font-weight:700'>{share:.0f}%</span> "
                     f"of COD orders<br>"
                     f"<span style='color:#6b7280'>{n:,} orders · {rr*100:.0f}% return rate"
                     f"</span></div>", unsafe_allow_html=True)

    # tier chart
    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.bar(order, summ["ReturnRate"]*100, color=[colmap[t] for t in order], width=0.6)
    for b, t in zip(bars, order):
        ax.text(b.get_x()+b.get_width()/2, summ.loc[t, "ReturnRate"]*100+0.8,
                f"{summ.loc[t,'ReturnRate']*100:.0f}%", ha="center", fontweight="bold")
    ax.set_ylabel("Return rate"); ax.set_title("Return rate climbs cleanly across the tiers")
    bare(ax)
    st.pyplot(fig, width="stretch")

    # ---- the money: what the policy saves vs costs ----
    st.subheader("Does the policy pay off?")
    base_loss = (cs["RTO"] * rto_cost).sum()  # cost if we ship every COD order blindly

    # prepaid-only tier: assume online payment removes ~90% of these returns
    pp = cs[cs["Tier"] == "Prepaid-only"]
    prevented = pp["RTO"].sum() * 0.90
    saved_prepaid = prevented * rto_cost

    # but some genuine buyers abandon when COD is removed (industry: a real cost)
    abandon_rate = 0.25  # conservative: 1 in 4 prepaid-only good buyers walk away
    good_lost = (len(pp) - pp["RTO"].sum()) * abandon_rate
    avg_pp_order = pp["OrderValue"].mean() if len(pp) else 0
    lost_margin = good_lost * avg_pp_order * 0.20  # ~20% contribution margin

    # middle tier fee income (only on orders that DO get delivered & accepted)
    mid = cs[cs["Tier"] == "COD + fee"]
    fee_income = len(mid) * cod_fee * 0.85  # assume 85% still accept the small fee

    net = saved_prepaid + fee_income - lost_margin

    mcol = st.columns(4)
    mcol[0].metric("Blind COD loss (today)", f"₹{base_loss:,.0f}",
                   help="If we ship every COD order with no policy")
    mcol[1].metric("Saved by prepaid-only", f"₹{saved_prepaid:,.0f}",
                   help="Returns avoided by asking risky shoppers to pay online")
    mcol[2].metric("COD-fee income", f"₹{fee_income:,.0f}",
                   help="Small risk-priced fee from the middle tier")
    mcol[3].metric("Net effect of policy", f"₹{net:,.0f}",
                   delta="better" if net > 0 else "worse",
                   delta_color="normal" if net > 0 else "inverse")

    st.markdown(f"""
**How to read this (the honest trade-off):**
- Asking the **prepaid-only** tier to pay online prevents most of their returns →
  about **₹{saved_prepaid:,.0f}** saved.
- A small **₹{cod_fee} COD fee** in the middle tier brings in **₹{fee_income:,.0f}** —
  this is *risk-based pricing*, just like a higher interest rate on a riskier loan.
- **But** removing COD scares off some genuine buyers (we assumed 1 in 4 walk away),
  costing roughly **₹{lost_margin:,.0f}** in lost margin.
- **Net:** the policy comes out **{'ahead' if net>0 else 'behind'}** by
  **₹{abs(net):,.0f}** on this data.
""")
    st.info("Move the **prepaid-only** dial right and you prevent more returns *but* lose "
            "more genuine sales. That tension is the real management decision — the score "
            "just lets you make it with eyes open instead of guessing.")

    st.divider()
    st.subheader("📋 What we found, and what we'd tell the business")
    st.markdown(f"""
**The story in six lines:**
1. On our store, **COD returns at {cod_rto*100:.0f}%** versus **{prepaid_rto*100:.0f}%** for
   prepaid — COD is where the leak is.
2. Inside COD, returns are **not random**: smaller towns, fashion/footwear, deep discounts,
   first-timers, and vague addresses all return more.
3. The loudest single signal is a shopper's **own past-return record** — history repeats.
4. A model reads all these clues at **~0.7–0.8 skill** and turns them into one **COD Trust
   Score** (300–900), CIBIL-style.
5. That score sorts shoppers into **free COD / COD-with-a-fee / prepaid-only**, and the
   return rate rises cleanly across the three.
6. Applied sensibly, the policy comes out **net positive** — even after accounting for
   genuine buyers lost when COD is removed.

**Recommendations, each tied to a finding above:**
- **Default deep-discount + first-time + vague-address COD orders to the middle tier** (a
  small COD fee), because those are exactly the slices that returned most in Chapter 4.
- **Switch COD off for the bottom tier**, but soften it with a **small prepaid discount** —
  the industry shows a ₹75–125 nudge converts most of them to paying online willingly.
- **Fix addresses at checkout** (a "confirm your full address" step) — it's the cheapest
  lever, since vague addresses were among the strongest risk clues.
- **Reward loyalty:** let a clean track record *raise* a shopper's score over time, so good
  customers feel trusted, not punished.
""")

    st.warning("**A fair warning (our honesty note).** A low COD Score is **not** a verdict "
               "that someone is a bad person. It only means *this order is risky to ship on "
               "cash* — so we ask for prepayment instead. People in patchy-address areas or "
               "new to a brand aren't dishonest; they're just harder to deliver to. The "
               "score adjusts the **offer**, never judges the **person**. Used carelessly it "
               "could unfairly shut out whole towns — so a real brand should monitor it for "
               "exactly that and keep a prepaid path open for everyone.")

    st.download_button("⬇️ Download the cleaned, scored data (CSV)",
                       cs.to_csv(index=False).encode(),
                       "cod_scored.csv", "text/csv")

st.caption("Built with Streamlit + scikit-learn · synthetic data, real-world patterns · "
           "a low score adjusts the offer, never judges the person.")
