import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import inspect

st.set_page_config(page_title="Birchbox LTV - Analyse complète", layout="wide")

# ============ UTILITAIRE ============
def month_diff(a, b):
    """Différence exacte en mois entre deux dates"""
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return (a.year - b.year) * 12 + (a.month - b.month)

# ============ CHARGEMENT ============
st.title("Birchbox LTV Analysis – Questions 1 à 6")


df = pd.read_csv("Alexandre Marie de ficquelmont- Albert School - B2 S1 - Data set_ LTV modelling for Birchbox - Albert School - B2 S1 - Data set_ LTV modelling for Birchbox - Feuille 1.csv", sep=None, engine="python")

df.columns = df.columns.str.strip()
df = df.rename(columns={
    "Customer ID": "customer_id",
    "Order date": "order_date",
    "Order value": "order_value",
    "Product contained in the order": "product"
})
df = df.replace(r"^\s*$", np.nan, regex=True)
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
df["order_value"] = (
    df["order_value"]
    .astype(str)
    .str.replace("\u202f", "", regex=False)
    .str.replace(" ", "", regex=False)
    .str.replace(",", ".", regex=False)
)
df["order_value"] = pd.to_numeric(df["order_value"], errors="coerce")
df = df.dropna(subset=["customer_id", "order_date", "order_value"])
df["order_month"] = df["order_date"].values.astype("datetime64[M]")


# =============================== Q1 ===============================
st.header("Q1. Revenu total par cohorte mensuelle et taille de cohorte")

def compute_q1(df):
    first_order = df.groupby("customer_id")["order_month"].min().rename("cohort_month")
    df = df.merge(first_order, on="customer_id", how="left")
    df["months_since"] = (df["order_month"].dt.year - df["cohort_month"].dt.year) * 12 + \
                         (df["order_month"].dt.month - df["cohort_month"].dt.month)
    cohort_revenue = df.groupby(["cohort_month", "months_since"], as_index=False)["order_value"].sum()
    cohort_size = df.groupby("cohort_month")["customer_id"].nunique().rename("cohort_size").to_frame()
    return cohort_revenue, cohort_size

cohort_revenue, cohort_size = compute_q1(df)
pivot_q1 = cohort_revenue.pivot(index="cohort_month", columns="months_since", values="order_value").fillna("")
st.dataframe(pivot_q1.round(2), use_container_width=True)
st.dataframe(cohort_size, use_container_width=True)

with st.expander("Code Q1"):
    st.code(inspect.getsource(compute_q1), language="python")

st.caption("**Modif clé** : On calcule `months_since` comme différence exacte en mois pour corriger le décalage entre périodes (corrige les erreurs temporelles).")

# =============================== Q2 ===============================
st.header("Q2. ARPU cumulé par cohorte")

def compute_q2(df):
    first_order = df.groupby("customer_id")["order_month"].min().rename("cohort_month")
    df = df.merge(first_order, on="customer_id", how="left")
    df["months_since"] = (df["order_month"].dt.year - df["cohort_month"].dt.year) * 12 + \
                         (df["order_month"].dt.month - df["cohort_month"].dt.month)
    cohort_size = df.groupby("cohort_month")["customer_id"].nunique().rename("cohort_size").to_frame()
    cohort_monthly = df.groupby(["cohort_month", "months_since"], as_index=False)["order_value"].sum()
    cohort_monthly = cohort_monthly.merge(cohort_size.reset_index(), on="cohort_month", how="left")
    cohort_monthly["cum_revenue"] = cohort_monthly.sort_values("months_since").groupby("cohort_month")["order_value"].cumsum()
    cohort_monthly["cum_arpu"] = cohort_monthly["cum_revenue"] / cohort_monthly["cohort_size"]
    return cohort_monthly

cohort_monthly = compute_q2(df)
pivot_q2 = cohort_monthly.pivot(index="cohort_month", columns="months_since", values="cum_arpu").fillna("")
st.dataframe(pivot_q2.round(2), use_container_width=True)

with st.expander("Code Q2"):
    st.code(inspect.getsource(compute_q2), language="python")

st.caption("On cumule le revenu par cohorte, puis on divise par la taille pour obtenir le **ARPU cumulé** (Average Revenue Per User).")

# =============================== Q3 ===============================
st.header("Q3. Moyenne pondérée de l’ARPU cumulé et modélisation LTV")

def weighted_arpu(cohort_monthly):
    recs = []
    for t in sorted(cohort_monthly["months_since"].unique()):
        sub = cohort_monthly[cohort_monthly["months_since"] == t]
        w = (sub["cum_arpu"] * sub["cohort_size"]).sum()
        denom = sub["cohort_size"].sum()
        if denom > 0:
            recs.append({"months_since": t, "weighted_cum_arpu": w / denom})
    return pd.DataFrame(recs)

weighted_df = weighted_arpu(cohort_monthly)
st.dataframe(weighted_df.round(2), use_container_width=True)

# Graphique global
fig, ax = plt.subplots()
ax.plot(weighted_df["months_since"], weighted_df["weighted_cum_arpu"], color="green", marker="o")
ax.set_xlabel("Mois depuis acquisition")
ax.set_ylabel("ARPU cumulé pondéré (€)")
ax.set_title("ARPU cumulé pondéré (global)")
st.pyplot(fig)

with st.expander("Code Q3"):
    st.code(inspect.getsource(weighted_arpu), language="python")

st.caption("Moyenne pondérée : elle tient compte de la taille de chaque cohorte. Utilisée ensuite pour modéliser le LTV 4–5 ans.")

# =============================== Q4 ===============================
st.header("Q4. Interprétation de l’évolution de l’ARPU et comportement client")
st.markdown("""
**Constat :**
- L’ARPU cumulé croît rapidement puis se stabilise : typique d’une décroissance du réachat.
- Les clients récurrents dépensent surtout au début, puis ralentissent.

**Hypothèses :**
1. Réassort précoce après acquisition, puis saturation.  
2. Promotions fortes au 1er mois puis normalisation.  
3. Perte progressive d’intérêt ou offre non renouvelée.
""")

# =============================== Q5 ===============================
st.header("Q5. Filtre ARPU cumulé par produit")

produits = ["(Tous)"] + sorted(df["product"].dropna().unique().tolist())
selected_product = st.selectbox("Sélectionne un produit :", produits)
df_filtered = df if selected_product == "(Tous)" else df[df["product"] == selected_product]
cohort_monthly_f = compute_q2(df_filtered)
pivot_q5 = cohort_monthly_f.pivot(index="cohort_month", columns="months_since", values="cum_arpu").fillna("")
st.dataframe(pivot_q5.round(2), use_container_width=True)

fig2, ax2 = plt.subplots(figsize=(8,5))
for c, g in cohort_monthly_f.groupby("cohort_month"):
    ax2.plot(g["months_since"], g["cum_arpu"], label=str(c.date()))
ax2.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
ax2.set_title(f"ARPU cumulé - {selected_product}")
st.pyplot(fig2)

st.caption("Ce filtre permet d’isoler la dynamique ARPU d’un produit précis.")

# =============================== Q6 ===============================
st.header("Q6. Récap LTV(1m) / LTV(24m) par produit")

def product_recap(df):
    d = df.copy()
    d["order_month"] = d["order_date"].values.astype("datetime64[M]")
    first = d.groupby(["product", "customer_id"])["order_month"].min().rename("cohort_month").reset_index()
    d = d.merge(first, on=["product", "customer_id"], how="left")
    d["months_since"] = (d["order_month"].dt.year - d["cohort_month"].dt.year) * 12 + \
                         (d["order_month"].dt.month - d["cohort_month"].dt.month)
    cohort_sizes = d.groupby(["product", "cohort_month"])["customer_id"].nunique().rename("cohort_size").reset_index()
    cmon = d.groupby(["product", "cohort_month", "months_since"], as_index=False)["order_value"].sum()
    cmon = cmon.merge(cohort_sizes, on=["product", "cohort_month"], how="left")
    cmon["cum_revenue"] = cmon.sort_values("months_since").groupby(["product", "cohort_month"])["order_value"].cumsum()
    cmon["cum_arpu"] = cmon["cum_revenue"] / cmon["cohort_size"]
    rows = []
    for prod, g in cmon.groupby("product"):
        for t in sorted(g["months_since"].unique()):
            sub = g[g["months_since"] == t]
            w = (sub["cum_arpu"] * sub["cohort_size"]).sum()
            denom = sub["cohort_size"].sum()
            if denom > 0:
                rows.append({"product": prod, "months_since": t, "weighted_cum_arpu": w / denom})
    prod_w = pd.DataFrame(rows)
    ltv1 = prod_w[prod_w["months_since"] == 1][["product", "weighted_cum_arpu"]].rename(columns={"weighted_cum_arpu": "LTV_1m"})
    ltv24 = prod_w[prod_w["months_since"] == 24][["product", "weighted_cum_arpu"]].rename(columns={"weighted_cum_arpu": "LTV_24m"})
    recap = ltv1.merge(ltv24, on="product", how="outer")
    recap["LTV_24m/1m_ratio"] = recap["LTV_24m"] / recap["LTV_1m"]
    return recap.sort_values("LTV_24m/1m_ratio", ascending=False)

recap_df = product_recap(df)
st.dataframe(recap_df.round(3), use_container_width=True)
st.caption("Le ratio LTV(24m)/LTV(1m) montre les produits qui génèrent le plus de valeur long terme.")

with st.expander("Code Q6"):
    st.code(inspect.getsource(product_recap), language="python")
