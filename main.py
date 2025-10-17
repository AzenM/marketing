import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import inspect

st.set_page_config(page_title="Birchbox LTV - Analyse complète", layout="wide")

def month_diff(a, b):
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return (a.year - b.year) * 12 + (a.month - b.month)

def show_table(df):
    st.dataframe(df.round(2).style.format(na_rep=""), width="stretch")

def show_code(obj_or_str, title="Code"):
    src = obj_or_str if isinstance(obj_or_str, str) else inspect.getsource(obj_or_str)
    tab_res, tab_code = st.tabs(["Résultats", "Code"])
    with tab_code:
        st.download_button("Télécharger ce code", src.encode("utf-8"), file_name=f"{title}.py")
        with st.expander("Voir / masquer le code", expanded=False):
            st.code(src, language="python")
    return tab_res

def line_chart(df, y_col, title):
    st.subheader(title)
    st.line_chart(data=df.set_index("months_since")[[y_col]], height=300)

def altair_cohort_lines(df, y_col, title):
    st.subheader(title)
    c = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("months_since:Q", title="Mois depuis acquisition"),
        y=alt.Y(f"{y_col}:Q", title="ARPU cumulé (€)"),
        color=alt.Color("yearmonth(cohort_month):T", title="Cohorte"),
        tooltip=[
            alt.Tooltip("yearmonth(cohort_month):T", title="Cohorte"),
            alt.Tooltip("months_since:Q", title="Mois"),
            alt.Tooltip(f"{y_col}:Q", title="Valeur", format=",.2f"),
        ],
    ).interactive()
    st.altair_chart(c)

def complete_and_cumsum(monthly_series, max_horizon=24):
    s = monthly_series.copy()
    if len(s.index) == 0:
        s = pd.Series(dtype=float)
    s.index = s.index.astype(int)
    s = s.reindex(range(0, max_horizon + 1), fill_value=0.0)
    return s.cumsum()

st.title("Birchbox LTV Analysis – Questions 1 à 6")

csv_path = "Alexandre Marie de ficquelmont- Albert School - B2 S1 - Data set_ LTV modelling for Birchbox - Albert School - B2 S1 - Data set_ LTV modelling for Birchbox - Feuille 1.csv"
df = pd.read_csv(csv_path, sep=None, engine="python")
df.columns = df.columns.str.strip()
df = df.rename(columns={"Customer ID": "customer_id", "Order date": "order_date", "Order value": "order_value", "Product contained in the order": "product"})
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
df["product"] = df["product"].astype("string").str.strip().replace({"": None}).fillna("(Inconnu)")
df = df.dropna(subset=["customer_id", "order_date", "order_value"])
df["order_month"] = df["order_date"].values.astype("datetime64[M]")

st.header("Q1. Revenu total par cohorte mensuelle et taille de cohorte")

def compute_q1(dfx):
    first_order = dfx.groupby("customer_id")["order_month"].min().rename("cohort_month")
    d = dfx.merge(first_order, on="customer_id", how="left")
    d["months_since"] = (d["order_month"].dt.year - d["cohort_month"].dt.year) * 12 + (d["order_month"].dt.month - d["cohort_month"].dt.month)
    cohort_revenue = d.groupby(["cohort_month", "months_since"], as_index=False)["order_value"].sum()
    cohort_size = d.groupby("cohort_month")["customer_id"].nunique().rename("cohort_size").to_frame()
    return cohort_revenue, cohort_size

cohort_revenue, cohort_size = compute_q1(df)
pivot_q1 = cohort_revenue.pivot(index="cohort_month", columns="months_since", values="order_value")
tab = show_code(compute_q1, title="compute_q1")
with tab:
    show_table(pivot_q1)
    show_table(cohort_size)

st.header("Q2. ARPU cumulé par cohorte")

def compute_q2(dfx):
    first_order = dfx.groupby("customer_id")["order_month"].min().rename("cohort_month")
    d = dfx.merge(first_order, on="customer_id", how="left")
    d["months_since"] = (d["order_month"].dt.year - d["cohort_month"].dt.year) * 12 + (d["order_month"].dt.month - d["cohort_month"].dt.month)
    cohort_size = d.groupby("cohort_month")["customer_id"].nunique().rename("cohort_size").to_frame()
    cohort_monthly = d.groupby(["cohort_month", "months_since"], as_index=False)["order_value"].sum()
    cohort_monthly = cohort_monthly.merge(cohort_size.reset_index(), on="cohort_month", how="left")
    cohort_monthly["cum_revenue"] = cohort_monthly.sort_values("months_since").groupby("cohort_month")["order_value"].cumsum()
    cohort_monthly["cum_arpu"] = cohort_monthly["cum_revenue"] / cohort_monthly["cohort_size"]
    return cohort_monthly

cohort_monthly = compute_q2(df)
pivot_q2 = cohort_monthly.pivot(index="cohort_month", columns="months_since", values="cum_arpu")
tab = show_code(compute_q2, title="compute_q2")
with tab:
    show_table(pivot_q2)

st.header("Q3. Moyenne pondérée de l’ARPU cumulé et modélisation LTV")

def weighted_arpu(cohort_monthly_df):
    recs = []
    for t in sorted(cohort_monthly_df["months_since"].unique()):
        sub = cohort_monthly_df[cohort_monthly_df["months_since"] == t]
        w = (sub["cum_arpu"] * sub["cohort_size"]).sum()
        denom = sub["cohort_size"].sum()
        if denom > 0:
            recs.append({"months_since": t, "weighted_cum_arpu": w / denom})
    return pd.DataFrame(recs)

weighted_df = weighted_arpu(cohort_monthly)
tab = show_code(weighted_arpu, title="weighted_arpu")
with tab:
    show_table(weighted_df)
    line_chart(weighted_df, "weighted_cum_arpu", "ARPU cumulé pondéré (global)")

st.header("Q4. Interprétation de l’évolution de l’ARPU et comportement client")
st.markdown(
    "- L’ARPU cumulé croît rapidement puis se stabilise.\n"
    "- Les clients récurrents dépensent surtout au début, puis ralentissent.\n\n"
    "**Hypothèses :**\n"
    "1. Réassort précoce après acquisition, puis saturation.\n"
    "2. Promotions fortes au 1er mois puis normalisation.\n"
    "3. Perte progressive d’intérêt ou offre non renouvelée."
)

st.header("Q5. Filtre ARPU cumulé par produit")
produits = ["(Tous)"] + sorted(df["product"].dropna().unique().tolist())
selected_product = st.selectbox("Sélectionne un produit :", produits)
df_filtered = df if selected_product == "(Tous)" else df[df["product"] == selected_product]
cohort_monthly_f = compute_q2(df_filtered)

fix = []
for cmo, g in cohort_monthly_f.groupby("cohort_month"):
    s = g.set_index("months_since")["cum_arpu"]
    inc = s - s.diff().fillna(s)
    s2 = complete_and_cumsum(inc, max_horizon=24)
    fix.append(pd.DataFrame({"cohort_month": cmo, "months_since": s2.index, "cum_arpu": s2.values}))
cohort_monthly_f_fixed = pd.concat(fix, ignore_index=True)

pivot_q5 = cohort_monthly_f_fixed.pivot(index="cohort_month", columns="months_since", values="cum_arpu")
tab = show_code("Q5_view", title=f"Q5_view_{selected_product.replace(' ','_')}")
with tab:
    show_table(pivot_q5)
    altair_cohort_lines(cohort_monthly_f_fixed, "cum_arpu", f"ARPU cumulé – {selected_product}")

st.header("Q6. Récap LTV(1m) / LTV(24m) par produit")

def product_recap_fixed(dfx, horizon_1=1, horizon_24=24):
    d = dfx.copy()
    d["order_month"] = d["order_date"].values.astype("datetime64[M]")
    first = d.groupby(["product", "customer_id"])["order_month"].min().rename("cohort_month").reset_index()
    d = d.merge(first, on=["product", "customer_id"], how="left")
    d["months_since"] = (d["order_month"].dt.year - d["cohort_month"].dt.year) * 12 + (d["order_month"].dt.month - d["cohort_month"].dt.month)
    rev = d.groupby(["product", "cohort_month", "months_since"], as_index=False)["order_value"].sum().rename(columns={"order_value": "rev"})
    sizes = d.groupby(["product", "cohort_month"])["customer_id"].nunique().rename("cohort_size").reset_index()
    recs = []
    max_h = max(horizon_1, horizon_24)
    for (prod, cmo), g in rev.groupby(["product", "cohort_month"]):
        s = g.set_index("months_since")["rev"]
        cum_rev = complete_and_cumsum(s, max_horizon=max_h)
        n = sizes[(sizes["product"] == prod) & (sizes["cohort_month"] == cmo)]["cohort_size"].iloc[0]
        arpu = cum_rev / float(n) if n > 0 else cum_rev * np.nan
        tmp = pd.DataFrame({"product": prod, "cohort_month": cmo, "months_since": arpu.index, "cum_arpu": arpu.values, "cohort_size": n})
        recs.append(tmp)
    full = pd.concat(recs, ignore_index=True)
    rows = []
    for prod, g in full.groupby("product"):
        for t, gt in g.groupby("months_since"):
            w = (gt["cum_arpu"] * gt["cohort_size"]).sum()
            denom = gt["cohort_size"].sum()
            if denom > 0:
                rows.append({"product": prod, "months_since": t, "weighted_cum_arpu": w / denom})
    prod_w = pd.DataFrame(rows)
    ltv1 = prod_w[prod_w["months_since"] == horizon_1][["product", "weighted_cum_arpu"]].rename(columns={"weighted_cum_arpu": "LTV_1m"})
    ltv24 = prod_w[prod_w["months_since"] == horizon_24][["product", "weighted_cum_arpu"]].rename(columns={"weighted_cum_arpu": "LTV_24m"})
    recap = ltv1.merge(ltv24, on="product", how="outer")
    recap["LTV_24m/1m_ratio"] = recap["LTV_24m"] / recap["LTV_1m"]
    return recap.sort_values("LTV_24m/1m_ratio", ascending=False)

recap_df = product_recap_fixed(df, horizon_1=1, horizon_24=24)
tab = show_code(product_recap_fixed, title="product_recap_fixed")
with tab:
    show_table(recap_df)
