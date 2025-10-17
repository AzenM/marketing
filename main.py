def product_recap_fixed(df, horizon_1=1, horizon_24=24):
    d = df.copy()
    d["order_month"] = d["order_date"].values.astype("datetime64[M]")
    first = (
        d.groupby(["product", "customer_id"])["order_month"]
        .min()
        .rename("cohort_month")
        .reset_index()
    )
    d = d.merge(first, on=["product", "customer_id"], how="left")
    d["months_since"] = (d["order_month"].dt.year - d["cohort_month"].dt.year) * 12 + \
                        (d["order_month"].dt.month - d["cohort_month"].dt.month)

    # revenus mensuels par produit/cohorte/mois
    rev = (
        d.groupby(["product", "cohort_month", "months_since"], as_index=False)["order_value"]
        .sum()
        .rename(columns={"order_value": "rev"})
    )
    # tailles de cohorte
    sizes = (
        d.groupby(["product", "cohort_month"])["customer_id"]
        .nunique()
        .rename("cohort_size")
        .reset_index()
    )

    recs = []
    for (prod, cmo), g in rev.groupby(["product", "cohort_month"]):
        # reindex 0..24 (ou plus) et cumul “n = n + (n-1) + …”
        s = g.set_index("months_since")["rev"]
        cum_rev = complete_and_cumsum(s, max_horizon=max(horizon_1, horizon_24))
        n = sizes[(sizes["product"] == prod) & (sizes["cohort_month"] == cmo)]["cohort_size"].iloc[0]
        arpu = cum_rev / float(n) if n > 0 else cum_rev * np.nan
        tmp = pd.DataFrame({
            "product": prod,
            "cohort_month": cmo,
            "months_since": arpu.index,
            "cum_arpu": arpu.values,
            "cohort_size": n
        })
        recs.append(tmp)

    full = pd.concat(recs, ignore_index=True)

    # moyenne pondérée par mois (pondérée par taille de cohorte)
    rows = []
    for prod, g in full.groupby("product"):
        for t, gt in g.groupby("months_since"):
            w = (gt["cum_arpu"] * gt["cohort_size"]).sum()
            denom = gt["cohort_size"].sum()
            if denom > 0:
                rows.append({"product": prod, "months_since": t, "weighted_cum_arpu": w / denom})
    prod_w = pd.DataFrame(rows)

    ltv1  = prod_w[prod_w["months_since"] == horizon_1][["product", "weighted_cum_arpu"]].rename(columns={"weighted_cum_arpu": "LTV_1m"})
    ltv24 = prod_w[prod_w["months_since"] == horizon_24][["product", "weighted_cum_arpu"]].rename(columns={"weighted_cum_arpu": "LTV_24m"})
    recap = ltv1.merge(ltv24, on="product", how="outer")
    recap["LTV_24m/1m_ratio"] = recap["LTV_24m"] / recap["LTV_1m"]
    return recap.sort_values("LTV_24m/1m_ratio", ascending=False)

recap_df = product_recap_fixed(df, horizon_1=1, horizon_24=24)
tab = show_code(product_recap_fixed, title="product_recap_fixed")
with tab:
    show_table(recap_df)
