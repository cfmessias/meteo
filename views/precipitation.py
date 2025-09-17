# -*- coding: utf-8 -*-
import io
import numpy as np
import pandas as pd
import streamlit as st
from utils.transform import polyfit_trend, fmt_num
from utils import charts

def render_precipitation_tab(
    view_df: pd.DataFrame,
    month_num: int | None,
    month_label: str,
    ref_year: int,
    last2_years: list[int],
    p_50: float | None,
    p_last2: float | None,
    show_50: bool,
    show_last2: bool,
):
    st.subheader(f"🌧️ Pluviosidade (acumulado mensal) — {'Todos os meses' if not month_num else month_label}")

    if month_num:
        x = view_df["year"].to_numpy(); y = view_df["precip"].to_numpy()
        fitted, per_decade = polyfit_trend(x, y)
        fig_p = charts.bar(view_df, x="year", y="precip",
                           title=f"Pluviosidade — {month_label}",
                           x_title="Ano", y_title="mm")
        if fitted is not None:
            charts.add_trend_line(fig_p, x, fitted, name=f"Tendência (~{per_decade:+.1f} mm/década)")
        if show_50 and (p_50 is not None):
            fig_p.add_scatter(x=[ref_year], y=[p_50], mode="markers+text",
                              name=f"{ref_year}", text=[f"{ref_year}"], textposition="top center")
        if show_last2 and (p_last2 is not None) and not np.isnan(p_last2):
            fig_p.add_scatter(x=[min(last2_years), max(last2_years)],
                              y=[p_last2, p_last2], mode="lines", name="Média últimos 2 anos")
    else:
        annual_p = view_df.groupby("year", as_index=False)["precip"].sum()
        fig_p = charts.bar(annual_p, x="year", y="precip",
                           title="Pluviosidade anual (soma dos 12 meses)",
                           x_title="Ano", y_title="mm")

    st.plotly_chart(fig_p, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.metric(f"Chuva em {month_label if month_num else 'mês atual'} — {ref_year}", fmt_num(p_50, " mm", 1))
    with c4:
        st.metric("Chuva — média últimos 2 anos", fmt_num(p_last2, " mm", 1),
                  delta=(None if (p_50 is None or p_last2 is None or np.isnan(p_last2))
                         else f"{p_last2 - p_50:+.1f} mm"))

    # Tabela + CSV (igual à de temperatura, por conveniência)
    with st.expander("📄 Dados (mensal por ano)"):
        show_cols = ["year","month","year_month","t_mean","t_norm","t_anom","precip","p_norm","p_anom"]
        grid = view_df[show_cols].sort_values(["year","month"]).copy()
        grid["year"] = grid["year"].astype(int).astype(str)
        grid["year-month"] = pd.to_datetime(grid["year_month"]).dt.strftime("%Y-%m")
        cols_out = ["year","month","year-month","t_mean","t_norm","t_anom","precip","p_norm","p_anom"]
        st.dataframe(grid[cols_out], use_container_width=True)
        buf = io.StringIO(); grid[cols_out].to_csv(buf, index=False)
        st.download_button(
            "💾 Download CSV",
            data=buf.getvalue(),
            file_name="tendencias_mensais_precip.csv",
            mime="text/csv",
            key="dl_csv_precip"        # <— chave única
        )

