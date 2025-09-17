# app.py  — versão sem filtros fora das tabs
# -*- coding: utf-8 -*-
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st

from services.open_meteo import geocode, fetch_daily, YESTERDAY
from utils.transform import monthly, normals, pick_value_for
from views.filters import render_filters
from views.temperature import render_temperature_tab
from views.precipitation import render_precipitation_tab
from views.comparison import render_comparison_tab
from views.seismicity import render_seismicity_tab
from views.forecast import render_forecast_tab
from views.climate_scenarios import render_climate_tab
from views.climate_indicators import render_climate_indicators_tab


# --------- helpers locais (iguais às que já te tinha passado) ----------
def _pick_place(query: str, key_prefix: str):
    try:
        places = geocode(query)
    except Exception as e:
        st.error(f"Falha a geocodificar '{query}': {e}")
        return None, None, None, None
    if places is None or places.empty:
        st.warning("Nenhum local encontrado.")
        return None, None, None, None

    idx = st.selectbox(
        "Escolher local",
        options=places.index,
        format_func=lambda i: places.loc[i, "label"],
        label_visibility="collapsed",
        key=f"{key_prefix}_place_sel",
    )
    row = places.loc[idx]
    lat = float(row["latitude"])
    lon = float(row["longitude"])
    tz = row.get("timezone", "auto")
    label = row.get("label", f"{lat:.4f},{lon:.4f}")
    return lat, lon, tz, label


def _prep_monthly(lat, lon, tz, start, end, month_num, base_start, base_end):
    try:
        df = fetch_daily(lat, lon, tz, start, end)
    except Exception as e:
        st.error(f"Falhou a descarga de diários: {e}")
        return None
    if df is None or df.empty:
        return None

    dfm = monthly(df)
    if dfm is None or dfm.empty:
        return None

    norm = normals(dfm, base_start, base_end)
    if norm is not None and not norm.empty:
        dfm = dfm.merge(norm, on="month", how="left")
        dfm["t_anom"] = (dfm["t_mean"] - dfm["t_norm"]) if "t_norm" in dfm.columns else pd.NA
        dfm["p_anom"] = (dfm["precip"] - dfm["p_norm"]) if "p_norm" in dfm.columns else pd.NA
    else:
        dfm["t_norm"] = pd.NA; dfm["p_norm"] = pd.NA
        dfm["t_anom"] = pd.NA; dfm["p_anom"] = pd.NA

    view_df = dfm if not month_num else dfm[dfm["month"] == month_num]
    ref_year = max(start.year, end.year - 50)
    last2_years = [end.year, end.year - 1]
    m = month_num or end.month

    def _safe(v):
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    t_50 = _safe(pick_value_for(dfm, m, ref_year, "t_mean"))
    p_50 = _safe(pick_value_for(dfm, m, ref_year, "precip"))

    t_last2 = view_df[(view_df["month"] == m) & (view_df["year"].isin(last2_years))]["t_mean"].mean()
    p_last2 = view_df[(view_df["month"] == m) & (view_df["year"].isin(last2_years))]["precip"].mean()

    return dict(
        dfm=dfm, view_df=view_df,
        ref_year=ref_year, last2_years=last2_years,
        t_50=t_50, p_50=p_50,
        t_last2=(None if pd.isna(t_last2) else float(t_last2)),
        p_last2=(None if pd.isna(p_last2) else float(p_last2)),
    )
# -----------------------------------------------------------------------


# --------- página / estilo leve ----------
st.set_page_config(page_title="Metereologia", layout="wide")
st.markdown("""
<style>
.stSelectbox, .stDateInput, .stTextInput { font-size: 0.9rem; }
div[data-baseweb="select"] > div { min-height: 34px; }
.stDateInput input { height: 34px; padding: 2px 8px; }
</style>
""", unsafe_allow_html=True)

st.title("☁️ Metereologia (ERA5)")

# --------- APENAS tabs (nenhum filtro fora das tabs) ----------
tab_fc, tab_hist, tab_eq, tab_ind, tab_cs = st.tabs(
    ["🌦️ Previsão", "📚 Histórico", "🌍 Sismicidade", "🧭 Indicadores", "📅 Cenários 2100"]
)

# ─────────────────────────── PREVISÃO ───────────────────────────
with tab_fc:
    # só LOCAL nesta página
    flt = render_filters(mode="place_only", key_prefix="fc", default_place="Lisboa")
    q_fc = flt["query"]
    lat, lon, tz, label = _pick_place(q_fc, key_prefix="fc")
    if lat is not None:
        render_forecast_tab()  # a tua view continua a gerir o resto

# ─────────────────────────── HISTÓRICO ─────────────────────────
with tab_hist:
    # filtros completos (aqui PRECISAM de estar visíveis)
    flt_h = render_filters(
        mode="full",
        key_prefix="hist",
        default_place="Lisboa",
        default_start=date(YESTERDAY.year - 10, 1, 1),
        default_end=YESTERDAY,
        place_full_label=st.session_state.get("hist_place_label"),
    )
    q = flt_h["query"]; start = flt_h["start"]; end = flt_h["end"]
    month_num = flt_h["month_num"]; month_label = flt_h["month_label"]
    base_start = flt_h["base_start"]; base_end = flt_h["base_end"]
    show_50 = flt_h["show_50"]; show_last2 = flt_h["show_last2"]

    lat, lon, tz, label = _pick_place(q, key_prefix="hist")
    if label:
        st.session_state["hist_place_label"] = label

    if lat is not None:
        prep = _prep_monthly(lat, lon, tz, start, end, month_num, base_start, base_end)
        if not prep:
            st.info("Sem dados para o período selecionado.")
        else:
            dfm = prep["dfm"]; view_df = prep["view_df"]
            ref_year = prep["ref_year"]; last2_years = prep["last2_years"]
            t_50 = prep["t_50"]; p_50 = prep["p_50"]
            t_last2 = prep["t_last2"]; p_last2 = prep["p_last2"]

            st.caption(f"Local: **{label}** • Lat/Lon: {lat:.4f}, {lon:.4f} • Fuso: {tz}")

            sub_t, sub_p, sub_cmp = st.tabs(["🌡️ Temperatura", "🌧️ Precipitação", "📊 Comparação"])
            with sub_t:
                render_temperature_tab(
                    view_df, month_num, month_label,
                    ref_year, last2_years, t_50, t_last2,
                    show_50, show_last2
                )
            with sub_p:
                render_precipitation_tab(
                    view_df, month_num, month_label,
                    ref_year, last2_years, p_50, p_last2,
                    show_50, show_last2
                )
            with sub_cmp:
                render_comparison_tab(dfm)

# ────────────────────────── SISMICIDADE ────────────────────────
with tab_eq:
    # só LOCAL; datas opcionais num expander (fechado por defeito)
    flt_eq = render_filters(mode="place_only", key_prefix="eq", default_place="Lisboa")
    q_eq = flt_eq["query"]
    lat, lon, tz, label = _pick_place(q_eq, key_prefix="eq")
    with st.expander("Período (opcional)", expanded=False):
        s_eq = st.date_input("Início", date.today().replace(year=date.today().year-10), key="eq_s")
        e_eq = st.date_input("Fim", date.today(), key="eq_e")
    if lat is not None:
        render_seismicity_tab(lat, lon, s_eq, e_eq)

# ─────────────────────────── INDICADORES ───────────────────────
with tab_ind:
    # aqui NÃO há filtros visíveis
    render_climate_indicators_tab()

# ─────────────────────────── CENÁRIOS 2100 ─────────────────────
with tab_cs:
    # aqui também não há filtros visíveis
    render_climate_tab()
