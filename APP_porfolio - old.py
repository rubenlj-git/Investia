
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import plotly.express as px
import re
import requests
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import plotly.express as px
import streamlit as st
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
from streamlit_gsheets import GSheetsConnection
from datetime import datetime

st.set_page_config(
    page_title="Investia",page_icon=":bar_chart:",layout="wide"
)

@st.cache_resource
def calculate_returns(df):
    df = df.set_index("Date")
    returns_df = pd.DataFrame()
    for i in df.columns.tolist():
        returns_df[i] = df[i].pct_change()
    return(returns_df)
@st.cache_resource
def calculate_cum_ret(df):
    df_cumprod = ((df +1).cumprod())-1
    return(df_cumprod) 
@st.cache_resource
def returns_annualized(df):
    mean_return_year = df.resample("Y").apply(
        lambda x: np.prod(1 + x)**(12 / len(x)) - 1
    )
    mean_return_year = mean_return_year.reset_index()
    mean_return_year['Year']= pd.DatetimeIndex(mean_return_year['Date']).year
    return(round(mean_return_year.drop("Date",axis=1).set_index("Year").mul(100),2))
@st.cache_resource
def calculate_annual_volatility(df, index, starting_date, ending_date):
    df = df[(df.index>=pd.to_datetime(starting_date)) & (df.index<=pd.to_datetime(ending_date))][index]
    monthly_volatility = df.std()
    annual_volatility = monthly_volatility * np.sqrt(12)
    return annual_volatility
@st.cache_resource
def calculate_annual_returns(df):
    monthly_returns = df.mean()
    annual_returns = (1 + monthly_returns)**12-1
    return annual_returns
@st.cache_resource
def calculate_ratio_sharpe(annual_ret_avg, risk_free_rate, annual_vol_avg):
    sharpe_r= (annual_ret_avg - risk_free_rate)/annual_vol_avg
    return round(sharpe_r,2)

@st.cache_resource
def calculate_cagr(df, index, starting_date, ending_date):
    df = df[(df.index>=pd.to_datetime(starting_date)) & (df.index<=pd.to_datetime(ending_date))]
    df = df +1
    beginning_value = df[index].dropna().values[0]
    ending_value = df[index].dropna().values[-1]
    start_date = df[index].dropna().index[0]
    end_date = df[index].dropna().index[-1]
    n_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    n_years = n_months / 12
    cagr = ((ending_value / beginning_value) ** (1 / n_years)) - 1
    return cagr
@st.cache_resource
def calculate_net_asset_value(ret_df, index, starting_date, ending_date):
    ret_df = ret_df[(ret_df.index>=pd.to_datetime(starting_date)) & (ret_df.index<=pd.to_datetime(ending_date))]
    cumulative_ret_df = calculate_cum_ret(ret_df)
    net_asset_value = 10000*(1+cumulative_ret_df[index].values[-1])
    return round(net_asset_value,2)

@st.cache_resource
def maximum_drawdown(df):
    max_drawdown_info = {}
    for column in df.columns.tolist():
        cumulative_returns = (1 + df[column]).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        end_date = drawdown.idxmin()
        
        start_date = cumulative_returns[cumulative_returns.index<=end_date].idxmax()
        
        max_drawdown_info[column] = {
            'Max Drawdown': max_drawdown,
            'Start Date': start_date,
            'End Date': end_date
        }
    
    return max_drawdown_info
@st.cache_resource
def calculate_minimum_investment_horizon(df, max_years=15):
    prob_winning_dict = {}

    for column in df.columns.tolist():
        prob_winning_list = []

        for years in range(1, max_years + 1):
            window_size = years * 12
            rolling_cum_returns = df[column].rolling(window=window_size).apply(lambda x: (1 + x).prod() - 1)
            prob_winning = (1 - len(rolling_cum_returns[rolling_cum_returns < 0]) / len(rolling_cum_returns))
            prob_winning_list.append(prob_winning)

        prob_winning_dict[column] = prob_winning_list

    # Convert the dictionary to a DataFrame
    prob_winning_df = pd.DataFrame(prob_winning_dict, index=range(1, max_years + 1))
    prob_winning_df.index.name = 'Years'
    
    return prob_winning_df

@st.cache_resource
def portfolio_returns(df_ret, list_weights):
    weights = np.array(list_weights)

    if df_ret.shape[1] != len(weights):
        st.warning(":red[WARNING: El número de índices debe coincidir con el número de ponderaciones proporcionadas.]", icon="⚠️")

    if not np.isclose(weights.sum(), 1.0):
        st.warning("WARNING: Las ponderaciones deben sumar 1. Las ponderaciones proporcionadas suman {:.2f}".format(weights.sum()), icon="⚠️")

    weighted_returns = df_ret.dropna().mul(weights, axis=1)
    portfolio_returns = weighted_returns.sum(axis=1)
    portfolio_df = pd.DataFrame(portfolio_returns, columns=['Portfolio'])

    return (portfolio_df)

@st.cache_resource
def plot_line_chart(df, index=None):
    fig = go.Figure()

    if index is not None:
        # When an index (column name) is provided, plot just that column
        y_data = df[index].dropna()
        y_data = (y_data+1)*10000
        fig.add_trace(go.Scatter(
            x=y_data.index,  # Assuming 'Date' is already the index
            y=y_data.values,
            mode='lines',
            name=index
        ))
        fig.update_layout(
            title=f'Evolution of 10.000€ invested in {index}',
            xaxis_title='Date',
            yaxis_title=f'{index} (%)',
            xaxis=dict(
                showline=True,
                showgrid=False,
                showticklabels=True,
                tickangle=-45
            ),
            yaxis=dict(showgrid=True),
        )
    else:
        # When no index is provided, plot all columns
        for column in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,  # Assuming 'Date' is the index of the DataFrame
                y=df[column],
                mode='lines',
                name=column
            ))
        fig.update_layout(
            title='Cumulative Returns Over Time',
            xaxis_title='Date',
            yaxis_title='Cumulative Return (%)',
            xaxis=dict(showline=True, showgrid=False, showticklabels=True),
            yaxis=dict(showgrid=True),
            legend=dict(x=0.01, y=0.99)
        )

    return fig

################################################################################################################
@st.cache_resource
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Retornos simples diarios."""
    return prices.pct_change()
@st.cache_resource
def compute_cagr(prices: pd.Series, periods_per_year: int = 252) -> float:
    """CAGR basado en primera y última observación válidas."""
    s = prices.dropna()
    if len(s) < 2:
        return np.nan
    n_days = (s.index[-1] - s.index[0]).days
    if n_days <= 0:
        return np.nan
    years = n_days / 365.25
    return (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1
@st.cache_resource
def _max_drawdown(prices: pd.Series) -> float:
    """Máximo drawdown (valor negativo)."""
    s = prices.dropna()
    if s.empty:
        return np.nan
    dd = s / s.cummax() - 1
    return dd.min()
@st.cache_resource
def rolling_max_drawdown(prices: pd.Series, window: int = 252) -> pd.Series:
    """Max drawdown rolling en ventana (por ejemplo 252 sesiones ~ 1 año)."""
    s = prices.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    def _mdd(x):
        x = pd.Series(x)
        dd = x / x.cummax() - 1
        return dd.min()
    return s.rolling(window).apply(_mdd, raw=False)
@st.cache_resource
def perf_metrics(prices: pd.DataFrame, rf_annual: float = 0.0, periods_per_year: int = 360) -> pd.DataFrame:
    rets = compute_returns(prices)
    out = []
    for col in prices.columns:
        s = prices[col].dropna()
        if len(s) < 2:
            out.append({
                "isin": col,
                "inicio_hist": pd.NaT,
                "fin_hist": pd.NaT,
                "n_obs": 0,
                "retorno_total": np.nan,
                "cagr": np.nan,
                "vol_anual": np.nan,
                "sharpe": np.nan,
                "max_drawdown": np.nan,
            })
            continue

        inicio, fin = s.index[0], s.index[-1]
        n_obs = len(s)
        total_return = s.iloc[-1] / s.iloc[0] - 1
        cagr = compute_cagr(s, periods_per_year=periods_per_year)

        r = rets[col].dropna()
        vol_annual = r.std() * np.sqrt(periods_per_year) if len(r) > 1 else np.nan
        rf_daily = (1 + rf_annual) ** (1 / periods_per_year) - 1
        mean_excess_daily = (r - rf_daily).mean() if len(r) > 0 else np.nan
        sharpe = (mean_excess_daily * periods_per_year) / vol_annual if (vol_annual and vol_annual > 0) else np.nan
        ultimo_nav = s.iloc[-1]
        current_dd = s.iloc[-1] / s.cummax().iloc[-1] - 1
        z = (s - s.rolling(63).mean()) / s.rolling(63).std()
        z_last = z.iloc[-1]

        mdd = _max_drawdown(s)

        out.append({
            "isin": col,
            "inicio_hist": inicio,
            "fin_hist": fin,
            "n_obs": n_obs,
            "cagr": cagr,
            "vol_anual": vol_annual,
            "sharpe": sharpe,
            "current_drawdown":current_dd,
            "max_drawdown": mdd,
            "z_score3meses": z_last,
            "ultimo_vl":ultimo_nav
        })

    return pd.DataFrame(out).set_index("isin").sort_values("sharpe", ascending=False)
@st.cache_resource
def top3_dd(s: pd.Series) -> pd.DataFrame:
    s = s.dropna()
    if s.empty:
        return pd.DataFrame(columns=["max_drawdown", "start", "trough", "recovery"])

    dd = s / s.cummax() - 1
    seg = (dd == 0).cumsum()

    dd_neg = dd[dd < 0]
    if dd_neg.empty:
        return pd.DataFrame(columns=["max_drawdown", "start", "trough", "recovery"])

    out_rows = []
    for k, grp in dd_neg.groupby(seg.loc[dd_neg.index]):
        trough_idx = grp.idxmin()
        mdd = grp.min()

        peak_candidates = dd[(seg == k) & (dd == 0)]
        start_idx = peak_candidates.index.max() if not peak_candidates.empty else dd.index.min()

        after_trough = dd.loc[trough_idx:]
        rec_candidates = after_trough[after_trough.eq(0)]
        recovery_idx = rec_candidates.index.min() if not rec_candidates.empty else pd.NaT
        days_to_recovery = (recovery_idx - start_idx).days if pd.notna(recovery_idx) else np.nan
        out_rows.append(
            {"max_drawdown": mdd, "start": start_idx, "trough": trough_idx, "recovery": recovery_idx, "days_to_recover": days_to_recovery}
        )

    out = pd.DataFrame(out_rows)
    return out.sort_values("max_drawdown").head(10).reset_index(drop=True)
@st.cache_resource
def compute_signals(P: pd.DataFrame) -> pd.DataFrame:
    # ---------- INDICADORES (último valor) ----------
    z = (P - P.rolling(63).mean()) / P.rolling(63).std()
    z_last = z.iloc[-1]
    p_min = P.rolling(63).min().iloc[-1]
    p_max = P.rolling(63).max().iloc[-1]
    pct = (P.iloc[-1] - p_min) / (p_max - p_min)
    dist_ma63 = P.iloc[-1] / P.rolling(63).mean().iloc[-1] - 1
    dd = P.iloc[-1] / P.cummax().iloc[-1] - 1
    # ---------- SEÑALES (umbrales suaves para fondos) ----------
    signals = pd.DataFrame(index=P.columns)
    signals["zscore"] = np.where(
        z_last > 0.5, "sobrecompra",
        np.where(z_last < -0.5, "sobreventa", "neutral")
    )
    signals["percentil_52w"] = np.where(
        pct > 0.80, "sobrecompra",
        np.where(pct < 0.20, "sobreventa", "neutral")
    )
    signals["dist_ma63"] = np.where(
        dist_ma63 > 0.02, "sobrecompra",
        np.where(dist_ma63 < -0.02, "sobreventa", "neutral")
    )
    signals["drawdown"] = np.where(
        dd < -0.03, "sobreventa", "neutral"
    )
    # ---------- CONSENSO ----------
    score = (
        (signals == "sobrecompra").sum(axis=1)
        - (signals == "sobreventa").sum(axis=1)
    )

    signals["consenso"] = np.where(
        score >= 2, "sobrecompra",
        np.where(score <= -2, "sobreventa", "neutral")
    )
    return signals.reset_index().rename(columns={"index": "isin"})
def retornos(df, ticker, year_month, n_year_month):
    serie = df[ticker].dropna()
    hoy = serie.index.max()
    if year_month=="year":
        hace_xy = hoy - pd.DateOffset(years=n_year_month)
        valor_hoy = serie.loc[hoy]
        valor_hace_xy = serie.loc[serie.index <= hace_xy].iloc[-1]
        retorno_xy_anualizado = (valor_hoy / valor_hace_xy) ** (1/n_year_month) - 1
    if year_month=="month":
        hace_xy = hoy - pd.DateOffset(months=n_year_month)
        valor_hoy = serie.loc[hoy]
        valor_hace_xy = serie.loc[serie.index <= hace_xy].iloc[-1]
        retorno_xy_anualizado = (valor_hoy / valor_hace_xy) - 1
    return(retorno_xy_anualizado)
def volatilidad(df, ticker, n):
    s = df[ticker].dropna().sort_index()
    m = s.resample("M").last()
    hoy = m.index.max()
    inicio = hoy - pd.DateOffset(years=n)
    m = m.loc[m.index >= inicio]
    r = m.pct_change().dropna()
    if r.empty:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(12))
@st.cache_resource
def backtest_cartera(
    P: pd.DataFrame,
    w,
    plot: bool = True
):
    R = P[w.index].pct_change()
    R = R.loc[R[w.index].notna().all(1)].dot(w)     # backtest desde primera fecha común
    nav = (1 + R).cumprod()
    top3 = top3_dd(nav)
    years = (R.index[-1] - R.index[0]).days / 365.25
    retorno_anualizado = (1 + R).prod()**(1/years) - 1
    vol_anualizada = R.std() * np.sqrt(252)
    if plot==True:
        return({"CAGR": round(retorno_anualizado,3), "vol":round(vol_anualizada, 3), "Drawdown": top3})
    else:
        return(nav / nav.iloc[0] * 100)

    # return out, retorno_anualizado, vol_anualizada, top3
@st.cache_resource
def plot_donut_posiciones(metrics_df, amount_col="Posición", label_col="nombre_fondo",
                          title="Distribución de la inversión por fondo (€)", figsize=(13,13)):
    
    s = (metrics_df[metrics_df["Posición"]>0][[amount_col, label_col]]
         .dropna()
         .assign(**{amount_col: pd.to_numeric(metrics_df[metrics_df["Posición"]>0][amount_col], errors="coerce")})
         .dropna())
    s = s[s[amount_col] > 0].sort_values(amount_col, ascending=False)
    if s.empty: 
        return

    amounts = s[amount_col].to_numpy()
    labels  = s[label_col].astype(str).to_list()
    colors  = plt.cm.Set3(np.linspace(0, 1, len(amounts)))

    fig, ax = plt.subplots(figsize=figsize)
    wedges, _, autotexts = ax.pie(
        amounts, labels=None,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90, colors=colors,
        wedgeprops=dict(width=0.4, edgecolor="white")
    )
    plt.setp(autotexts, size=10, weight="bold", color="black")

    legend_labels = [f"{n} — {a:,.0f} €" for n, a in zip(labels, amounts)]
    ax.legend(wedges, legend_labels, title="Fondos", loc="center left",
              bbox_to_anchor=(1, 0.5), frameon=False)

    ax.set(aspect="equal")
    ax.set_title(title, fontsize=14, weight="bold")
    plt.tight_layout()
    plt.show()

def plot_donut_cartera(df, category_col="categoria", value_col="Posición", title="Distribución por Categoría"):
    fig = px.pie(
        df,
        names=category_col,
        values=value_col,
        hole=0.4,                  # donut effect
        title=title
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label'
    )
    
    st.plotly_chart(fig, use_container_width=True)
@st.cache_resource
def plot_treemap_cartera(df, title="Renta Variable"):
    df = df.copy()
    df["nombre_fondo"] = df["nombre_fondo"].astype(str).str.strip()
    df = df[df["nombre_fondo"].ne("") & df["nombre_fondo"].ne("nan")]
    df["Posición"] = pd.to_numeric(df["Posición"], errors="coerce").round(1)
    df["Var"] = pd.to_numeric(df["Var"], errors="coerce").round(1)
    df["% Var"] = pd.to_numeric(df["% Var"], errors="coerce").round(1)

    fig = px.treemap(
        df,
        path=[px.Constant(title), "nombre_fondo"],
        values="Posición",
        color="Var",
        color_continuous_scale=["#8b0000","#ff4d4d","#f5f5f5","#66cc66","#006400"],
        color_continuous_midpoint=0,
        custom_data=["Posición","Var","% Var"]
    )

    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{customdata[0]:,.1f} €<br>%{customdata[1]:+,.1f} € (%{customdata[2]:+.1f}%)",
        textinfo="text",
        hovertemplate="<b>%{label}</b><br>Peso: %{percentParent:.1%}<br>Posición: %{customdata[0]:,.1f} €<br>Variación: %{customdata[1]:+,.1f} € (%{customdata[2]:+.1f}%)<extra></extra>",
        root_color="white"
    )

    fig.update_layout(
        margin=dict(t=0,l=0,r=0,b=0),
        height=700,
        coloraxis_showscale=False
    )

    st.plotly_chart(fig, use_container_width=True)

@st.cache_resource
def plot_df_lines(df, title="", ylabel="Base 100"):

    df_plot = df.reset_index().melt(
        id_vars=df.index.name or "index",
        var_name="Serie",
        value_name="Valor"
    )

    fig = px.line(
        df_plot,
        x=df_plot.columns[0],
        y="Valor",
        color="Serie",
        title=title,
        labels={"Valor": ylabel, df_plot.columns[0]: ""},
    )

    fig.update_layout(
        showlegend=False,
        template="simple_white",
        legend_title_text="",
        hovermode="x unified",
        height=500,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    fig.update_traces(line=dict(width=2.2))
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")

    st.plotly_chart(fig, use_container_width=True)
@st.cache_resource
def plot_df_lines_aux(df, title="", ylabel="Base 100"):

    df_plot = df.reset_index().melt(
        id_vars=df.index.name or "index",
        var_name="Serie",
        value_name="Valor"
    )

    xcol = df_plot.columns[0]

    fig = px.line(
        df_plot,
        x=xcol,
        y="Valor",
        color="Serie",
        title=title,
        labels={"Valor": ylabel, xcol: ""},
    )

    # --- marcar último valor (mínimo añadido) ---
    last_points = (
        df_plot.sort_values(xcol)
              .groupby("Serie", as_index=False)
              .tail(1)
    )

    fig.add_scatter(
        x=last_points[xcol],
        y=last_points["Valor"],
        mode="markers+text",
        text=last_points["Valor"].round(2).astype(str),
        textposition="top center",
        showlegend=False,
    )
    # -------------------------------------------

    fig.update_layout(
        showlegend=False,
        template="simple_white",
        legend_title_text="",
        hovermode="x unified",
        height=500,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    fig.update_traces(line=dict(width=2.2))
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")

    st.plotly_chart(fig, use_container_width=True)


@st.cache_resource
def show_positions(name: str, w: pd.Series, metrics_df: pd.DataFrame):
    mf = metrics_df.set_index("isin")[["nombre_fondo","Perc_posicion"]]
    pos = (w.rename("w").to_frame()
            .join(mf, how="left")
            .assign(w=lambda d: d["w"].astype(float))
            .assign(Perc=lambda d: (d["w"]*100).round(2))
            .sort_values("Perc", ascending=False)
            .reset_index(drop=True))
    st.dataframe(
        pos[["nombre_fondo","Perc"]].rename(columns={"nombre_fondo":"Fondo","Perc":"%"}),
        use_container_width=True,
        hide_index=True,
        column_config={
            "%": st.column_config.ProgressColumn("%", min_value=0, max_value=float(pos["Perc"].max() or 100), format="%.2f")
        },
    )
@st.cache_resource
def show_positions_eur(metrics_df: pd.DataFrame, df_final_ff: pd.DataFrame):
    st.dataframe(
        metrics_df[["nombre_fondo", "Perc_posicion", "Posición", "% Var", "Var"]].dropna(axis=0).rename(columns={"nombre_fondo": "Fondo", "Perc_posicion":"Pesos (%)"}),
        use_container_width=True,
        hide_index=True,
    )

@st.cache_resource
def show_backtest_report(name: str, out: dict):
    st.subheader(name)
    c1, c2, c3 = st.columns([1,1,1])
    c1.metric("CAGR", f"{out.get('CAGR', np.nan):.2%}" if pd.notna(out.get("CAGR")) else "—")
    c2.metric("Volatilidad", f"{out.get('vol', np.nan):.2%}" if pd.notna(out.get("vol")) else "—")
    dd = out.get("Drawdown", pd.DataFrame())
    mdd = dd["max_drawdown"].min() if isinstance(dd, pd.DataFrame) and not dd.empty else np.nan
    c3.metric("Máx. Drawdown", f"{mdd:.2%}" if pd.notna(mdd) else "—")
    dd_show = dd.copy()
    for col in ["start", "trough", "recovery"]:
        if col in dd_show.columns:
            dd_show[col] = pd.to_datetime(dd_show[col], utc=True, errors="coerce").dt.tz_convert(None).dt.date
    if "max_drawdown" in dd_show.columns:
        dd_show["max_drawdown"] = dd_show["max_drawdown"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
    if "days_to_recover" in dd_show.columns:
        dd_show["days_to_recover"] = dd_show["days_to_recover"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "—")

    dd_show = dd_show.rename(columns={
        "max_drawdown":"Max DD",
        "start":"Start",
        "trough":"Trough",
        "recovery":"Recovery",
        "days_to_recover":"Días"
    })
    st.caption("Top 3 drawdowns (start → trough → recovery)")
    st.dataframe(dd_show[["Max DD","Start","Trough","Recovery","Días"]], use_container_width=True, hide_index=True)


conn = st.connection("gsheets", type=GSheetsConnection)
nombre_fondo = conn.read(worksheet="Links")[["nombre_fondo", "isin", "categoria"]]
carteras_x = conn.read(worksheet="Carteras_x").drop("nombre_fondo",axis=1)
df_final = pd.read_json("NAV.json", orient="records").set_index("date")
df_final_ff = df_final.sort_index().ffill()
signals = compute_signals(df_final_ff)
metrics_df = perf_metrics(df_final_ff, rf_annual=0.0, periods_per_year=252).reset_index()
metrics_df = pd.merge(metrics_df, nombre_fondo, how="left", on="isin")
metrics_df = pd.merge(metrics_df, signals, how="left", on="isin")
weights = conn.read(worksheet="INDEX")[["Unnamed: 1", "Unnamed: 4", "Unnamed: 5"]]
weights.columns= ["isin", "Participaciones", "Coste_medio"]
metrics_df = pd.merge(metrics_df, weights, how="left", on="isin")
metrics_df["Posición"] = metrics_df["ultimo_vl"]*metrics_df["Participaciones"]
metrics_df['Perc_posicion'] = (metrics_df['Posición'] / metrics_df['Posición'].sum() * 100).round(2)
metrics_df["Posición_ini"] = metrics_df["Coste_medio"]*metrics_df["Participaciones"]
metrics_df['% Var'] = (metrics_df['Posición']/metrics_df['Posición_ini']-1).mul(100).round(2)
metrics_df['Var'] = (metrics_df['Posición'] - metrics_df['Posición_ini']).round(2)
dd_top3 = pd.concat(
    {
        isin: top3_dd(df_final[isin]).assign(rank=lambda d: np.arange(1, len(d) + 1))
        for isin in df_final.columns
    },
    names=["isin"]
).reset_index(level=0).reset_index(drop=True)
posiciones_df = metrics_df[metrics_df["inicio_hist"] <= "2020-02-08"].set_index("isin")["Perc_posicion"].dropna() / 100 # [metrics_df["inicio_hist"] <= "2020-02-08"]
dd_top3 = pd.merge(dd_top3, nombre_fondo, how="left", on="isin")
ddtop_aux = dd_top3[(dd_top3["rank"]==1)].sort_values("days_to_recover", ascending=False)[["isin", "start", "trough", "recovery", "days_to_recover"]].dropna()
metrics_df_final = pd.merge(metrics_df, ddtop_aux, how="left", on="isin").set_index("nombre_fondo").reset_index()
metrics_df_final["nombre_fondo_isin"] = metrics_df_final["nombre_fondo"] + " (" +metrics_df_final["isin"] + ")"
def retornos(df, ticker, year_month, n_year_month):
    serie = df[ticker].dropna()
    hoy = serie.index.max()
    if year_month=="year":
        hace_xy = hoy - pd.DateOffset(years=n_year_month)
        valor_hoy = serie.loc[hoy]
        valor_hace_xy = serie.loc[serie.index <= hace_xy].iloc[-1]
        retorno_xy_anualizado = (valor_hoy / valor_hace_xy) ** (1/n_year_month) - 1
    if year_month=="month":
        hace_xy = hoy - pd.DateOffset(months=n_year_month)
        valor_hoy = serie.loc[hoy]
        valor_hace_xy = serie.loc[serie.index <= hace_xy].iloc[-1]
        retorno_xy_anualizado = (valor_hoy / valor_hace_xy) - 1
    return(retorno_xy_anualizado)
def volatilidad(df, ticker, n):
    s = df[ticker].dropna().sort_index()
    m = s.resample("M").last()
    hoy = m.index.max()
    inicio = hoy - pd.DateOffset(years=n)
    m = m.loc[m.index >= inicio]
    r = m.pct_change().dropna()
    if r.empty:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(12))

COMBINACIONES = [
    ("year", 1),
    ("year", 3),
    ("year", 5),
    ("year", 7),
    ("year", 10),
    ("month", 1),
    ("month", 3),
    ("month", 6),
]
resultados = {}

for ticker in df_final.columns:
    resultados[ticker] = {}
    for freq, n in COMBINACIONES:
        col_ret = f"ret_{freq}_{n}"
        try:
            resultados[ticker][col_ret] = retornos(df_final, ticker, freq, n)
        except:
            resultados[ticker][col_ret] = None
        if freq=="year":
            col_vol = f"vol_{freq}_{n}"
            try:
                resultados[ticker][col_vol] = volatilidad(df_final, ticker, n)
            except:
                resultados[ticker][col_vol] = None

df_retornos_volatilidad = pd.DataFrame(resultados).T.reset_index().rename(columns={"index":"isin"})
df_retornos_volatilidad["sharpe_3"] = df_retornos_volatilidad["ret_year_3"]/df_retornos_volatilidad["vol_year_3"]
df_retornos_volatilidad["sharpe_5"] = df_retornos_volatilidad["ret_year_5"]/df_retornos_volatilidad["vol_year_5"]
df_retornos_volatilidad["sharpe_10"] = df_retornos_volatilidad["ret_year_10"]/df_retornos_volatilidad["vol_year_10"]
metrics_df_final = metrics_df_final.merge(df_retornos_volatilidad, on="isin", how="left")
TRAMOS_RIESGO = {
    (float("-inf"), 0.02): "Muy Bajo Riesgo",
    (0.02, 0.04): "Riesgo bajo",
    (0.04, 0.06): "Riesgo medio",
    (0.06, float("inf")): "Riesgo alto",
}

def asignar_riesgo(vol_5, vol_3, tramos=TRAMOS_RIESGO):
    vol = vol_5 if not pd.isna(vol_5) else vol_3

    if pd.isna(vol):
        return np.nan

    for (a, b), etiqueta in tramos.items():
        if a <= vol < b:
            return etiqueta
    return np.nan
metrics_df_final["categoria_riesgo"] = metrics_df_final.apply(
    lambda x: asignar_riesgo(x["vol_year_5"], x["vol_year_3"]),
    axis=1
)

############################################################################################################

@st.cache_resource
def bond_returns_from_yield(yield_series, maturity):
    y = yield_series / 100.0
    duration_map = {
        2: 1.9,
        5: 4.7,
        10: 8.5,
        30: 18.0
    }
    D = duration_map[maturity]
    dy = y.diff()
    r = y.shift(1)/12 - D * dy
    price = 100 * (1 + r).cumprod()
    return price

# @st.cache_resource
# def import_yahoo_fin():
#     tickers = {
#         "^GSPC": "S&P 500",
#         "^NDX": "Nasdaq 100",
#         "GC=F": "Gold Spot price",
#         "EEM": "MSCI Emerging",
#         "^990100-USD-STRD": "MSCI World",
#         "VFISX": "UST Short (≈2Y)",
#         "VFITX": "UST Intermediate (≈5Y)",
#         "VUSTX": "UST Long (≈10–30Y)"
        
#     }

#     merged_df = pd.DataFrame()

#     for t, name in tickers.items():
#         df = yf.download(t, start="1980-01-01", progress=False)
#         merged_df[name] = df["Close"]
#     merged_df = merged_df.resample("M").last().reset_index().copy()
#     # merged_df["Bono 2Y"]  = bond_returns_from_yield(merged_df["UST 2Y Yield"], 2)
#     # merged_df["Bono 5Y"]  = bond_returns_from_yield(merged_df["UST 5Y Yield"], 5)
#     # merged_df["Bono 10Y"] = bond_returns_from_yield(merged_df["UST 10Y Yield"], 10)
#     # merged_df["Bono 30Y"] = bond_returns_from_yield(merged_df["UST 30Y Yield"], 30)
#     # merged_df = merged_df.drop(["UST 2Y Yield","UST 5Y Yield", "UST 10Y Yield", "UST 30Y Yield"], axis=1)
#     return(merged_df)

# merged_df = import_yahoo_fin()
# merged_df = merged_df.set_index("Date")

# merged_df = import_yahoo_fin()
# ret_df = calculate_returns(merged_df)
# cumulative_ret_df = calculate_cum_ret(ret_df)
# annual_ret_df = returns_annualized(ret_df)
# max_drawdown = maximum_drawdown(ret_df)
# min_horizon = calculate_minimum_investment_horizon(ret_df)


@st.cache_resource
def load_css_file(css_file_path):
    with open(css_file_path) as f:
        return st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css_file("main.css")

selected = option_menu(
    menu_title=None,
    options=["Situación actual", "Última actualización","Renta Variable", "Renta Fija", "Análisis","Ejemplos"], # "Crea tu cartera", "Histórico índices"
    icons=[None, None, None, None, None],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {
            "padding": "0px important!",
            "background-color": "#ffffff",
            "align": "center",
            "overflow": "hidden"
        },
        "icon": {
            "color": "#2F5D62",
            "font-size": "17px"
        },
        "nav-link": {
            "font-size": "17px",
            "text-align": "center",
            "font-weight": "bold",
            "color": "#2F5D62",
            "padding": "5px 10px",
            "background-color": "#f2f2f2",
            "margin": "0px 5px",
            "border-radius": "5px",
            "--hover-color": "#24494d",
        },
        "nav-link-selected": {
            "background-color": "#2F5D62",
            "color": "#ffffff",
            "border-radius": "5px"
        }
    }
)


# if selected == "Histórico índices":
#     st.title("Índices mundiales")
#     left, center, right = st.columns((1,1,1))
#     with left:
#         index = st.selectbox("**Select Index**", merged_df.set_index("Date").columns, index=merged_df.set_index("Date").columns.tolist().index("MSCI World"))
#     with center:
#         starting_date = st.date_input("**Starting date for simulation**", value=pd.to_datetime(cumulative_ret_df[index].dropna().index[0]))
#     with right:
#         ending_date = st.date_input("**Ending date for simulation**", value= pd.to_datetime(cumulative_ret_df[index].dropna().index[-1]))
#     left, right = st.columns((1,1))
#     risk_free_rate = 0.035
#     annual_ret_avg = calculate_cagr(cumulative_ret_df, index, starting_date, ending_date)
#     annual_vol_avg = calculate_annual_volatility(ret_df, index, starting_date, ending_date)
#     cumulative_final = cumulative_ret_df[index].values[-1]
#     max_drawdown_final = max_drawdown[index]["Max Drawdown"]
#     with left:
#         st.metric(label=f"CAGR (Cumulative annual growth rate)", value=f"{annual_ret_avg:.2%}")
#         st.metric(label=f"Maximum Drawdown", value=f"{max_drawdown_final:.2%}")
#         st.metric(label=f"Amount invested in {str(starting_date)}", value="10.000€")

#     with right:
#         st.metric(label=f"Ratio Sharpe", value=f"{calculate_ratio_sharpe(annual_ret_avg, risk_free_rate, annual_vol_avg)}")
#         st.metric(label=f"Volatilty (Standard Deviation)", value=f"{annual_vol_avg:.2%}")
#         st.metric(label=f"Net asset value in {str(ending_date)}", value=f"{calculate_net_asset_value(ret_df, index, starting_date, ending_date)}€")

#     st.plotly_chart(plot_line_chart(cumulative_ret_df, index))
#     st.subheader("Historic annual returns (%)")
#     st.markdown(annual_ret_df[annual_ret_df.index>2007][[index]].T.to_html(), unsafe_allow_html=True)
#     @st.cache_resource
#     def plot_annual_returns(df, index_column, start_year=2007):
#         filtered_df = df[df.index > start_year][[index_column]]

#         filtered_df = filtered_df.reset_index()

#         fig = px.bar(filtered_df, x='Year', y=index_column, 
#                     title=f"Annual Returns for {index_column})")
#         return fig
#     @st.cache_resource
#     def plot_annual_returns(df, index_column, start_year=1999):
#         filtered_df = df[df.index > start_year][[index_column]]
#         filtered_df = filtered_df.reset_index()
#         filtered_df['Color'] = filtered_df[index_column].apply(lambda x: 'Positive' if x >= 0 else 'Negative')
#         fig = px.bar(filtered_df, x='Year', y=index_column, 
#                     title=f"Annual Returns for {index_column}",
#                     color='Color', 
#                     color_discrete_map={'Positive': 'green', 'Negative': 'red'})
#         fig.update_layout(showlegend=False)
#         return fig
#     st.plotly_chart(plot_annual_returns(annual_ret_df, index))
#     @st.cache_resource
#     def plot_returns_histogram(returns_df, type):
#         if type=="annual":
#             bins = np.linspace(-50, 50, 11)
#         if type=="monthly":
#             bins = np.linspace(-20, 20, 6)
        
#         positive_returns = returns_df[returns_df >= 0]
#         negative_returns = returns_df[returns_df < 0]

#         bin_width = bins[1] - bins[0]

#         fig = go.Figure()
        
#         fig.add_trace(go.Histogram(
#             x=positive_returns,
#             nbinsx=5,
#             marker_color='green',
#             name='Positive Returns',
#             opacity=0.7,
#             xbins=dict(start=-50, end=50, size=bin_width * 0.25)
#         ))
        
#         fig.add_trace(go.Histogram(
#             x=negative_returns,
#             nbinsx=5,
#             marker_color='red',
#             name='Negative Returns',
#             opacity=0.7,
#             xbins=dict(start=-50, end=50, size=bin_width * 0.25)
#         ))
#         if type=="annual":
#             fig.update_layout(

#                 title='Histogram of Annual Returns',
#                 xaxis_title='Annual return',
#                 yaxis_title='Number of years',
#                 barmode='overlay',
#                 xaxis=dict(range=[-50, 50]),
#                 legend=dict(x=0.01, y=0.99),
#                 dragmode=False
#             )
#         else:
#             fig.update_layout(

#                 title='Histogram of Monthly Returns',
#                 xaxis_title='Monthly returns',
#                 yaxis_title='Number of months',
#                 barmode='group',
#                 xaxis=dict(range=[-30, 30]),
#                 legend=dict(x=0.01, y=0.99),
#                 dragmode=False
#             )
#         return(fig)
#     left, right = st.columns((1,1))
#     with left:
#         st.plotly_chart(plot_returns_histogram(ret_df[index].mul(100), "monthly"))
#     with right:
#         st.plotly_chart(plot_returns_histogram(annual_ret_df[index], "annual"))
#     st.subheader("Probability of a profit vs years invested")
#     st.bar_chart(min_horizon[index].mul(100))

#     st.title("Histórico de combinaciones de índices mundiales")
#     if 'rows' not in st.session_state:
#         st.session_state.rows = 1
#     def add_row():
#         st.session_state.rows += 1
#     st.button('Añade otro índice a tu portafolio:', on_click=add_row)
#     weights_l = []
#     indices_l = []
#     for i in range(st.session_state.rows):
#         left, right = st.columns((4, 1))
#         with left:
#             index = st.selectbox("", merged_df.set_index("Date").columns, placeholder="Choose an option", key=f"index_{i}")
#             indices_l.append(index)
#         with right:
#             weight = st.number_input("", min_value=0, max_value=100, key=f"weight_{i}")
#             weights_l.append(weight)
#     left, right = st.columns((1,1))
#     with left:
#         starting_date = st.date_input("**Fecha de inicio de la simulación**", value=pd.to_datetime(ret_df[index].dropna().index[0]))
#     with right:
#         ending_date = st.date_input("**Fecha final de simulación**", value= pd.to_datetime(ret_df[index].dropna().index[-1]))
#     weights_l = [x/100 for x in weights_l]
#     ret_portfolio_df = portfolio_returns(ret_df[indices_l], weights_l)
#     cumulative_ret_portfolio_df = calculate_cum_ret(ret_portfolio_df)
#     cumulative_ret_benchmark_df = calculate_cum_ret(ret_df[ret_df.index>=cumulative_ret_portfolio_df.index[0]])
#     annual_ret_portfolio_df = returns_annualized(ret_portfolio_df)
#     annual_vol_portfolio_avg = calculate_annual_volatility(ret_portfolio_df, "Portfolio", starting_date, ending_date)
#     annual_ret_portfolio_avg = calculate_cagr(cumulative_ret_portfolio_df, "Portfolio", starting_date, ending_date)
#     min_horizon_portfolio = calculate_minimum_investment_horizon(ret_portfolio_df)
#     max_drawdown_portfolio = maximum_drawdown(ret_portfolio_df)
#     max_drawdown_final = max_drawdown_portfolio["Portfolio"]["Max Drawdown"]
#     risk_free_rate = 0.035
#     left, right = st.columns((1,1))
#     with left:
#         st.metric(label=f"CAGR (Cumulative annual growth rate)", value=f"{annual_ret_portfolio_avg:.2%}")
#         st.metric(label=f"Maximum Drawdown", value=f"{max_drawdown_final:.2%}")
#         st.metric(label=f"Amount invested in {str(starting_date)}", value="10.000€")

#     with right:
#         st.metric(label=f"Ratio Sharpe", value=f"{calculate_ratio_sharpe(annual_ret_portfolio_avg, risk_free_rate, annual_vol_portfolio_avg)}")
#         st.metric(label=f"Volatilty (Standard Deviation)", value=f"{annual_vol_portfolio_avg:.2%}")
#         st.metric(label=f"Net asset value in {str(ending_date)}", value=f"""{calculate_net_asset_value(ret_portfolio_df, "Portfolio", starting_date, ending_date)}€""")

#     benchmark = st.selectbox("Select a Benchmark:", merged_df.set_index("Date").columns.tolist())
#     benchmark_plot_df = pd.merge(cumulative_ret_portfolio_df, cumulative_ret_benchmark_df[benchmark], on="Date", how="left")
#     st.plotly_chart(plot_line_chart(benchmark_plot_df.mul(100)))
    # left, right = st.columns((1,1))
    # risk_free_rate = 0.035
    # annual_ret_avg = calculate_cagr(cumulative_ret_df, benchmark, starting_date, ending_date)
    # annual_vol_avg = calculate_annual_volatility(ret_df, benchmark, starting_date, ending_date)
    # cumulative_final = cumulative_ret_df[benchmark].values[-1]
    # max_drawdown_final = max_drawdown[benchmark]["Max Drawdown"]
    # with left:
    #     st.metric(label=f"CAGR (Cumulative annual growth rate)", value=f"{annual_ret_avg:.2%}")
    #     st.metric(label=f"Maximum Drawdown", value=f"{max_drawdown_final:.2%}")
    #     st.metric(label=f"Amount invested in {str(starting_date)}", value="10.000€")

    # with right:
    #     st.metric(label=f"Ratio Sharpe", value=f"{calculate_ratio_sharpe(annual_ret_avg, risk_free_rate, annual_vol_avg)}")
    #     st.metric(label=f"Volatilty (Standard Deviation)", value=f"{annual_vol_avg:.2%}")
    #     st.metric(label=f"Net asset value in {str(ending_date)}", value=f"{calculate_net_asset_value(ret_df, benchmark, starting_date, ending_date)}€")
###########################################################################################################################################################################
if selected=="Situación actual":

    cartera = backtest_cartera(df_final_ff, posiciones_df, False).dropna()
    cartera_100 = cartera.div(cartera.iloc[0]).mul(100).to_frame("Cartera_actual")
    dd = (cartera / cartera.cummax() - 1).to_frame("Drawdown")
    current_dd = dd.iloc[-1, 0]

    pos_ini, pos_act, var_eur = metrics_df["Posición_ini"].sum(), metrics_df["Posición"].sum(), metrics_df["Var"].sum()
    var_pct = var_eur / pos_ini if pos_ini else np.nan
    ret = lambda n: (cartera.iloc[-1] / cartera.iloc[-n] - 1) if len(cartera) > n else None
    r1, r3, r6, r12 = ret(30), ret(90), ret(180), ret(360)

    for col, (lab, val) in zip(st.columns(4), [
        ("Posición inicial", f"{pos_ini:,.1f} €"),
        ("Posición actual", f"{pos_act:,.1f} €"),
        ("Variación", f"{var_eur:,.1f} €"),
        ("Variación %", f"{var_pct:.1%}" if pd.notna(var_pct) else "—")
    ]): col.metric(lab, val)

    for col, (lab, val) in zip(st.columns(5), [
        ("Current DD", f"{current_dd:.1%}"),
        ("1M", f"{r1:.1%}" if r1 is not None else "—"),
        ("3M", f"{r3:.1%}" if r3 is not None else "—"),
        ("6M", f"{r6:.1%}" if r6 is not None else "—"),
        ("12M", f"{r12:.1%}" if r12 is not None else "—")
    ]): col.metric(lab, val)

    st.subheader("Evolución cartera (EUR)")
    parts = metrics_df.set_index("isin")["Participaciones"].dropna().astype(float)
    eur = df_final_ff[[c for c in df_final_ff.columns if c in parts.index]].dropna(how="all").mul(parts, axis=1).sum(axis=1).dropna()
    plot_df_lines_aux(eur[eur.index >= pd.to_datetime("2021-05-01")].to_frame("Cartera_EUR"), "EUR")

    st.subheader("Posiciones actuales")

    pos_df = metrics_df_final[["nombre_fondo", "Posición", "Var", "% Var"]].copy()
    pos_df["nombre_fondo"] = pos_df["nombre_fondo"].astype(str).str.strip()
    pos_df = pos_df[pos_df["nombre_fondo"].ne("") & pos_df["nombre_fondo"].ne("nan")]
    pos_df["Posición"] = pd.to_numeric(pos_df["Posición"], errors="coerce").round(1)
    pos_df["Var"] = pd.to_numeric(pos_df["Var"], errors="coerce").round(1)
    pos_df["% Var"] = pd.to_numeric(pos_df["% Var"], errors="coerce").round(1)

    fig = px.treemap(
        pos_df,
        path=[px.Constant("Cartera"), "nombre_fondo"],
        values="Posición",
        color="% Var",
        color_continuous_scale=["#8b0000","#ff4d4d","#f5f5f5","#66cc66","#006400"],
        color_continuous_midpoint=0,
        custom_data=["Posición", "Var", "% Var"]
    )

    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{customdata[0]:,.1f} €<br>%{customdata[1]:+,.1f} € (%{customdata[2]:+.1f}%)",
        textinfo="text",
        hovertemplate="<b>%{label}</b><br>Peso: %{percentParent:.1%}<br>Posición: %{customdata[0]:,.1f} €<br>Variación: %{customdata[1]:+,.1f} € (%{customdata[2]:+.1f}%)<extra></extra>",
        root_color="white"
    )

    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=700, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    show_positions_eur(metrics_df, df_final_ff)

    c1, c2 = st.columns(2)

    with c1:
        fig = px.treemap(
            metrics_df_final.groupby("categoria", as_index=False)["Posición"].sum(),
            path=[px.Constant("Cartera"), "categoria"],
            values="Posición",
            custom_data=["Posición"]
        )

        fig.update_traces(
            texttemplate="<b>%{label}</b><br>%{customdata[0]:,.1f} €<br>%{percentParent:.1%}",
            textinfo="text",
            hovertemplate="<b>%{label}</b><br>Posición: %{customdata[0]:,.1f} €<br>Peso: %{percentParent:.1%}<extra></extra>",
            root_color="white"
        )

        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=500)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(
            metrics_df_final.groupby("categoria_riesgo", as_index=False)["Posición"].sum().sort_values("Posición"),
            x="Posición",
            y="categoria_riesgo",
            orientation="h",
            text="Posición"
        )

        fig.update_traces(
            texttemplate="%{text:,.1f} €",
            hovertemplate="<b>%{y}</b><br>Posición: %{x:,.1f} €<extra></extra>"
        )

        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=500, xaxis_title=None, yaxis_title=None, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    plot_df_lines(cartera_100, "Base 100")
    plot_df_lines(dd.mul(100), "Drawdown (%)")

    show_backtest_report("", backtest_cartera(df_final_ff, posiciones_df, True))


if selected=="Renta Variable":
    st.subheader("Evolución cartera (EUR)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Posición inicial", round(metrics_df[metrics_df["categoria"]=="Renta variable"]["Posición_ini"].sum(),2))
    c2.metric("Posición actual", round(metrics_df[metrics_df["categoria"]=="Renta variable"]["Posición"].sum(),2))
    c3.metric("Variación", round(metrics_df[metrics_df["categoria"]=="Renta variable"]["Var"].sum(),2))
    posiciones_df = metrics_df.loc[metrics_df["categoria"]=="Renta variable"].set_index("isin")["Perc_posicion"].dropna() / 100
    posiciones_df = (metrics_df.loc[metrics_df["categoria"]=="Renta variable"]
                .set_index("isin")["Perc_posicion"]
                .dropna()
                .astype(float)
                .pipe(lambda s: (s/100) / (s/100).sum()))

    parts = metrics_df[metrics_df["categoria"]=="Renta variable"].set_index("isin")["Participaciones"].dropna().astype(float)
    common = [c for c in df_final_ff.columns if c in parts.index]
    nav = df_final_ff[common].dropna(how="all")
    eur = nav.mul(parts[common], axis=1).sum(axis=1).dropna()
    plot_df_lines_aux(eur.to_frame("Cartera_EUR"), "EUR")
    cartera = backtest_cartera(df_final_ff, posiciones_df, False).dropna()
    cartera_100 = cartera.div(cartera.iloc[0]).mul(100).to_frame("Cartera_actual")
    current_dd = cartera / cartera.cummax() - 1
    current_dd = current_dd.iloc[-1]
    def retornos(df, ticker, year_month, n_year_month):
        serie = df[ticker].dropna()
        hoy = serie.index.max()
        if year_month=="year":
            hace_xy = hoy - pd.DateOffset(years=n_year_month)
            valor_hoy = serie.loc[hoy]
            valor_hace_xy = serie.loc[serie.index <= hace_xy].iloc[-1]
            retorno_xy_anualizado = (valor_hoy / valor_hace_xy) ** (1/n_year_month) - 1
        if year_month=="month":
            hace_xy = hoy - pd.DateOffset(months=n_year_month)
            valor_hoy = serie.loc[hoy]
            valor_hace_xy = serie.loc[serie.index <= hace_xy].iloc[-1]
            retorno_xy_anualizado = (valor_hoy / valor_hace_xy) - 1
        return(retorno_xy_anualizado)
    def volatilidad(df, ticker, n):
        s = df[ticker].dropna().sort_index()
        m = s.resample("M").last()
        hoy = m.index.max()
        inicio = hoy - pd.DateOffset(years=n)
        m = m.loc[m.index >= inicio]
        r = m.pct_change().dropna()
        if r.empty:
            return np.nan
        return float(r.std(ddof=1) * np.sqrt(12))
    
    ret = lambda n: (cartera.iloc[-1] / cartera.iloc[-n] - 1) if len(cartera) > n else None
    r1, r3, r6, r12 = ret(30), ret(90), ret(180), ret(360)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current DD", f"{current_dd:.2%}")
    c2.metric("1M", f"{r1:.2%}" if r1 is not None else "—")
    c3.metric("3M", f"{r3:.2%}" if r3 is not None else "—")
    c4.metric("6M", f"{r6:.2%}" if r6 is not None else "—")
    c5.metric("12M", f"{r12:.2%}" if r12 is not None else "—")
    left, center, right = st.columns([0.1,1,0.1])
    with center:
        plot_treemap_cartera(
            metrics_df_final[metrics_df_final["categoria"]=="Renta variable"][["nombre_fondo","Posición","Var", "% Var"]],
            "Renta Variable"
        )
    show_backtest_report("", backtest_cartera(df_final_ff, posiciones_df, True))



if selected=="Renta Fija":
    st.subheader("Evolución cartera (EUR)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Posición inicial", round(metrics_df[metrics_df["categoria"]!="Renta variable"]["Posición_ini"].sum(),2))
    c2.metric("Posición actual", round(metrics_df[metrics_df["categoria"]!="Renta variable"]["Posición"].sum(),2))
    c3.metric("Variación", round(metrics_df[metrics_df["categoria"]!="Renta variable"]["Var"].sum(),2))
    posiciones_df = metrics_df.loc[metrics_df["categoria"]!="Renta variable"].set_index("isin")["Perc_posicion"].dropna() / 100
    posiciones_df = (metrics_df.loc[metrics_df["categoria"]!="Renta variable"]
                .set_index("isin")["Perc_posicion"]
                .dropna()
                .astype(float)
                .pipe(lambda s: (s/100) / (s/100).sum()))

    parts = metrics_df[metrics_df["categoria"]!="Renta variable"].set_index("isin")["Participaciones"].dropna().astype(float)
    common = [c for c in df_final_ff.columns if c in parts.index]
    nav = df_final_ff[common].dropna(how="all")
    eur = nav.mul(parts[common], axis=1).sum(axis=1).dropna()
    plot_df_lines_aux(eur[eur.index>=pd.to_datetime("2020-03-01")].to_frame("Cartera_EUR"), "EUR")
    cartera = backtest_cartera(df_final_ff, posiciones_df, False).dropna()
    cartera_100 = cartera.div(cartera.iloc[0]).mul(100).to_frame("Cartera_actual")
    current_dd = cartera / cartera.cummax() - 1
    current_dd = current_dd.iloc[-1]
    def retornos(df, ticker, year_month, n_year_month):
        serie = df[ticker].dropna()
        hoy = serie.index.max()
        if year_month=="year":
            hace_xy = hoy - pd.DateOffset(years=n_year_month)
            valor_hoy = serie.loc[hoy]
            valor_hace_xy = serie.loc[serie.index <= hace_xy].iloc[-1]
            retorno_xy_anualizado = (valor_hoy / valor_hace_xy) ** (1/n_year_month) - 1
        if year_month=="month":
            hace_xy = hoy - pd.DateOffset(months=n_year_month)
            valor_hoy = serie.loc[hoy]
            valor_hace_xy = serie.loc[serie.index <= hace_xy].iloc[-1]
            retorno_xy_anualizado = (valor_hoy / valor_hace_xy) - 1
        return(retorno_xy_anualizado)
    def volatilidad(df, ticker, n):
        s = df[ticker].dropna().sort_index()
        m = s.resample("M").last()
        hoy = m.index.max()
        inicio = hoy - pd.DateOffset(years=n)
        m = m.loc[m.index >= inicio]
        r = m.pct_change().dropna()
        if r.empty:
            return np.nan
        return float(r.std(ddof=1) * np.sqrt(12))
    
    ret = lambda n: (cartera.iloc[-1] / cartera.iloc[-n] - 1) if len(cartera) > n else None
    r1, r3, r6, r12 = ret(30), ret(90), ret(180), ret(360)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current DD", f"{current_dd:.2%}")
    c2.metric("1M", f"{r1:.2%}" if r1 is not None else "—")
    c3.metric("3M", f"{r3:.2%}" if r3 is not None else "—")
    c4.metric("6M", f"{r6:.2%}" if r6 is not None else "—")
    c5.metric("12M", f"{r12:.2%}" if r12 is not None else "—")
    left, center, right = st.columns([0.1,1,0.1])
    with center:
        plot_treemap_cartera(
            metrics_df_final[metrics_df_final["categoria"]!="Renta variable"][["nombre_fondo","Posición","Var", "% Var"]],
            "Renta Fija - Alternativos - Monetarios"
        )
    show_backtest_report("", backtest_cartera(df_final_ff, posiciones_df, True))
if selected=="Análisis":
    st.title("Selección para el análisis")
    opciones_universo = sorted(metrics_df_final["categoria"].dropna().unique().tolist())
    top_left, left, center, right = st.columns((1,1,1,1))
    with top_left:
        universo_sel = st.selectbox("**Universo**", opciones_universo)
    if universo_sel == "Cartera actual":
        universe = metrics_df_final[
            metrics_df_final["isin"].isin(carteras_x["isin"].to_list())
        ].copy()
    else:
        universe = metrics_df_final[
            metrics_df_final["categoria"] == universo_sel
        ].copy()

    with left:
        index = st.selectbox("**Select Index**", universe["nombre_fondo_isin"].dropna())
    with center:
        starting_date = st.date_input("**Starting date for simulation**", value=pd.to_datetime("2015-01-01"))
    with right:
        ending_date = st.date_input("**Ending date for simulation**", value= pd.to_datetime(datetime.now().strftime("%d/%m/%Y")))

    # --- 1) fila del fondo ---
    row = (universe.loc[universe["nombre_fondo_isin"] == index]
        .iloc[0])

    # Helpers de formato
    def fmt_pct(x):
        return "—" if pd.isna(x) else f"{x*100:.2f}%"

    def fmt_num(x, nd=2):
        return "—" if pd.isna(x) else f"{x:.{nd}f}"

    def fmt_eur(x):
        return "—" if pd.isna(x) else f"{x:,.0f} €"

    def fmt_days(x):
        return "—" if pd.isna(x) else f"{int(x):,} días"

    st.subheader(f"📌 {row['nombre_fondo_isin']}")

    # --- 2) Bloque principal de métricas (rendimiento/riesgo) ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CAGR", fmt_pct(row.get("cagr")))
    c2.metric("Volatilidad anual", fmt_pct(row.get("vol_anual")))
    c3.metric("Sharpe", fmt_num(row.get("sharpe"), 2))
    c4.metric("Último VL", fmt_num(row.get("ultimo_vl"), 2))

    # --- 3) Drawdowns + señales ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current drawdown", fmt_pct(row.get("current_drawdown")))
    c2.metric("Max drawdown", fmt_pct(row.get("max_drawdown")))
    c3.metric("Z-score (63d)", fmt_num(row.get("z_score3meses"), 2))
    c4.metric("Días hasta recovery (DD#1)", fmt_days(row.get("days_to_recover")))

    # --- 5) Señales / consenso ---
    st.markdown("### Señales (técnico)")
    sig_cols = ["zscore", "percentil_52w", "dist_ma63", "drawdown", "consenso"]
    sig = {k: (row.get(k) if pd.notna(row.get(k)) else "—") for k in sig_cols}

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Zscore", sig["zscore"])
    s2.metric("52w", sig["percentil_52w"])
    s3.metric("MA63", sig["dist_ma63"])
    s4.metric("DD", sig["drawdown"])
    s5.metric("Consenso", sig["consenso"])

    plot_df_lines(df_final_ff[universe.loc[universe["nombre_fondo_isin"] == index]["isin"].values[0]])

    st.title("Comparables de señales y rendimientos")

    cols_metrics = [
        "ret_month_1", "ret_month_3", "ret_month_6",
        "ret_year_1", "ret_year_3", "ret_year_5", "ret_year_7","ret_year_10",
        "vol_year_1", "vol_year_3", "vol_year_5", "vol_year_7","vol_year_10", "sharpe_3", "sharpe_5", "sharpe_10"
    ]

    # CAMBIO MÍNIMO: aquí ya usamos universe
    view = universe.copy()

    view = view[
        ["nombre_fondo_isin", "current_drawdown", "max_drawdown", "days_to_recover", "z_score3meses"]+cols_metrics
    ].rename(columns={
        "nombre_fondo_isin": "Fondo",
        "current_drawdown": "DD",
        "max_drawdown": "Max Drawdown",
        "days_to_recover": "Dias para recuperar"
    })

    def heatmap_col(col):
        m = col.abs().max()
        out = []

        for v in col:
            if pd.isna(v):
                out.append("background-color: #f2f2f2; color: #888888;")

            elif v > 0:
                a = v / m if m else 0
                g = int(255 - 120 * a)
                out.append(f"background-color: rgb({g}, 255, {g}); color: #1a1a1a;")

            elif v < 0:
                a = abs(v) / m if m else 0
                r = int(255 - 120 * a)
                out.append(f"background-color: rgb(255, {r}, {r}); color: #1a1a1a;")

            else:
                out.append("background-color: white; color: #1a1a1a;")

        return out

    def pintar_tabla(df, titulo):
        nums = df.select_dtypes("number").columns
        styled = df.style.format("{:.3f}", subset=nums).apply(heatmap_col, subset=nums, axis=0)
        st.markdown(f"### {titulo}")
        st.write(styled)

    view_1 = view[[
        "Fondo", "ret_month_1", "ret_month_3", "ret_month_6", "ret_year_1", "vol_year_1"
    ]].copy()

    view_2 = view[[
        "Fondo", "ret_year_3", "ret_year_5", "ret_year_7", "ret_year_10",
        "vol_year_3", "vol_year_5", "vol_year_7", "vol_year_10"
    ]].copy()

    view_3 = view[[
        "Fondo",
        "z_score3meses", "DD", "Max Drawdown", "Dias para recuperar",
        "sharpe_3", "sharpe_5", "sharpe_10"
    ]].copy()

    pintar_tabla(view_1, "Retornos y volatilidades inferiores a 3 años")
    pintar_tabla(view_2, "Retornos y volatilidades de 3 años o más")
    pintar_tabla(view_3, "Ratios e indicadores")

    def top_bottom_block(df, ret_col, vol_col, n=20, name_col="nombre_fondo"):
        d = df[[name_col, ret_col, vol_col]].dropna().sort_values(ret_col, ascending=False)

        top = d.head(n)
        bot = d.tail(n).sort_values(ret_col)

        tb = pd.concat([top, bot])

        vmin = float(tb[[ret_col, vol_col]].min().min())
        vmax = float(tb[[ret_col, vol_col]].max().max()) + 0.1

        fig = px.scatter(
            tb,
            x=vol_col,
            y=ret_col,
            hover_name=name_col,
            text=name_col
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(
            height=520,
            xaxis_title="Volatilidad",
            yaxis_title="Retorno"
        )
        fig.update_xaxes(range=[vmin, 0.3])
        fig.update_yaxes(range=[vmin, vmax])
        fig.add_shape(
            type="line",
            x0=vmin, y0=vmin,
            x1=vmax, y1=vmax,
            line=dict(color="grey", dash="dash", width=2),
        )

        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(f"### Top {n}")
            st.dataframe(
                top[[name_col, ret_col, vol_col]].rename(columns={name_col: "Fondo"}),
                use_container_width=True,
                hide_index=True,
                column_config={
                    ret_col: st.column_config.NumberColumn("Retorno", format="%.4f"),
                    vol_col: st.column_config.NumberColumn("Volatilidad", format="%.4f"),
                }
            )

        with c2:
            st.markdown(f"### Bottom {n}")
            st.dataframe(
                bot[[name_col, ret_col, vol_col]].rename(columns={name_col: "Fondo"}),
                use_container_width=True,
                hide_index=True,
                column_config={
                    ret_col: st.column_config.NumberColumn("Retorno", format="%.4f"),
                    vol_col: st.column_config.NumberColumn("Volatilidad", format="%.4f"),
                }
            )

    def correlation_block(prices_df, isins, years, name_map):
        months = years * 12

        corr = (
            prices_df[isins]
            .resample("M").last()
            .pct_change()
            .tail(months)
            .corr()
            .round(2)
        )

        corr = corr.rename(index=name_map, columns=name_map)

        st.markdown(f"### Matriz de correlaciones ({years} año{'s' if years > 1 else ''})")

        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdYlGn",
            zmin=-1,
            zmax=1,
            aspect="auto"
        )
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

    fondos_cartera = [c for c in universe["isin"].dropna().tolist() if c in df_final_ff.columns]
    name_map = (
        universe[["isin", "nombre_fondo"]]
        .dropna()
        .drop_duplicates("isin")
        .set_index("isin")["nombre_fondo"]
        .to_dict()
    )

    tabs = st.tabs(["1 año", "3 años", "5 años", "7 años"])

    with tabs[0]:
        top_bottom_block(universe, "ret_year_1", "vol_year_1", n=20)
        correlation_block(df_final_ff, fondos_cartera, 1, name_map)

    with tabs[1]:
        top_bottom_block(universe, "ret_year_3", "vol_year_3", n=20)
        correlation_block(df_final_ff, fondos_cartera, 3, name_map)

    with tabs[2]:
        top_bottom_block(universe, "ret_year_5", "vol_year_5", n=20)
        correlation_block(df_final_ff, fondos_cartera, 5, name_map)

    with tabs[3]:
        top_bottom_block(universe, "ret_year_7", "vol_year_7", n=20)
        correlation_block(df_final_ff, fondos_cartera, 7, name_map)

    st.title("Correlograma completo (5 años)")
    cats = [
        "Renta variable",
        "Renta Fija medio plazo",
        "Renta Fija largo plazo",
        "Alternativos"
    ]
    isins = metrics_df_final.loc[
        (metrics_df_final["categoria"].isin(cats)) &
        (metrics_df_final["isin"].isin(carteras_x["isin"])),
        "isin"
    ].dropna().unique().tolist()
    isins = [i for i in isins if i in df_final_ff.columns]
    rets = (
        df_final_ff[isins]
        .resample("M").last()
        .pct_change()
        .tail(60)
    )
    rets = rets.dropna(axis=1, thresh=48)
    corr = rets.corr().round(2)

    # Usar nombres reales
    name_map = dict(zip(nombre_fondo["isin"], nombre_fondo["nombre_fondo"]))
    corr = corr.rename(index=name_map, columns=name_map)

    # Heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            zmin=-1,
            zmax=1,
            colorscale="RdYlGn",
            hovertemplate="X: %{x}<br>Y: %{y}<br>Corr: %{z:.2f}<extra></extra>"
        )
    )

    fig.update_layout(height=1000, width=1000)

    st.plotly_chart(fig, use_container_width=True)

    st.title("Comparativa histórica de fondos")

    def plot_comparativa_fondos(prices_df, isins, periodo="todo", name_map=None, titulo=""):
        isins = [i for i in isins if i in prices_df.columns]
        if not isins:
            st.info("No hay fondos para mostrar.")
            return

        data = prices_df[isins].copy()

        if periodo == "ytd":
            data = data[data.index >= pd.Timestamp(datetime.now().year, 1, 1)]
        elif periodo == "1y":
            data = data.last("1Y")
        elif periodo == "3y":
            data = data.last("3Y")
        elif periodo == "5y":
            data = data.last("5Y")

        data = data.ffill()

        # base 100 individual: cada fondo empieza cuando tiene primer dato válido
        base100 = pd.DataFrame(index=data.index)
        for col in data.columns:
            s = data[col].dropna()
            if len(s) > 1:
                base100[col] = data[col] / s.iloc[0] * 100

        base100 = base100.dropna(axis=1, how="all")

        if base100.empty:
            st.info("No hay histórico suficiente para este periodo.")
            return

        if name_map is not None:
            base100 = base100.rename(columns=name_map)

        fig = px.line(base100, title=titulo)
        fig.update_layout(
            height=600,
            xaxis_title="Fecha",
            yaxis_title="Base 100"
        )

        st.plotly_chart(fig, use_container_width=True)

    fondos_universo = [i for i in universe["isin"].dropna().unique().tolist() if i in df_final_ff.columns]
    name_map = (
        universe[["isin", "nombre_fondo"]]
        .dropna()
        .drop_duplicates("isin")
        .set_index("isin")["nombre_fondo"]
        .to_dict()
    )

    tabs_comp = st.tabs(["YTD", "1 año", "3 años", "5 años", "Todo"])

    with tabs_comp[0]:
        plot_comparativa_fondos(df_final_ff, fondos_universo, periodo="ytd", name_map=name_map, titulo="Comparativa YTD")

    with tabs_comp[1]:
        plot_comparativa_fondos(df_final_ff, fondos_universo, periodo="1y", name_map=name_map, titulo="Comparativa 1 año")

    with tabs_comp[2]:
        plot_comparativa_fondos(df_final_ff, fondos_universo, periodo="3y", name_map=name_map, titulo="Comparativa 3 años")

    with tabs_comp[3]:
        plot_comparativa_fondos(df_final_ff, fondos_universo, periodo="5y", name_map=name_map, titulo="Comparativa 5 años")

    with tabs_comp[4]:
        plot_comparativa_fondos(df_final_ff, fondos_universo, periodo="todo", name_map=name_map, titulo="Comparativa histórica completa")


if selected=="Ejemplos":
    posiciones_df = metrics_df[metrics_df["inicio_hist"] <= "2020-02-08"].set_index("isin")["Perc_posicion"].dropna() / 100 # [metrics_df["inicio_hist"] <= "2020-02-08"]
    cartera_cols = (carteras_x.columns.to_series().str.extract(r'^(Cartera_\d+)$')[0].dropna().tolist())
    cartera_cols = sorted(cartera_cols, key=lambda c: int(c.split("_")[1]))
    carteras_comp = pd.concat(
        {
            "Cartera_actual": backtest_cartera(df_final_ff, posiciones_df, False),
            **{
                c: backtest_cartera(
                    df_final_ff,
                    carteras_x.set_index("isin")[c].dropna(),
                    False
                )
                for c in cartera_cols
            }
        },
        axis=1
    ).dropna()


    carteras_comp= carteras_comp.div(carteras_comp.apply(lambda s: s.iloc[0]), axis=1).mul(100)
    plot_df_lines(carteras_comp)

    w_x = carteras_x.set_index("isin")[cartera_cols].apply(pd.to_numeric, errors="coerce")
    w_x = w_x.div(w_x.sum(axis=0), axis=1)  # quita esta línea si ya vienen normalizadas
    show_backtest_report("Cartera actual", backtest_cartera(df_final_ff, posiciones_df, True))
    show_positions("Cartera actual", posiciones_df, metrics_df)
    for c in cartera_cols:
        show_backtest_report(c, backtest_cartera(df_final_ff, carteras_x.set_index("isin")[c].dropna(), True))    # Carteras del excel (asumo que sus pesos suman 1; si no, normalizo)
        show_positions(c, w_x[c].dropna(), metrics_df)


if selected == "Crea tu cartera":
    st.title("Carteras")
    st.write("Crea tu cartera asignando el peso de cada fondo de inversión en tu cartera:")

    # ---------- Universo de fondos ----------
    funds = sorted(metrics_df_final["nombre_fondo"].dropna().unique())
    name2isin = (metrics_df_final.dropna(subset=["nombre_fondo", "isin"])
                 .drop_duplicates("nombre_fondo")
                 .set_index("nombre_fondo")["isin"])

    # ---------- Estado (2 filas por defecto) ----------
    if "pf_rows" not in st.session_state:
        st.session_state.pf_rows = 2

    def add_row(): st.session_state.pf_rows += 1
    def remove_row(): st.session_state.pf_rows = max(2, st.session_state.pf_rows - 1)

    c1, c2, _ = st.columns([1, 1, 4])
    c1.button("➕ Add fund", on_click=add_row)
    c2.button("➖ Remove fund", on_click=remove_row)

    with st.form("Crea tu cartera"):
        picks, weights = [], []

        for i in range(st.session_state.pf_rows):
            a, b = st.columns([4, 1])
            with a:
                f = st.selectbox(
                    "", [""] + funds,
                    key=f"pf_f_{i}",
                    format_func=lambda x: "Select a fund…" if x == "" else x
                )
            with b:
                w_i = st.number_input("", 0.0, 100.0, 0.0, 1.0, key=f"pf_w_{i}")

            picks.append(f)
            weights.append(w_i)

        submitted = st.form_submit_button("✅ Submit")

    if not submitted:
        st.info("Selecciona los fondos y las ponderaciones, y luego haz clic en Enviar para ejecutar los cálculos.")
        st.stop()

    pf = [(f, w_i) for f, w_i in zip(picks, weights) if f and w_i > 0]
    if not pf:
        st.warning("Selecciona al menos un fondo y asígnale una ponderación > 0.")
        st.stop()

    w = pd.Series([x[1] for x in pf], index=[x[0] for x in pf]).groupby(level=0).sum()
    w = w / w.sum()

    pos = (w.mul(100).round(2).rename("Peso (%)").reset_index()
             .rename(columns={"index": "Fondo"})
             .sort_values("Peso (%)", ascending=False))

    st.subheader("Portfolio positions")
    st.dataframe(
        pos,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Peso (%)": st.column_config.ProgressColumn(
                "Peso (%)", min_value=0, max_value=100, format="%.2f"
            )
        },
    )
    weights_isin = w.rename(index=name2isin).dropna()
    out_user = backtest_cartera(df_final_ff, weights_isin, True)
    show_backtest_report("Cartera Seleccionada", out_user)
    cartera = backtest_cartera(df_final_ff, weights_isin, False).dropna()
    current_dd = (cartera / cartera.cummax() - 1).iloc[-1]

    def cagr(n):
        if len(cartera) <= n:
            return None
        years = n / 360          # misma convención que usas (360 ~ 1Y)
        return (cartera.iloc[-1] / cartera.iloc[-n]) ** (1 / years) - 1

    def vol(n):
        if len(cartera) <= n:
            return None
        return cartera.pct_change().tail(n).std() * np.sqrt(252)

    sharpe = cartera.pct_change().dropna().tail(252*5).mean() / cartera.pct_change().dropna().tail(252*5).std() * np.sqrt(252)

    r3  = cagr(90)     # 3M (CAGR trimestralizado)
    r6  = cagr(180)    # 6M
    r12 = cagr(360)    # 12M
    r36 = cagr(1080)   # 3Y
    r60 = cagr(1800)   # 5Y

    v3  = vol(90)
    v6  = vol(180)
    v12 = vol(360)
    v36 = vol(1080)
    v60 = vol(1800)

    # Fila 1: drawdown + CAGR
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Current DD", f"{current_dd:.2%}")
    c2.metric("3M CAGR",  f"{r3:.2%}"  if r3  is not None else "—")
    c3.metric("6M CAGR",  f"{r6:.2%}"  if r6  is not None else "—")
    c4.metric("12M CAGR", f"{r12:.2%}" if r12 is not None else "—")
    c5.metric("3Y CAGR",  f"{r36:.2%}" if r36 is not None else "—")
    c6.metric("5Y CAGR",  f"{r60:.2%}" if r60 is not None else "—")

    # Fila 2: volatilidad anualizada
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Sharpe 5Y", f"{sharpe:.2f}" if np.isfinite(sharpe) else "—")
    c2.metric("Vol 3M",  f"{v3:.2%}"  if v3  is not None else "—")
    c3.metric("Vol 6M",  f"{v6:.2%}"  if v6  is not None else "—")
    c4.metric("Vol 12M", f"{v12:.2%}" if v12 is not None else "—")
    c5.metric("Vol 3Y",  f"{v36:.2%}" if v36 is not None else "—")
    c6.metric("Vol 5Y",  f"{v60:.2%}" if v60 is not None else "—")
    plot_df_lines(cartera, "Base 100")

if selected=="Última actualización":

    st.title("Última actualización de valor liquidativo")

    upd = pd.DataFrame([
        {"nombre": r["nombre_fondo"], "categoria": r["categoria"], "fecha": s.index[-1], "var": s.iloc[-1]/s.iloc[-2]-1}
        for _, r in metrics_df_final[["isin","nombre_fondo","categoria"]].dropna().drop_duplicates("isin").iterrows()
        if r["isin"] in df_final.columns
        for s in [df_final[r["isin"]].dropna()]
        if len(s) > 1
    ])

    if upd.empty:
        st.warning("No hay datos disponibles")
        st.stop()

    upd["fecha_str"] = upd["fecha"].dt.strftime("%d/%m/%Y")
    top_fechas = upd["fecha"].value_counts().head(2).index.sort_values(ascending=False)
    upd_top = upd[upd["fecha"].isin(top_fechas)].assign(fecha_lbl=lambda x: x["fecha"].dt.strftime("%d/%m/%Y"))

    for c, (lab, val) in zip(st.columns(4), [
        ("Fondos", len(upd)),
        ("Fechas activas", upd["fecha"].nunique()),
        ("Última fecha común", top_fechas.max().strftime("%d/%m/%Y")),
        ("Variación media", f"{upd_top.loc[upd_top['fecha'].eq(top_fechas.max()), 'var'].mean():.2%}")
    ]): c.metric(lab, val)

    cats = [
        "Monetarios",
        "Renta fija corto plazo",
        "Renta Fija medio plazo",
        "Renta Fija largo plazo",
        "Alternativos",
        "Renta variable"
    ]

    for cat in cats:
        fondos = upd[upd["categoria"].eq(cat)].sort_values(["fecha","nombre"], ascending=[False,True])
        if fondos.empty:
            continue

        st.markdown(f"### {cat}")
        cols = st.columns(5)

        for i, r in enumerate(fondos.itertuples()):
            cols[i % 5].markdown(
                f"""
                <div style="border:1px solid #e5e7eb;border-radius:10px;padding:10px 12px;margin:4px 0;line-height:1.2;">
                    <div style="font-size:13px;font-weight:600;">{r.nombre}</div>
                    <div style="font-size:12px;color:#6b7280;">{r.fecha_str}</div>
                    <div style="font-size:14px;font-weight:700;color:{'#16a34a' if r.var>0 else '#dc2626' if r.var<0 else '#6b7280'};margin-top:4px;">{r.var:.2%}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.subheader("Comparación de las últimas fechas con más fondos actualizados")

    fig = px.bar(
        upd_top.sort_values(["fecha","var"]),
        x="var",
        y="nombre",
        color="fecha_lbl",
        orientation="h",
        barmode="group",
        hover_data={"fecha_lbl":True,"var":":.2%"},
        labels={"var":"","nombre":"","fecha_lbl":"Fecha"}
    )

    fig.update_layout(
        margin=dict(t=10,l=0,r=0,b=0),
        height=650,
        xaxis_title=None,
        yaxis_title=None
    )

    st.plotly_chart(fig, use_container_width=True)