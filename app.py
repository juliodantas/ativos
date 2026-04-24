"""
Dashboard de Análise de Ativos — Top 10 do Mês Anterior
========================================================
Aplicação Streamlit que usa yfinance para buscar ações do Ibovespa e principais
FIIs, analisar performance do mês anterior e sugerir os 10 melhores candidatos
para compra com base em análise probabilística (multi-fator).

Como executar:
    pip install -r requirements.txt
    streamlit run app.py

Autor: Julio Dantas (gerado com Claude)
Versão: 0.1 (primeira versão — iremos refinando)
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard de Ativos — Top 10 do Mês",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# UNIVERSO DE ATIVOS
# --------------------------------------------------------------------------
# Principais ações do Ibovespa (sufixo .SA para B3 no yfinance)
IBOV_TICKERS = [
    "PETR4.SA", "PETR3.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA",
    "B3SA3.SA", "WEGE3.SA", "RENT3.SA", "LREN3.SA", "SUZB3.SA", "GGBR4.SA",
    "CSNA3.SA", "USIM5.SA", "ELET3.SA", "ELET6.SA", "CMIG4.SA", "SBSP3.SA",
    "TAEE11.SA", "EGIE3.SA", "EQTL3.SA", "ENGI11.SA", "CPLE6.SA", "CPFE3.SA",
    "ITSA4.SA", "SANB11.SA", "BBAS3.SA", "BPAC11.SA", "BRFS3.SA", "JBSS3.SA",
    "MRFG3.SA", "BEEF3.SA", "NTCO3.SA", "CCRO3.SA", "AZUL4.SA", "RADL3.SA",
    "HAPV3.SA", "QUAL3.SA", "ASAI3.SA", "COGN3.SA", "YDUQ3.SA", "GOAU4.SA",
    "BRKM5.SA", "CYRE3.SA", "MRVE3.SA", "EZTC3.SA", "RDOR3.SA", "FLRY3.SA",
    "HYPE3.SA", "DXCO3.SA", "KLBN11.SA", "EMBR3.SA", "TIMS3.SA", "VIVT3.SA",
    "MULT3.SA", "CVCB3.SA", "PRIO3.SA", "UGPA3.SA", "RAIZ4.SA", "PETZ3.SA",
    "SLCE3.SA", "JHSF3.SA", "ALPA4.SA", "RAIL3.SA", "TOTS3.SA", "MGLU3.SA",
]

# Principais FIIs (Fundos Imobiliários)
FII_TICKERS = [
    "HGLG11.SA", "KNRI11.SA", "MXRF11.SA", "BCFF11.SA", "VISC11.SA",
    "XPML11.SA", "HGBS11.SA", "HGRE11.SA", "HGCR11.SA", "KNCR11.SA",
    "RECR11.SA", "BTLG11.SA", "XPLG11.SA", "VILG11.SA", "PVBI11.SA",
    "VGHF11.SA", "HGRU11.SA", "HSML11.SA", "CPTS11.SA", "IRDM11.SA",
    "MCCI11.SA", "KNIP11.SA", "DEVA11.SA", "RBRR11.SA",
    "SNAG11.SA", "SNCI11.SA",
]

ALL_TICKERS = IBOV_TICKERS + FII_TICKERS


# --------------------------------------------------------------------------
# FUNÇÕES DE DATA / PERÍODO
# --------------------------------------------------------------------------
def previous_month_range(reference: Optional[date] = None) -> tuple[date, date]:
    """Retorna (primeiro_dia, ultimo_dia) do mês anterior à data de referência."""
    if reference is None:
        reference = date.today()
    first_of_current = reference.replace(day=1)
    last_of_prev = first_of_current - timedelta(days=1)
    first_of_prev = last_of_prev.replace(day=1)
    return first_of_prev, last_of_prev


def month_label(d: date) -> str:
    meses = [
        "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
        "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro",
    ]
    return f"{meses[d.month - 1]}/{d.year}"


# --------------------------------------------------------------------------
# DOWNLOAD DE DADOS (com cache diário)
# --------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def download_prices(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Baixa preços ajustados (Close) e Volume para a lista de tickers."""
    # end+1 pois yfinance trata end como exclusivo
    data = yf.download(
        tickers=" ".join(tickers),
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    return data


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_ticker_info(ticker: str) -> dict:
    """Busca informações fundamentalistas (DY, P/L, MarketCap, etc.) via .info."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        return {
            "shortName": info.get("shortName") or info.get("longName") or ticker,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "marketCap": info.get("marketCap"),
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "priceToBook": info.get("priceToBook"),
            "dividendYield": info.get("dividendYield"),  # já vem em fração (0.05 = 5%)
            "trailingAnnualDividendYield": info.get("trailingAnnualDividendYield"),
            "beta": info.get("beta"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "recommendationKey": info.get("recommendationKey"),
        }
    except Exception:
        return {"shortName": ticker}


# --------------------------------------------------------------------------
# INDICADORES TÉCNICOS
# --------------------------------------------------------------------------
def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def annualized_volatility(series: pd.Series) -> float:
    rets = series.pct_change().dropna()
    if len(rets) < 5:
        return np.nan
    return float(rets.std() * math.sqrt(252))


def trend_signal(close: pd.Series) -> str:
    """Sinaliza tendência baseada em alinhamento de médias."""
    s20 = sma(close, 20).iloc[-1] if len(close) >= 20 else np.nan
    s50 = sma(close, 50).iloc[-1] if len(close) >= 50 else np.nan
    s200 = sma(close, 200).iloc[-1] if len(close) >= 200 else np.nan
    price = close.iloc[-1]
    if np.isnan(s20) or np.isnan(s50):
        return "indefinida"
    if price > s20 > s50 and (np.isnan(s200) or price > s200):
        return "alta forte"
    if price > s20 and price > s50:
        return "alta"
    if price < s20 < s50:
        return "baixa"
    return "lateral"


# --------------------------------------------------------------------------
# ANÁLISE POR ATIVO
# --------------------------------------------------------------------------
def analyze_ticker(
    ticker: str,
    data: pd.DataFrame,
    prev_start: date,
    prev_end: date,
) -> Optional[dict]:
    """Calcula métricas do ativo para o mês anterior + análise técnica ampla."""
    try:
        df = data[ticker] if isinstance(data.columns, pd.MultiIndex) else data
    except KeyError:
        return None

    if df is None or df.empty or "Close" not in df.columns:
        return None

    df = df.dropna(subset=["Close"]).copy()
    if len(df) < 20:
        return None

    # Recorte do mês anterior
    mask = (df.index.date >= prev_start) & (df.index.date <= prev_end)
    prev_df = df.loc[mask]
    if len(prev_df) < 3:
        return None

    first_close = prev_df["Close"].iloc[0]
    last_close = prev_df["Close"].iloc[-1]
    month_return = (last_close / first_close - 1) * 100

    # Volume médio diário (R$) no mês anterior
    avg_daily_volume_financial = float(
        (prev_df["Close"] * prev_df["Volume"]).mean()
    )
    avg_daily_volume_qty = float(prev_df["Volume"].mean())

    # Indicadores técnicos usando a série histórica completa
    close = df["Close"]
    last_price = float(close.iloc[-1])
    last_rsi = float(rsi(close).iloc[-1]) if len(close) >= 15 else np.nan
    last_sma20 = float(sma(close, 20).iloc[-1]) if len(close) >= 20 else np.nan
    last_sma50 = float(sma(close, 50).iloc[-1]) if len(close) >= 50 else np.nan
    last_sma200 = float(sma(close, 200).iloc[-1]) if len(close) >= 200 else np.nan
    vol_anual = annualized_volatility(close)
    trend = trend_signal(close)

    # Retorno YTD (desde início do ano)
    ytd_start = date(prev_end.year, 1, 1)
    ytd_df = df.loc[df.index.date >= ytd_start]
    ytd_return = (
        (ytd_df["Close"].iloc[-1] / ytd_df["Close"].iloc[0] - 1) * 100
        if len(ytd_df) > 1
        else np.nan
    )

    # Retorno 12 meses
    ret12m = np.nan
    if len(close) > 252:
        ret12m = (close.iloc[-1] / close.iloc[-252] - 1) * 100

    # Dados fundamentalistas
    info = fetch_ticker_info(ticker)
    dy = info.get("dividendYield") or info.get("trailingAnnualDividendYield")
    dy_pct = (dy * 100) if dy is not None and dy < 1.5 else dy  # yfinance às vezes retorna já em %

    return {
        "ticker": ticker,
        "nome": info.get("shortName", ticker),
        "setor": info.get("sector") or ("FII" if ticker.endswith("11.SA") else "-"),
        "tipo": "FII" if ticker in FII_TICKERS else "Ação",
        "preço": last_price,
        "retorno_mês_%": month_return,
        "retorno_ytd_%": ytd_return,
        "retorno_12m_%": ret12m,
        "volume_médio_R$": avg_daily_volume_financial,
        "volume_médio_qtd": avg_daily_volume_qty,
        "DY_%": dy_pct,
        "P/L": info.get("trailingPE"),
        "P/VP": info.get("priceToBook"),
        "beta": info.get("beta"),
        "market_cap": info.get("marketCap"),
        "RSI": last_rsi,
        "SMA20": last_sma20,
        "SMA50": last_sma50,
        "SMA200": last_sma200,
        "vol_anual_%": vol_anual * 100 if not np.isnan(vol_anual) else np.nan,
        "tendência": trend,
        "max_52s": info.get("fiftyTwoWeekHigh"),
        "min_52s": info.get("fiftyTwoWeekLow"),
        "_close_series": close,  # para gráficos
    }


# --------------------------------------------------------------------------
# SCORE PROBABILÍSTICO (MULTI-FATOR)
# --------------------------------------------------------------------------
def _to_float(value):
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def _norm(value, lo, hi):
    """Normaliza para [0,1] com clamp."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.5  # neutro
    if hi == lo:
        return 0.5
    return float(max(0.0, min(1.0, (value - lo) / (hi - lo))))


def compute_score(row: dict) -> dict:
    """
    Score probabilístico 0-100 combinando fatores:
      - Momentum (retorno do mês anterior)            25%
      - Tendência técnica (RSI + alinhamento SMA)     20%
      - Liquidez (volume financeiro)                  15%
      - Dividend Yield                                15%
      - Valuation (P/L, P/VP invertidos)              10%
      - Volatilidade (menor é melhor)                 10%
      - Distância do topo 52 semanas                   5%
    """
    # Momentum — retorno do mês anterior
    momentum = _norm(row.get("retorno_mês_%"), -10, 20)

    # RSI: prefere zona 40-65 (não sobrecomprado)
    rsi_v = row.get("RSI")
    if rsi_v is None or np.isnan(rsi_v):
        rsi_score = 0.5
    else:
        # função triangular centrada em 55
        rsi_score = max(0.0, 1 - abs(rsi_v - 55) / 45)

    # Tendência
    trend_map = {"alta forte": 1.0, "alta": 0.75, "lateral": 0.4, "baixa": 0.1, "indefinida": 0.4}
    trend_score = trend_map.get(row.get("tendência", "indefinida"), 0.4)
    tech_score = 0.5 * rsi_score + 0.5 * trend_score

    # Liquidez (log do volume financeiro)
    vol = row.get("volume_médio_R$") or 0
    liq_score = _norm(math.log10(vol + 1), 5, 9)  # 100K a 1B

    # Dividend Yield
    dy = _to_float(row.get("DY_%"))
    dy_score = _norm(dy, 0, 12)  # cap a 12%

    # Valuation — P/L e P/VP (menor é melhor, dentro de faixa razoável)
    pl = _to_float(row.get("P/L"))
    pl_score = 1 - _norm(pl, 3, 35) if pl and pl > 0 else 0.4
    pvp = _to_float(row.get("P/VP"))
    pvp_score = 1 - _norm(pvp, 0.5, 4) if pvp and pvp > 0 else 0.4
    val_score = 0.5 * pl_score + 0.5 * pvp_score

    # Volatilidade (menor é melhor)
    vola = _to_float(row.get("vol_anual_%"))
    vol_score = 1 - _norm(vola, 15, 70)

    # Distância do topo 52s (quanto menor a distância, melhor momentum)
    price = _to_float(row.get("preço"))
    hi52 = _to_float(row.get("max_52s"))
    if price and hi52 and hi52 > 0:
        dist = (hi52 - price) / hi52
        dist_score = 1 - _norm(dist, 0, 0.4)
    else:
        dist_score = 0.5

    score = (
        0.25 * momentum
        + 0.20 * tech_score
        + 0.15 * liq_score
        + 0.15 * dy_score
        + 0.10 * val_score
        + 0.10 * vol_score
        + 0.05 * dist_score
    ) * 100

    return {
        "score": round(score, 1),
        "score_momentum": round(momentum * 100, 1),
        "score_técnico": round(tech_score * 100, 1),
        "score_liquidez": round(liq_score * 100, 1),
        "score_DY": round(dy_score * 100, 1),
        "score_valuation": round(val_score * 100, 1),
        "score_volatilidade": round(vol_score * 100, 1),
    }


def classify(score: float) -> str:
    if score >= 70:
        return "🟢 Forte candidato"
    if score >= 55:
        return "🟡 Atrativo"
    if score >= 40:
        return "🟠 Neutro"
    return "🔴 Evitar"


# --------------------------------------------------------------------------
# GRÁFICOS
# --------------------------------------------------------------------------
def plot_price_chart(close: pd.Series, ticker: str) -> go.Figure:
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(
        x=close.index, y=close.values, mode="lines",
        name="Preço", line=dict(color="#1f77b4", width=2),
    ))
    if len(close) >= 20:
        fig.add_trace(go.Scatter(
            x=close.index, y=sma(close, 20).values, mode="lines",
            name="SMA 20", line=dict(color="#ff7f0e", width=1, dash="dot"),
        ))
    if len(close) >= 50:
        fig.add_trace(go.Scatter(
            x=close.index, y=sma(close, 50).values, mode="lines",
            name="SMA 50", line=dict(color="#2ca02c", width=1, dash="dash"),
        ))
    if len(close) >= 200:
        fig.add_trace(go.Scatter(
            x=close.index, y=sma(close, 200).values, mode="lines",
            name="SMA 200", line=dict(color="#d62728", width=1, dash="longdash"),
        ))
    fig.update_layout(
        title=f"{ticker} — Preço e Médias Móveis",
        height=350, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


# --------------------------------------------------------------------------
# INTERFACE
# --------------------------------------------------------------------------
def format_money(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if v >= 1e9:
        return f"R$ {v/1e9:.2f}B"
    if v >= 1e6:
        return f"R$ {v/1e6:.2f}M"
    if v >= 1e3:
        return f"R$ {v/1e3:.1f}K"
    return f"R$ {v:.2f}"


def format_pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.2f}%"


def main():
    st.title("📈 Dashboard de Ativos — Top 10 do Mês Anterior")
    st.caption(
        "Análise probabilística multi-fator de ações do Ibovespa e principais FIIs. "
        "Dados via yfinance. Atualização diária."
    )

    # --------- SIDEBAR ---------
    with st.sidebar:
        st.header("⚙️ Configurações")

        ref_date = st.date_input(
            "Data de referência",
            value=date.today(),
            help="A análise considera o mês anterior a esta data.",
        )
        prev_start, prev_end = previous_month_range(ref_date)
        st.info(f"**Mês analisado:** {month_label(prev_start)}")

        tipo_filtro = st.multiselect(
            "Tipos de ativo",
            ["Ação", "FII"],
            default=["Ação", "FII"],
        )

        top_n = st.slider("Quantidade de ativos no ranking", 5, 20, 10)

        min_volume = st.number_input(
            "Volume financeiro mínimo (R$ milhões/dia)",
            value=5.0, step=1.0,
            help="Filtra ativos com baixa liquidez.",
        )

        st.divider()
        st.caption(f"🗓️ Executado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        run_btn = st.button("🔄 Atualizar análise", type="primary", use_container_width=True)

    # --------- SELEÇÃO DO UNIVERSO ---------
    tickers = []
    if "Ação" in tipo_filtro:
        tickers += IBOV_TICKERS
    if "FII" in tipo_filtro:
        tickers += FII_TICKERS

    if not tickers:
        st.warning("Selecione ao menos um tipo de ativo na barra lateral.")
        return

    # Para ter histórico suficiente para SMA200, puxamos 2 anos.
    hist_start = prev_start - timedelta(days=500)
    hist_end = ref_date

    # --------- DOWNLOAD ---------
    with st.spinner(f"Baixando dados de {len(tickers)} ativos (cache diário)..."):
        data = download_prices(tickers, hist_start, hist_end)

    # --------- ANÁLISE ---------
    rows = []
    progress = st.progress(0.0, text="Analisando ativos...")
    for i, tk in enumerate(tickers):
        res = analyze_ticker(tk, data, prev_start, prev_end)
        if res:
            res.update(compute_score(res))
            res["classificação"] = classify(res["score"])
            rows.append(res)
        progress.progress((i + 1) / len(tickers))
    progress.empty()

    if not rows:
        st.error("Não foi possível analisar nenhum ativo. Verifique conexão/tickers.")
        return

    df = pd.DataFrame(rows)

    # Filtro de liquidez
    df = df[df["volume_médio_R$"].fillna(0) >= min_volume * 1_000_000]

    if df.empty:
        st.warning("Nenhum ativo passou no filtro de liquidez. Reduza o mínimo.")
        return

    # --------- KPIs DE MERCADO ---------
    st.subheader("📊 Visão geral do mercado no período")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ativos analisados", len(df))
    c2.metric(
        "Retorno médio no mês",
        f"{df['retorno_mês_%'].mean():.2f}%",
    )
    positivos = (df["retorno_mês_%"] > 0).sum()
    c3.metric("% em alta", f"{positivos / len(df) * 100:.0f}%")
    c4.metric(
        "DY médio (anual)",
        f"{df['DY_%'].dropna().mean():.2f}%" if df["DY_%"].notna().any() else "-",
    )

    st.divider()

    # --------- TOP 10 POR PERFORMANCE DO MÊS ---------
    st.subheader(f"🏆 Top {top_n} — Melhor performance em {month_label(prev_start)}")
    top_perf = df.sort_values("retorno_mês_%", ascending=False).head(top_n)

    display_cols = [
        "ticker", "nome", "tipo", "preço", "retorno_mês_%", "retorno_ytd_%",
        "DY_%", "P/L", "P/VP", "RSI", "tendência", "score", "classificação",
    ]
    fmt_map = {
        "preço": "R$ {:.2f}",
        "retorno_mês_%": "{:.2f}%",
        "retorno_ytd_%": "{:.2f}%",
        "DY_%": "{:.2f}%",
        "P/L": "{:.1f}",
        "P/VP": "{:.2f}",
        "RSI": "{:.1f}",
        "score": "{:.1f}",
    }
    st.dataframe(
        top_perf[display_cols].style.format(fmt_map, na_rep="-").background_gradient(
            subset=["score", "retorno_mês_%"], cmap="RdYlGn",
        ),
        use_container_width=True, hide_index=True,
    )

    # --------- RANKING PROBABILÍSTICO (SCORE) ---------
    st.subheader(f"🎯 Top {top_n} — Melhor score probabilístico (sugestões de compra)")
    top_score = df.sort_values("score", ascending=False).head(top_n)
    st.dataframe(
        top_score[display_cols].style.format(fmt_map, na_rep="-").background_gradient(
            subset=["score"], cmap="RdYlGn",
        ),
        use_container_width=True, hide_index=True,
    )

    st.caption(
        "⚖️ **Score** combina: Momentum (25%) + Técnico/RSI/Tendência (20%) + "
        "Liquidez (15%) + DY (15%) + Valuation (10%) + Baixa volatilidade (10%) + "
        "Distância do topo 52s (5%)."
    )

    st.divider()

    # --------- ANÁLISE DETALHADA DE UM ATIVO ---------
    st.subheader("🔍 Análise detalhada por ativo")
    opcoes = top_score["ticker"].tolist()
    escolha = st.selectbox("Selecione um ativo:", opcoes)

    if escolha:
        linha = top_score[top_score["ticker"] == escolha].iloc[0]

        st.markdown(f"### {linha['nome']} (`{linha['ticker']}`)")
        st.caption(f"Setor: {linha['setor']} | Tipo: {linha['tipo']}")

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Preço atual", f"R$ {linha['preço']:.2f}")
        k2.metric("Retorno mês", format_pct(linha["retorno_mês_%"]))
        k3.metric("DY (anual)", format_pct(linha["DY_%"]))
        k4.metric("Score", f"{linha['score']:.1f}/100", linha["classificação"])
        k5.metric("Tendência", linha["tendência"].title())

        col_a, col_b = st.columns([2, 1])

        with col_a:
            serie = linha["_close_series"]
            st.plotly_chart(plot_price_chart(serie, escolha), use_container_width=True)

        with col_b:
            st.markdown("**📐 Indicadores técnicos**")
            st.dataframe(
                pd.DataFrame({
                    "Indicador": ["RSI (14)", "SMA 20", "SMA 50", "SMA 200", "Vol. anual"],
                    "Valor": [
                        f"{linha['RSI']:.1f}" if not np.isnan(linha["RSI"]) else "-",
                        f"R$ {linha['SMA20']:.2f}" if not np.isnan(linha["SMA20"]) else "-",
                        f"R$ {linha['SMA50']:.2f}" if not np.isnan(linha["SMA50"]) else "-",
                        f"R$ {linha['SMA200']:.2f}" if not np.isnan(linha["SMA200"]) else "-",
                        format_pct(linha["vol_anual_%"]),
                    ],
                }),
                hide_index=True,
            )
            st.markdown("**📊 Fundamentos**")
            st.dataframe(
                pd.DataFrame({
                    "Indicador": ["P/L", "P/VP", "Beta", "Market Cap",
                                  "Máx. 52s", "Mín. 52s", "Vol. médio"],
                    "Valor": [
                        f"{linha['P/L']:.1f}" if linha["P/L"] else "-",
                        f"{linha['P/VP']:.2f}" if linha["P/VP"] else "-",
                        f"{linha['beta']:.2f}" if linha["beta"] else "-",
                        format_money(linha["market_cap"]),
                        f"R$ {linha['max_52s']:.2f}" if linha["max_52s"] else "-",
                        f"R$ {linha['min_52s']:.2f}" if linha["min_52s"] else "-",
                        format_money(linha["volume_médio_R$"]),
                    ],
                }),
                hide_index=True,
            )

        # Detalhamento do score
        st.markdown("**🧮 Decomposição do score probabilístico**")
        score_parts = pd.DataFrame({
            "Fator": ["Momentum", "Técnico", "Liquidez", "DY", "Valuation", "Volatilidade"],
            "Pontuação (0-100)": [
                linha["score_momentum"], linha["score_técnico"],
                linha["score_liquidez"], linha["score_DY"],
                linha["score_valuation"], linha["score_volatilidade"],
            ],
            "Peso": ["25%", "20%", "15%", "15%", "10%", "10%"],
        })
        fig = go.Figure(go.Bar(
            x=score_parts["Fator"],
            y=score_parts["Pontuação (0-100)"],
            text=score_parts["Pontuação (0-100)"],
            textposition="outside",
            marker_color=["#2ecc71" if v >= 60 else "#f39c12" if v >= 40 else "#e74c3c"
                          for v in score_parts["Pontuação (0-100)"]],
        ))
        fig.update_layout(
            height=300, margin=dict(l=10, r=10, t=30, b=10),
            yaxis=dict(range=[0, 110], title="Pontos"),
            title="Contribuição de cada fator ao score",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --------- TABELA COMPLETA ---------
    with st.expander("📋 Ver tabela completa de todos os ativos analisados"):
        full_cols = [c for c in df.columns if not c.startswith("_")]
        st.dataframe(
            df[full_cols].sort_values("score", ascending=False),
            use_container_width=True, hide_index=True,
        )

    st.caption(
        "⚠️ **Aviso:** Esta aplicação é uma ferramenta de apoio à decisão. "
        "Não constitui recomendação de investimento. Rentabilidade passada não "
        "garante rentabilidade futura."
    )


if __name__ == "__main__":
    main()
