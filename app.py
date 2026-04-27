"""
Dashboard de Análise de Ativos — Top 10 do Mês
================================================
Aplicação Streamlit que usa yfinance para buscar ações do Ibovespa e principais
FIIs, analisar performance do mês selecionado (default: mês corrente parcial,
com possibilidade de navegar para meses anteriores) e sugerir os 10 melhores
candidatos para compra com base em análise probabilística (multi-fator).

Como executar:
    pip install -r requirements.txt
    streamlit run app.py

Autor: Julio Dantas (gerado com Claude)
Versão: 0.4 — adiciona seção "Sobre o ativo" na análise detalhada com
descrição do negócio (longBusinessSummary), indústria, sede, funcionários,
valor de mercado e site oficial, mais coerção robusta de tipos do yfinance.
v0.3 — análise default passa a ser o MÊS CORRENTE com seletor de mês.
v0.2 — detecção de padrões gráficos, benchmark setorial e narrativa textual.
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


def month_range(year: int, month: int, today: Optional[date] = None) -> tuple[date, date]:
    """
    Retorna (primeiro_dia, último_dia_observado) de um mês qualquer.
    Se for o mês corrente e ainda não terminou, o último dia é HOJE
    (análise parcial). Caso contrário, é o último dia do mês.
    """
    if today is None:
        today = date.today()
    first = date(year, month, 1)
    if month == 12:
        next_first = date(year + 1, 1, 1)
    else:
        next_first = date(year, month + 1, 1)
    last_of_month = next_first - timedelta(days=1)
    if (year, month) == (today.year, today.month):
        last = min(today, last_of_month)
    else:
        last = last_of_month
    return first, last


def list_recent_months(today: Optional[date] = None, n: int = 24) -> list[tuple[int, int]]:
    """Lista (ano, mês) dos últimos N meses, mais recente primeiro."""
    if today is None:
        today = date.today()
    months: list[tuple[int, int]] = []
    y, m = today.year, today.month
    for _ in range(n):
        months.append((y, m))
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return months


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
            "longName": info.get("longName"),
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
            # Descrição e metadados do negócio
            "longBusinessSummary": info.get("longBusinessSummary"),
            "website": info.get("website"),
            "country": info.get("country"),
            "city": info.get("city"),
            "state": info.get("state"),
            "fullTimeEmployees": info.get("fullTimeEmployees"),
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
# DETECÇÃO DE PADRÕES GRÁFICOS
# --------------------------------------------------------------------------
def find_swing_points(series: pd.Series, window: int = 5):
    """Encontra topos e fundos locais (pivôs) com confirmação de ±window dias."""
    series = series.dropna()
    highs, lows = [], []
    if len(series) < 2 * window + 1:
        return highs, lows
    vals = series.values
    idx = series.index
    for i in range(window, len(series) - window):
        win = vals[i - window: i + window + 1]
        v = vals[i]
        if v == win.max() and v > vals[i - 1]:
            highs.append((idx[i], float(v)))
        if v == win.min() and v < vals[i - 1]:
            lows.append((idx[i], float(v)))
    return highs, lows


def detect_patterns(df: pd.DataFrame) -> dict:
    """
    Identifica padrões gráficos relevantes na série de preços.
    Retorna dict {pattern_id: {label, tipo (alta/baixa/neutro), detail}}.
    """
    patterns: dict = {}
    if df is None or df.empty or "Close" not in df.columns:
        return patterns

    close = df["Close"].dropna()
    if len(close) < 60:
        return patterns

    # ---- 1. Topos e fundos (swings) ----
    highs, lows = find_swing_points(close, window=5)

    if len(lows) >= 3:
        recent = [v for _, v in lows[-3:]]
        if recent[0] < recent[1] < recent[2]:
            patterns["fundos_ascendentes"] = {
                "label": "Fundos ascendentes",
                "tipo": "alta",
                "detail": (
                    f"Últimos 3 fundos: R$ {recent[0]:.2f} → "
                    f"R$ {recent[1]:.2f} → R$ {recent[2]:.2f} "
                    "(estrutura clássica de tendência de alta)."
                ),
            }
        elif recent[0] > recent[1] > recent[2]:
            patterns["fundos_descendentes"] = {
                "label": "Fundos descendentes",
                "tipo": "baixa",
                "detail": (
                    f"Últimos 3 fundos: R$ {recent[0]:.2f} → "
                    f"R$ {recent[1]:.2f} → R$ {recent[2]:.2f} "
                    "(estrutura típica de tendência de baixa)."
                ),
            }

    if len(highs) >= 3:
        recent = [v for _, v in highs[-3:]]
        if recent[0] < recent[1] < recent[2]:
            patterns["topos_ascendentes"] = {
                "label": "Topos ascendentes",
                "tipo": "alta",
                "detail": (
                    f"Últimos 3 topos: R$ {recent[0]:.2f} → "
                    f"R$ {recent[1]:.2f} → R$ {recent[2]:.2f}."
                ),
            }
        elif recent[0] > recent[1] > recent[2]:
            patterns["topos_descendentes"] = {
                "label": "Topos descendentes",
                "tipo": "baixa",
                "detail": (
                    f"Últimos 3 topos: R$ {recent[0]:.2f} → "
                    f"R$ {recent[1]:.2f} → R$ {recent[2]:.2f}."
                ),
            }

    # ---- 2. Golden/Death cross (cruzamento SMA50 x SMA200) ----
    if len(close) >= 200:
        s50 = sma(close, 50)
        s200 = sma(close, 200)
        diff = (s50 - s200).dropna()
        recent_diff = diff.tail(30)
        if len(recent_diff) > 1:
            signs = np.sign(recent_diff.values)
            sign_changes = np.where(np.diff(signs) != 0)[0]
            if len(sign_changes) > 0:
                last_change = sign_changes[-1] + 1
                d = recent_diff.index[last_change]
                if signs[last_change] > 0:
                    patterns["golden_cross"] = {
                        "label": "Golden cross (SMA50 × SMA200)",
                        "tipo": "alta",
                        "detail": (
                            f"SMA50 cruzou ACIMA da SMA200 em "
                            f"{d.strftime('%d/%m/%Y')} — sinal clássico "
                            "de virada de tendência para alta."
                        ),
                    }
                else:
                    patterns["death_cross"] = {
                        "label": "Death cross (SMA50 × SMA200)",
                        "tipo": "baixa",
                        "detail": (
                            f"SMA50 cruzou ABAIXO da SMA200 em "
                            f"{d.strftime('%d/%m/%Y')} — sinal de virada "
                            "para baixa."
                        ),
                    }

    # ---- 3. Rompimento de resistência / perda de suporte (60 pregões) ----
    if len(close) >= 61:
        last = float(close.iloc[-1])
        prev_max = float(close.iloc[-61:-1].max())
        prev_min = float(close.iloc[-61:-1].min())
        if last > prev_max * 1.005:
            patterns["rompimento_resistencia"] = {
                "label": "Rompimento de resistência",
                "tipo": "alta",
                "detail": (
                    f"Preço (R$ {last:.2f}) rompeu a máxima dos últimos "
                    f"60 pregões (R$ {prev_max:.2f})."
                ),
            }
        elif last < prev_min * 0.995:
            patterns["perda_suporte"] = {
                "label": "Perda de suporte",
                "tipo": "baixa",
                "detail": (
                    f"Preço (R$ {last:.2f}) perdeu a mínima dos últimos "
                    f"60 pregões (R$ {prev_min:.2f})."
                ),
            }

    # ---- 4. Volume crescente nos últimos pregões ----
    if "Volume" in df.columns and len(df) >= 30:
        recent_vol = float(df["Volume"].tail(5).mean())
        avg_vol = float(df["Volume"].tail(30).mean())
        if avg_vol > 0 and recent_vol > avg_vol * 1.5:
            tipo = "alta" if close.iloc[-1] > close.iloc[-5] else "neutro"
            patterns["volume_crescente"] = {
                "label": "Volume crescente",
                "tipo": tipo,
                "detail": (
                    f"Volume dos últimos 5 pregões está "
                    f"{recent_vol / avg_vol:.1f}× acima da média de 30 dias — "
                    "indica entrada de fluxo no ativo."
                ),
            }

    # ---- 5. Divergência RSI (preço vs momentum) ----
    if len(close) >= 60:
        rsi_series = rsi(close).dropna()
        if len(rsi_series) >= 30:
            rc = close.tail(30)
            rr = rsi_series.tail(30)
            half = 15
            c1_min, c2_min = rc.iloc[:half].min(), rc.iloc[half:].min()
            r1_min, r2_min = rr.iloc[:half].min(), rr.iloc[half:].min()
            c1_max, c2_max = rc.iloc[:half].max(), rc.iloc[half:].max()
            r1_max, r2_max = rr.iloc[:half].max(), rr.iloc[half:].max()
            if c2_min < c1_min and r2_min > r1_min:
                patterns["divergencia_alta_rsi"] = {
                    "label": "Divergência de alta no RSI",
                    "tipo": "alta",
                    "detail": (
                        "Preço fez fundo mais baixo, mas RSI fez fundo "
                        "mais alto — sinal de exaustão da queda."
                    ),
                }
            if c2_max > c1_max and r2_max < r1_max:
                patterns["divergencia_baixa_rsi"] = {
                    "label": "Divergência de baixa no RSI",
                    "tipo": "baixa",
                    "detail": (
                        "Preço fez topo mais alto, mas RSI fez topo "
                        "mais baixo — sinal de exaustão da alta."
                    ),
                }

    return patterns


# --------------------------------------------------------------------------
# ANÁLISE POR ATIVO
# --------------------------------------------------------------------------
def analyze_ticker(
    ticker: str,
    data: pd.DataFrame,
    period_start: date,
    period_end: date,
) -> Optional[dict]:
    """
    Calcula métricas do ativo para o período (mês selecionado, podendo
    ser o mês corrente parcial) + análise técnica ampla sobre toda a série.
    """
    try:
        df = data[ticker] if isinstance(data.columns, pd.MultiIndex) else data
    except KeyError:
        return None

    if df is None or df.empty or "Close" not in df.columns:
        return None

    df = df.dropna(subset=["Close"]).copy()
    if len(df) < 20:
        return None

    # Recorte do período selecionado
    mask = (df.index.date >= period_start) & (df.index.date <= period_end)
    period_df = df.loc[mask]
    if len(period_df) < 3:
        return None

    first_close = period_df["Close"].iloc[0]
    last_close = period_df["Close"].iloc[-1]
    month_return = (last_close / first_close - 1) * 100

    # Volume médio diário (R$) no período
    avg_daily_volume_financial = float(
        (period_df["Close"] * period_df["Volume"]).mean()
    )
    avg_daily_volume_qty = float(period_df["Volume"].mean())

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
    ytd_start = date(period_end.year, 1, 1)
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

    # Dados fundamentalistas — sempre coagir para float (yfinance às vezes
    # devolve string, "Infinity" ou tipos exóticos para alguns tickers)
    info = fetch_ticker_info(ticker)
    dy_raw = _to_float(info.get("dividendYield"))
    if dy_raw is None:
        dy_raw = _to_float(info.get("trailingAnnualDividendYield"))
    # yfinance pode devolver em fração (0.08 = 8%) ou já em % — normaliza
    if dy_raw is not None:
        dy_pct = dy_raw * 100 if dy_raw < 1.5 else dy_raw
    else:
        dy_pct = None

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
        "P/L": _to_float(info.get("trailingPE")),
        "P/VP": _to_float(info.get("priceToBook")),
        "beta": _to_float(info.get("beta")),
        "market_cap": _to_float(info.get("marketCap")),
        "RSI": last_rsi,
        "SMA20": last_sma20,
        "SMA50": last_sma50,
        "SMA200": last_sma200,
        "vol_anual_%": vol_anual * 100 if not np.isnan(vol_anual) else np.nan,
        "tendência": trend,
        "max_52s": _to_float(info.get("fiftyTwoWeekHigh")),
        "min_52s": _to_float(info.get("fiftyTwoWeekLow")),
        # Descrição e metadados (para a seção "Sobre o ativo")
        "nome_longo": info.get("longName") or info.get("shortName"),
        "industry": info.get("industry"),
        "descrição": info.get("longBusinessSummary"),
        "website": info.get("website"),
        "country": info.get("country"),
        "city": info.get("city"),
        "state": info.get("state"),
        "funcionarios": _to_float(info.get("fullTimeEmployees")),
        "_close_series": close,
        "_ohlcv_df": df[["Open", "High", "Low", "Close", "Volume"]].copy(),
        "_patterns": detect_patterns(df),
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
      - Momentum (retorno do período analisado)        25%
      - Tendência técnica (RSI + alinhamento SMA)      20%
      - Liquidez (volume financeiro)                   15%
      - Dividend Yield                                 15%
      - Valuation (P/L, P/VP invertidos)               10%
      - Volatilidade (menor é melhor)                  10%
      - Distância do topo 52 semanas                    5%
    """
    # Momentum — retorno acumulado no mês selecionado
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
        return "Forte candidato"
    if score >= 55:
        return "Atrativo"
    if score >= 40:
        return "Neutro"
    return "Evitar"


# --------------------------------------------------------------------------
# BENCHMARK SETORIAL
# --------------------------------------------------------------------------
def compute_sector_benchmarks(df: pd.DataFrame) -> dict:
    """
    Calcula medianas por setor para DY, P/L, P/VP e retorno do mês.
    Setores com menos de 2 ativos são ignorados.
    """
    bench: dict = {}
    metrics = ["DY_%", "P/L", "P/VP", "retorno_mês_%"]
    for setor, group in df.groupby("setor"):
        if len(group) < 2:
            continue
        b = {"n": int(len(group))}
        for m in metrics:
            if m in group.columns:
                series = pd.to_numeric(group[m], errors="coerce").dropna()
                b[m] = float(series.median()) if len(series) > 0 else None
        bench[setor] = b
    return bench


def compare_to_sector(row: dict, benchmarks: dict) -> dict:
    """
    Compara métricas do ativo com a mediana do seu setor.
    Retorna {metrica: {valor, mediana_setor, delta_pct, delta_abs}}.
    """
    setor = row.get("setor")
    out: dict = {}
    if not setor or setor not in benchmarks:
        return out
    bench = benchmarks[setor]
    for m in ["DY_%", "P/L", "P/VP", "retorno_mês_%"]:
        v = row.get(m)
        b = bench.get(m)
        if v is None or b is None:
            continue
        try:
            v = float(v)
            b = float(b)
        except (TypeError, ValueError):
            continue
        if np.isnan(v) or np.isnan(b):
            continue
        delta_pct = ((v / b - 1) * 100) if b != 0 else None
        out[m] = {
            "valor": v,
            "mediana_setor": b,
            "delta_pct": delta_pct,
            "delta_abs": v - b,
            "n_setor": bench.get("n"),
        }
    return out


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


def plot_candlestick_chart(df: pd.DataFrame, ticker: str, period_days: int = 90) -> go.Figure:
    df_plot = df.dropna(subset=["Open", "High", "Low", "Close"]).tail(period_days).copy()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
    )

    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot["Open"],
        high=df_plot["High"],
        low=df_plot["Low"],
        close=df_plot["Close"],
        name="OHLC",
        increasing_line_color="#2ecc71",
        decreasing_line_color="#e74c3c",
    ), row=1, col=1)

    close_plot = df_plot["Close"]
    for window, color, dash, label in [
        (20, "#ff7f0e", "dot", "SMA 20"),
        (50, "#2ca02c", "dash", "SMA 50"),
    ]:
        if len(close_plot) >= window:
            sma_vals = sma(close_plot, window)
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=sma_vals.values, mode="lines",
                name=label, line=dict(color=color, width=1, dash=dash),
            ), row=1, col=1)

    bar_colors = [
        "#2ecc71" if c >= o else "#e74c3c"
        for c, o in zip(df_plot["Close"], df_plot["Open"])
    ]
    fig.add_trace(go.Bar(
        x=df_plot.index, y=df_plot["Volume"],
        marker_color=bar_colors, showlegend=False, name="Volume",
    ), row=2, col=1)

    fig.update_layout(
        title=f"{ticker} — Candlestick",
        height=450,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=-0.12),
    )
    fig.update_yaxes(title_text="Preço (R$)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
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


# --------------------------------------------------------------------------
# GERAÇÃO DE NARRATIVA (combina score + tendência + padrões + setor)
# --------------------------------------------------------------------------
def _safe_float(v):
    try:
        f = float(v)
        return f if not np.isnan(f) else None
    except (TypeError, ValueError):
        return None


def generate_narrative(row: dict, patterns: dict, sector_compare: dict) -> str:
    """
    Gera um parágrafo de análise consolidando:
      - score + classificação
      - tendência por médias móveis
      - padrões gráficos detectados
      - RSI, volatilidade, liquidez
      - comparação com mediana do setor (DY, P/L)
      - distância do topo de 52 semanas
    """
    parts: list[str] = []

    nome = row.get("nome", row.get("ticker", "Ativo"))
    score = _safe_float(row.get("score")) or 0.0
    setor = row.get("setor") or "-"

    # 1. Frase de abertura — tese baseada no score
    if score >= 70:
        parts.append(
            f"**{nome}** é um forte candidato a compra (score {score:.1f}/100, "
            f"setor: {setor})."
        )
    elif score >= 55:
        parts.append(
            f"**{nome}** é um candidato atrativo (score {score:.1f}/100, "
            f"setor: {setor})."
        )
    elif score >= 40:
        parts.append(
            f"**{nome}** está em zona neutra (score {score:.1f}/100, "
            f"setor: {setor}) — aguardar melhores sinais."
        )
    else:
        parts.append(
            f"**{nome}** apresenta sinais desfavoráveis (score "
            f"{score:.1f}/100, setor: {setor})."
        )

    # 2. Tendência
    trend = row.get("tendência")
    if trend == "alta forte":
        parts.append(
            "Tendência de alta forte com SMA20 > SMA50 > SMA200 e preço "
            "acima das três médias."
        )
    elif trend == "alta":
        parts.append(
            "Preço acima das médias de curto e médio prazo, configurando "
            "tendência de alta."
        )
    elif trend == "lateral":
        parts.append("Tendência lateral — sem direção clara nas médias móveis.")
    elif trend == "baixa":
        parts.append(
            "Tendência de baixa com preço abaixo das principais médias."
        )

    # 3. Padrões gráficos — separar bullish e bearish
    if patterns:
        bullish = [p for p in patterns.values() if p.get("tipo") == "alta"]
        bearish = [p for p in patterns.values() if p.get("tipo") == "baixa"]
        if bullish:
            labels = ", ".join(p["label"].lower() for p in bullish)
            parts.append(f"Sinais positivos no gráfico: {labels}.")
        if bearish:
            labels = ", ".join(p["label"].lower() for p in bearish)
            parts.append(f"Sinais negativos no gráfico: {labels}.")

    # 4. RSI
    rsi_v = _safe_float(row.get("RSI"))
    if rsi_v is not None:
        if rsi_v < 30:
            parts.append(
                f"RSI em {rsi_v:.0f} indica sobrevenda — possível ponto de "
                "retomada técnica."
            )
        elif rsi_v > 70:
            parts.append(
                f"RSI em {rsi_v:.0f} indica sobrecompra — risco de correção."
            )
        elif 45 <= rsi_v <= 65:
            parts.append(f"RSI em {rsi_v:.0f} (faixa saudável de momentum).")
        else:
            parts.append(f"RSI em {rsi_v:.0f}.")

    # 5. Comparação setorial — DY
    dy = _safe_float(row.get("DY_%"))
    if dy is not None:
        cmp = sector_compare.get("DY_%")
        if cmp and cmp.get("delta_pct") is not None:
            d = cmp["delta_pct"]
            if d > 15:
                parts.append(
                    f"DY de {dy:.2f}% supera a mediana do setor "
                    f"({cmp['mediana_setor']:.2f}%) em {d:.0f}% — "
                    "atrativo para renda."
                )
            elif d < -15:
                parts.append(
                    f"DY de {dy:.2f}% está {-d:.0f}% abaixo da mediana "
                    f"setorial ({cmp['mediana_setor']:.2f}%)."
                )
            else:
                parts.append(
                    f"DY de {dy:.2f}% em linha com a mediana setorial "
                    f"({cmp['mediana_setor']:.2f}%)."
                )
        else:
            parts.append(f"DY anual de {dy:.2f}%.")

    # 6. Comparação setorial — P/L
    pl = _safe_float(row.get("P/L"))
    if pl is not None and pl > 0:
        cmp = sector_compare.get("P/L")
        if cmp and cmp.get("delta_pct") is not None:
            d = cmp["delta_pct"]
            if d < -20:
                parts.append(
                    f"P/L de {pl:.1f} está {-d:.0f}% abaixo da mediana do "
                    "setor — valuation potencialmente atrativo."
                )
            elif d > 30:
                parts.append(
                    f"P/L de {pl:.1f} está {d:.0f}% acima do setor — "
                    "valuation esticado."
                )

    # 7. Liquidez
    vol = _safe_float(row.get("volume_médio_R$"))
    if vol is not None:
        if vol >= 100e6:
            parts.append(
                f"Liquidez excelente ({format_money(vol)}/dia em média)."
            )
        elif vol >= 20e6:
            parts.append(
                f"Liquidez confortável ({format_money(vol)}/dia em média)."
            )
        elif vol < 5e6:
            parts.append(
                f"Atenção: liquidez baixa ({format_money(vol)}/dia) pode "
                "dificultar entrada/saída sem afetar preço."
            )

    # 8. Volatilidade
    vola = _safe_float(row.get("vol_anual_%"))
    if vola is not None:
        if vola > 50:
            parts.append(
                f"Volatilidade alta ({vola:.0f}% a.a.) — exige tolerância "
                "a oscilações."
            )
        elif vola < 25:
            parts.append(
                f"Volatilidade baixa ({vola:.0f}% a.a.) — perfil mais estável."
            )

    # 9. Distância do topo de 52 semanas
    price = _safe_float(row.get("preço"))
    hi52 = _safe_float(row.get("max_52s"))
    if price and hi52 and hi52 > 0:
        dist = (hi52 - price) / hi52 * 100
        if dist < 5:
            parts.append(
                f"Cotação a {dist:.1f}% da máxima de 52 semanas — momentum "
                "positivo confirmado."
            )
        elif dist > 30:
            parts.append(
                f"Cotação {dist:.0f}% abaixo da máxima de 52 semanas — "
                "pode ser oportunidade de valor ou problema estrutural; "
                "checar fundamentos."
            )

    return " ".join(parts)


def main():
    st.title("Dashboard de Ativos — Top 10 do Mês")
    st.caption(
        "Análise probabilística multi-fator de ações do Ibovespa e principais "
        "FIIs. Padrão: mês corrente (parcial). Use o seletor para navegar pelos "
        "meses anteriores. Dados via yfinance, cache diário."
    )

    # --------- SIDEBAR ---------
    today = date.today()
    with st.sidebar:
        st.header("Configurações")

        # Seletor de mês — default é o mês corrente (índice 0 da lista)
        recent = list_recent_months(today, n=24)

        def _fmt_month(ym: tuple[int, int]) -> str:
            y, m = ym
            label = month_label(date(y, m, 1))
            if (y, m) == (today.year, today.month):
                return f"{label} (mês corrente)"
            return label

        selected_ym = st.selectbox(
            "Mês de análise",
            options=recent,
            index=0,
            format_func=_fmt_month,
            help=(
                "Mês corrente é parcial (até hoje). Meses anteriores cobrem o "
                "mês inteiro."
            ),
        )
        sel_year, sel_month = selected_ym
        period_start, period_end = month_range(sel_year, sel_month, today)
        is_current_month = (sel_year, sel_month) == (today.year, today.month)
        n_dias = (period_end - period_start).days + 1
        if is_current_month:
            st.info(
                f"**{month_label(period_start)}** (parcial: "
                f"{period_start.strftime('%d/%m')} → "
                f"{period_end.strftime('%d/%m')}, {n_dias} dias corridos)"
            )
        else:
            st.info(f"**{month_label(period_start)}** (mês completo)")

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
        st.caption(f"Executado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        run_btn = st.button("Atualizar análise", type="primary", use_container_width=True)

    # --------- SELEÇÃO DO UNIVERSO ---------
    tickers = []
    if "Ação" in tipo_filtro:
        tickers += IBOV_TICKERS
    if "FII" in tipo_filtro:
        tickers += FII_TICKERS

    if not tickers:
        st.warning("Selecione ao menos um tipo de ativo na barra lateral.")
        return

    # Para ter histórico suficiente para SMA200, puxamos ~2 anos antes
    # do início do período. O fim sempre vai até hoje (atualização diária).
    hist_start = period_start - timedelta(days=500)
    hist_end = today

    # --------- DOWNLOAD ---------
    with st.spinner(f"Baixando dados de {len(tickers)} ativos (cache diário)..."):
        data = download_prices(tickers, hist_start, hist_end)

    # Aviso quando o mês corrente tem poucos pregões (período curto distorce
    # leituras de retorno e volume).
    if is_current_month and n_dias < 5:
        st.warning(
            f"Mês corrente com apenas {n_dias} dias corridos — retornos e "
            "volumes do período podem ser pouco representativos. Considere "
            "comparar com o mês anterior no seletor."
        )

    # --------- ANÁLISE ---------
    rows = []
    progress = st.progress(0.0, text="Analisando ativos...")
    for i, tk in enumerate(tickers):
        res = analyze_ticker(tk, data, period_start, period_end)
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

    # Benchmarks setoriais (calculados sobre o universo filtrado)
    sector_benchmarks = compute_sector_benchmarks(df)

    # --------- KPIs DE MERCADO ---------
    st.subheader("Visão geral do mercado no período")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ativos analisados", len(df))
    c2.metric(
        "Retorno médio no mês",
        f"{df['retorno_mês_%'].mean():.2f}%",
    )
    positivos = (df["retorno_mês_%"] > 0).sum()
    c3.metric("% em alta", f"{positivos / len(df) * 100:.0f}%")
    dy_series = df["DY_%"].dropna()
    c4.metric(
        "DY médio (anual)",
        f"{dy_series.mean():.2f}%" if len(dy_series) > 0 else "-",
        f"{len(dy_series)}/{len(df)} ativos com dado",
    )

    st.divider()

    # --------- TOP 10 POR PERFORMANCE DO MÊS ---------
    label_periodo = month_label(period_start)
    if is_current_month:
        label_periodo += " (parcial)"
    st.subheader(f"Top {top_n} — Melhor performance em {label_periodo}")
    top_perf = df.sort_values("retorno_mês_%", ascending=False).head(top_n)

    display_cols = [
        "ticker", "nome", "tipo", "preço", "retorno_mês_%", "retorno_ytd_%",
        "DY_%", "P/L", "P/VP", "RSI", "tendência", "score", "classificação",
    ]
    numeric_cols = [
        "preço", "retorno_mês_%", "retorno_ytd_%", "DY_%", "P/L", "P/VP",
        "RSI", "score",
    ]

    def _coerce_numeric(d: pd.DataFrame) -> pd.DataFrame:
        """Garante que colunas numéricas sejam float (evita erro do Styler)."""
        d = d.copy()
        for c in numeric_cols:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        return d

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
        _coerce_numeric(top_perf[display_cols]).style.format(
            fmt_map, na_rep="-"
        ).background_gradient(
            subset=["score", "retorno_mês_%"], cmap="RdYlGn",
        ),
        use_container_width=True, hide_index=True,
    )

    # --------- RANKING PROBABILÍSTICO (SCORE) ---------
    st.subheader(f"Top {top_n} — Melhor score probabilístico (sugestões de compra)")
    top_score = df.sort_values("score", ascending=False).head(top_n)
    st.dataframe(
        _coerce_numeric(top_score[display_cols]).style.format(
            fmt_map, na_rep="-"
        ).background_gradient(
            subset=["score"], cmap="RdYlGn",
        ),
        use_container_width=True, hide_index=True,
    )

    st.caption(
        "**Score** combina: Momentum (25%) + Técnico/RSI/Tendência (20%) + "
        "Liquidez (15%) + DY (15%) + Valuation (10%) + Baixa volatilidade (10%) + "
        "Distância do topo 52s (5%)."
    )

    st.divider()

    # --------- ANÁLISE DETALHADA DE UM ATIVO ---------
    st.subheader("Análise detalhada por ativo")
    opcoes = top_score["ticker"].tolist()
    escolha = st.selectbox("Selecione um ativo:", opcoes)

    if escolha:
        linha = top_score[top_score["ticker"] == escolha].iloc[0]

        st.markdown(f"### {linha['nome']} (`{linha['ticker']}`)")
        st.caption(f"Setor: {linha['setor']} | Tipo: {linha['tipo']}")

        # ----- Sobre o ativo (descrição do negócio) -----
        with st.expander("Sobre o ativo", expanded=True):
            descricao = linha.get("descrição")
            nome_longo = linha.get("nome_longo")

            if nome_longo and nome_longo != linha["nome"]:
                st.markdown(f"**Razão social / nome completo:** {nome_longo}")

            if descricao:
                st.markdown(descricao)
                st.caption(
                    "Descrição original do yfinance (frequentemente em inglês "
                    "para ativos brasileiros)."
                )
            else:
                if linha["tipo"] == "FII":
                    st.info(
                        "Descrição detalhada não disponível na base do yfinance "
                        "para este FII. Consulte o material do gestor (lâmina, "
                        "relatório gerencial) para entender a estratégia, "
                        "segmento dos imóveis e composição da carteira."
                    )
                else:
                    st.info(
                        "Descrição detalhada não disponível na base do yfinance "
                        "para este ativo."
                    )

            # Metadados em colunas
            metadados = []
            industry = linha.get("industry")
            if industry:
                metadados.append(("Indústria", industry))

            country = linha.get("country")
            city = linha.get("city")
            state = linha.get("state")
            if country or city or state:
                partes = [p for p in [city, state, country] if p]
                if partes:
                    metadados.append(("Sede", ", ".join(partes)))

            funcionarios = linha.get("funcionarios")
            if funcionarios and not (isinstance(funcionarios, float) and np.isnan(funcionarios)):
                metadados.append(
                    ("Funcionários", f"{int(funcionarios):,}".replace(",", "."))
                )

            mc = linha.get("market_cap")
            if mc and not (isinstance(mc, float) and np.isnan(mc)):
                metadados.append(("Valor de mercado", format_money(mc)))

            website = linha.get("website")
            if website:
                # Garante prefixo http
                site = website if website.startswith("http") else f"https://{website}"
                metadados.append(("Site", f"[{website}]({site})"))

            if metadados:
                st.markdown("---")
                # Distribui em até 3 colunas
                ncols = min(len(metadados), 3)
                cols = st.columns(ncols)
                for i, (k, v) in enumerate(metadados):
                    cols[i % ncols].markdown(f"**{k}**  \n{v}")

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Preço atual", f"R$ {linha['preço']:.2f}")
        k2.metric("Retorno mês", format_pct(linha["retorno_mês_%"]))
        k3.metric("DY (anual)", format_pct(linha["DY_%"]))
        k4.metric("Score", f"{linha['score']:.1f}/100", linha["classificação"])
        k5.metric("Tendência", linha["tendência"].title())

        # ----- Narrativa gerada (combina padrões + setor + indicadores) -----
        linha_dict = linha.to_dict()
        patterns = linha_dict.get("_patterns") or {}
        sector_cmp = compare_to_sector(linha_dict, sector_benchmarks)
        narrative = generate_narrative(linha_dict, patterns, sector_cmp)

        st.markdown("#### Análise consolidada")
        st.info(narrative)

        # ----- Chips de padrões detectados -----
        if patterns:
            st.markdown("**Padrões gráficos detectados:**")
            cols_patt = st.columns(min(len(patterns), 4))
            for i, (pid, p) in enumerate(patterns.items()):
                col = cols_patt[i % len(cols_patt)]
                tipo = p.get("tipo", "neutro")
                if tipo == "alta":
                    col.success(f"**{p['label']}**\n\n{p['detail']}")
                elif tipo == "baixa":
                    col.error(f"**{p['label']}**\n\n{p['detail']}")
                else:
                    col.warning(f"**{p['label']}**\n\n{p['detail']}")
        else:
            st.caption(
                "Nenhum padrão gráfico relevante detectado no histórico recente."
            )

        # ----- Comparação com o setor -----
        if sector_cmp:
            st.markdown("**Comparação com o setor**")
            n_setor = next(iter(sector_cmp.values())).get("n_setor", "?")
            st.caption(
                f"Mediana calculada sobre {n_setor} ativos do setor "
                f"\"{linha['setor']}\" no universo analisado."
            )
            cmp_rows = []
            label_map = {
                "DY_%": "Dividend Yield (%)",
                "P/L": "P/L",
                "P/VP": "P/VP",
                "retorno_mês_%": "Retorno do mês (%)",
            }
            for metric, data in sector_cmp.items():
                d_pct = data.get("delta_pct")
                if d_pct is None:
                    sit = "—"
                elif d_pct > 15:
                    sit = f"↑ {d_pct:+.0f}% acima do setor"
                elif d_pct < -15:
                    sit = f"↓ {d_pct:+.0f}% abaixo do setor"
                else:
                    sit = f"≈ {d_pct:+.0f}% (em linha)"
                cmp_rows.append({
                    "Métrica": label_map.get(metric, metric),
                    "Ativo": f"{data['valor']:.2f}",
                    "Mediana setor": f"{data['mediana_setor']:.2f}",
                    "Posição": sit,
                })
            st.dataframe(pd.DataFrame(cmp_rows), hide_index=True, use_container_width=True)

        col_a, col_b = st.columns([2, 1])

        with col_a:
            tab_linha, tab_candle = st.tabs(["Linha", "Candles"])
            with tab_linha:
                serie = linha["_close_series"]
                st.plotly_chart(plot_price_chart(serie, escolha), use_container_width=True)
            with tab_candle:
                periodo = st.radio(
                    "Período:",
                    ["1 mês", "3 meses", "6 meses", "1 ano", "Máximo"],
                    horizontal=True,
                    key="candle_period",
                )
                days_map = {"1 mês": 22, "3 meses": 66, "6 meses": 132, "1 ano": 252, "Máximo": 9999}
                ohlcv = linha["_ohlcv_df"]
                st.plotly_chart(
                    plot_candlestick_chart(ohlcv, escolha, days_map[periodo]),
                    use_container_width=True,
                )

        with col_b:
            st.markdown("**Indicadores técnicos**")
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
            st.markdown("**Fundamentos**")
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
        st.markdown("**Decomposição do score probabilístico**")
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
    with st.expander("Ver tabela completa de todos os ativos analisados"):
        full_cols = [c for c in df.columns if not c.startswith("_")]
        st.dataframe(
            df[full_cols].sort_values("score", ascending=False),
            use_container_width=True, hide_index=True,
        )

    st.caption(
        "**Aviso:** Esta aplicação é uma ferramenta de apoio à decisão. "
        "Não constitui recomendação de investimento. Rentabilidade passada não "
        "garante rentabilidade futura."
    )


if __name__ == "__main__":
    main()
