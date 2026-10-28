#!/usr/bin/env python3
"""
Bot de signaux boursiers -> Discord, avec vérification par plusieurs sources.

Pour chaque action, le bot détecte un signal technique (moyennes mobiles + RSI), puis
le fait confirmer ou contredire par 4 autres sources indépendantes :
  - Actualité   : Yahoo Finance + Google Actualités (une info reprise par 2 sources compte double)
  - Analystes   : relèvements / abaissements de recommandation des 14 derniers jours
  - Tendance    : cours au-dessus ou en dessous de sa moyenne 200 jours
  - Marché      : S&P 500 (SPY) au-dessus ou en dessous de sa moyenne 50 jours
En plus : dates de résultats, et dépôts officiels « 8-K » de la SEC (événements majeurs).
Il ne passe AUCUN ordre : il te prévient, t'explique, et finit par un RÉSUMÉ clair
(ACHETER / ATTENDRE / ÉVITER-VENDRE) pour chaque action.

Installation :  pip install yfinance pandas numpy requests
Webhook Discord : setx DISCORD_WEBHOOK_URL "https://discord.com/api/webhooks/..."
La SEC demande de s'identifier : setx SEC_USER_AGENT "TonPrenom ton@email.com"
(puis rouvrir le terminal)

Utilisation :
  python bot_signaux.py --test        # message de test sur Discord
  python bot_signaux.py --backtest    # simule la partie technique sur 5 ans
  python bot_signaux.py               # vérifie une fois
  python bot_signaux.py --loop        # vérifie toutes les 60 min (--every 30 pour changer)

Chaque source qui ne répond pas est simplement ignorée (le terminal l'indique).
Plus de sources réduit les fausses alertes, mais aucune alerte n'est jamais sûre à 100 %.
"""
import argparse
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Actions populaires et très échangées (modifie la liste comme tu veux)
TICKERS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "AMD",
    "NFLX", "INTC", "PLTR", "BAC", "F", "PFE", "SOFI",
]
# Noms utilisés pour chercher l'actualité (évite les confusions, ex. « F » = Ford)
NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "TSLA": "Tesla", "AMZN": "Amazon",
    "GOOGL": "Google Alphabet", "META": "Meta Platforms", "AMD": "AMD", "NFLX": "Netflix",
    "INTC": "Intel", "PLTR": "Palantir", "BAC": "Bank of America", "F": "Ford Motor",
    "PFE": "Pfizer", "SOFI": "SoFi",
}

WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
SEC_UA = os.environ.get("SEC_USER_AGENT") or "bot-signaux-perso contact@example.com"
WEB_UA = "Mozilla/5.0 (compatible; bot-signaux-perso)"
STATE_FILE = Path(__file__).with_name("signaux_envoyes.json")
FEE = 0.001                 # 0,1 % de frais par ordre (pour le backtest)
NEWS_HOURS = 48             # on lit l'actualité des 48 dernières heures
NEWS_ALERT_SCORE = 3        # seuil pour une alerte « actu importante » (24 h)
ANALYST_DAYS = 14           # fenêtre pour les avis d'analystes
FILING_DAYS = 3             # fenêtre pour les dépôts officiels SEC
EARNINGS_WARN_DAYS = 7      # on mentionne les résultats s'ils sont dans moins de 7 jours
EARNINGS_CAUTION_DAYS = 3   # en dessous, la confiance du signal passe à « faible »
FOOTER = "_Signaux indicatifs, pas un conseil financier._"

# Mots-clés pour noter les titres (en anglais). On ne garde que les ÉVÉNEMENTS (nouveau produit,
# procès, avis d'analystes...) et pas les simples mouvements de prix (« stock jumps »), qui sont
# des conséquences et non des causes.
POS = {
    "launch", "launches", "launched", "unveil", "unveils", "unveiled", "introduces", "debut",
    "debuts", "beat", "beats", "record", "upgrade", "upgrades", "upgraded", "raises", "raised",
    "partnership", "wins", "won", "approval", "approved", "breakthrough", "buyback", "outperform",
    "bullish", "expands", "expansion", "milestone", "powerful",
}
NEG = {
    "lawsuit", "sues", "sued", "investigation", "probe", "fine", "fined", "recall", "recalls",
    "downgrade", "downgrades", "downgraded", "miss", "misses", "missed", "cut", "cuts", "layoffs",
    "layoff", "ban", "banned", "delay", "delays", "delayed", "fraud", "antitrust", "warning",
    "warns", "weak", "halts", "resigns", "resign", "bearish", "scandal", "breach", "hack", "hacked",
}
# Noms qui permettent de reconnaître de quelle entreprise parle un titre
ALIASES = {
    "AAPL": ["apple", "aapl"], "MSFT": ["microsoft", "msft"], "NVDA": ["nvidia", "nvda"],
    "TSLA": ["tesla", "tsla"], "AMZN": ["amazon", "amzn"], "GOOGL": ["google", "alphabet", "googl"],
    "META": ["meta", "facebook", "instagram"], "AMD": ["amd", "advanced micro devices"],
    "NFLX": ["netflix", "nflx"], "INTC": ["intel", "intc"], "PLTR": ["palantir", "pltr"],
    "BAC": ["bank of america", "bofa", "bac"], "F": ["ford"], "PFE": ["pfizer", "pfe"],
    "SOFI": ["sofi"],
}


# ---------- Données et indicateurs ----------
def fetch(ticker, period="1y"):
    for attempt in range(3):  # Yahoo refuse parfois une requête : on réessaie
        try:
            close = yf.Ticker(ticker).history(period=period, interval="1d")["Close"].dropna()
            if len(close) >= 60:
                return close
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    raise ValueError("pas assez de données (Yahoo ne répond pas ?)")


def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def compute(close):
    d = pd.DataFrame({"close": close})
    d["sma20"] = close.rolling(20).mean()
    d["sma50"] = close.rolling(50).mean()
    d["sma200"] = close.rolling(200).mean()
    d["rsi"] = rsi(close)
    up = d.sma20 > d.sma50
    d["cross_up"] = up & ~up.shift(1, fill_value=False) & d.sma50.shift(1).notna()
    d["cross_dn"] = ~up & up.shift(1, fill_value=False)
    d["rsi_up"] = (d.rsi > 30) & (d.rsi.shift(1) <= 30)   # rebond depuis la survente
    d["rsi_dn"] = (d.rsi < 70) & (d.rsi.shift(1) >= 70)   # retournement depuis la surachat
    d["buy"] = d.cross_up | d.rsi_up
    d["sell"] = d.cross_dn | d.rsi_dn
    return d


def market_trend():
    """True si le S&P 500 (SPY) est au-dessus de sa moyenne 50 jours."""
    try:
        c = fetch("SPY", "1y")
        return bool(c.iloc[-1] > c.rolling(50).mean().iloc[-1])
    except Exception as e:
        print(f"Marché (SPY) indisponible : {e}")
        return None


# ---------- Actualité (plusieurs sources, fusionnées) ----------
def title_score(title):
    words = set(re.findall(r"[a-z']+", title.lower()))
    return len(words & POS) - len(words & NEG)


def google_news(ticker):
    q = quote_plus(f"{NAMES.get(ticker, ticker)} stock")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    r = requests.get(url, headers={"User-Agent": WEB_UA}, timeout=10)
    r.raise_for_status()
    out = []
    for item in ET.fromstring(r.content).iter("item"):
        title, pub = (item.findtext("title") or "").strip(), item.findtext("pubDate")
        if not title or not pub:
            continue
        try:
            when = parsedate_to_datetime(pub)
        except Exception:
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        out.append({"title": title, "link": item.findtext("link") or "", "when": when,
                    "source": "Google Actualités"})
    return out


def yahoo_news(ticker):
    out = []
    for it in yf.Ticker(ticker).news or []:
        c = it.get("content") or it
        title = c.get("title")
        if not title:
            continue
        when = None
        if c.get("pubDate"):
            try:
                when = datetime.fromisoformat(c["pubDate"].replace("Z", "+00:00"))
            except ValueError:
                pass
        elif it.get("providerPublishTime"):
            when = datetime.fromtimestamp(it["providerPublishTime"], tz=timezone.utc)
        if when is None:
            continue
        link = (c.get("canonicalUrl") or {}).get("url") or c.get("link") or it.get("link") or ""
        out.append({"title": title, "link": link, "when": when, "source": "Yahoo Finance"})
    return out


def merge_stories(raw):
    """Regroupe les titres qui parlent de la même info (même sujet, sources différentes)."""
    stories = []
    for r in sorted(raw, key=lambda x: x["when"], reverse=True):
        k = {w for w in re.findall(r"[a-z0-9']+", r["title"].lower()) if len(w) > 3}
        for s in stories:
            if k and s["k"] and len(k & s["k"]) / len(k | s["k"]) >= 0.5:
                s["sources"].add(r["source"])
                break
        else:
            stories.append({**r, "k": k, "sources": {r["source"]}, "score": title_score(r["title"])})
    return stories


def mentions(title):
    """Entreprises suivies que le titre mentionne."""
    low = " " + re.sub(r"[^a-z0-9]+", " ", title.lower()) + " "
    return {k for k, al in ALIASES.items() if any(f" {a} " in low for a in al)}


def is_relevant(ticker, r):
    """Garde un titre s'il parle de cette entreprise et d'elle seule (pas d'un article multi-sociétés)."""
    m = mentions(r["title"])
    if r["source"] == "Yahoo Finance" and not m:
        return True  # Yahoo classe déjà ses articles par action
    return m == {ticker}


def get_news(ticker, hours=NEWS_HOURS):
    raw, now = [], datetime.now(timezone.utc)
    for name, fn in (("Yahoo Finance", yahoo_news), ("Google Actualités", google_news)):
        try:
            raw += fn(ticker)
        except Exception as e:
            print(f"  [{ticker}] {name} indisponible : {e}")
    fresh = [r for r in raw if now - r["when"] <= timedelta(hours=hours) and is_relevant(ticker, r)]
    return merge_stories(fresh)


def news_net(news):
    """Score net de l'actualité : une info reprise par plusieurs sources compte double."""
    return sum(s["score"] * (2 if len(s["sources"]) > 1 else 1) for s in news)


def tone(net):
    return "positif" if net >= 1 else "négatif" if net <= -1 else "neutre"


def headlines(news, n=3):
    lines = []
    for x in news[:n]:
        e = "🟢" if x["score"] > 0 else "🔴" if x["score"] < 0 else "⚪"
        t = x["title"].replace("[", "(").replace("]", ")")[:105]
        extra = f" ({len(x['sources'])} sources)" if len(x["sources"]) > 1 else ""
        lines.append((f"{e} [{t}](<{x['link']}>)" if x["link"] else f"{e} {t}") + extra)
    return lines


# ---------- Résultats, dépôts officiels SEC, analystes ----------
def next_earnings(ticker):
    """Date des prochains résultats trimestriels (ou None)."""
    try:
        cal = yf.Ticker(ticker).calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        today, future = datetime.now().date(), []
        for x in dates or []:
            x = x.date() if hasattr(x, "date") else x
            if x >= today:
                future.append(x)
        return min(future) if future else None
    except Exception:
        return None


def sec_filings(ticker, days=FILING_DAYS):
    """Dépôts 8-K récents (événements majeurs déclarés officiellement à la SEC)."""
    try:
        url = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
               f"&CIK={ticker}&type=8-K&dateb=&owner=include&count=10&output=atom")
        r = requests.get(url, headers={"User-Agent": SEC_UA}, timeout=10)
        r.raise_for_status()
        ns, now, out = {"a": "http://www.w3.org/2005/Atom"}, datetime.now(timezone.utc), []
        for e in ET.fromstring(r.content).findall("a:entry", ns):
            when = datetime.fromisoformat(e.findtext("a:updated", "", ns))
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            if now - when > timedelta(days=days):
                continue
            items = re.findall(r"Item \d\.\d{2}: [^<\n]+", e.findtext("a:summary", "", ns))
            link = e.find("a:link", ns)
            out.append({"when": when, "items": [i.strip() for i in items[:2]],
                        "link": link.get("href") if link is not None else ""})
        return out
    except Exception as ex:
        print(f"  [{ticker}] SEC indisponible : {ex}")
        return []


def analyst_changes(ticker, days=ANALYST_DAYS):
    """(relèvements, abaissements) de recommandation par les analystes."""
    try:
        df = yf.Ticker(ticker).upgrades_downgrades
        if df is None or len(df) == 0:
            return 0, 0
        idx = pd.to_datetime(df.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
        recent = df[idx >= pd.Timestamp.now() - pd.Timedelta(days=days)]
        act = recent["Action"].astype(str).str.lower()
        return int((act == "up").sum()), int((act == "down").sum())
    except Exception as ex:
        print(f"  [{ticker}] Analystes indisponibles : {ex}")
        return 0, 0


# ---------- Messages Discord ----------
def send_discord(text):
    if not WEBHOOK:
        raise SystemExit("DISCORD_WEBHOOK_URL n'est pas défini (voir l'en-tête du script).")
    r = requests.post(WEBHOOK, json={"content": text[:1990]}, timeout=15)
    r.raise_for_status()


def load_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def explain(sig, t, last, d, day, ctx):
    """Message clair : quoi faire, pourquoi maintenant, et ce que confirment les autres sources."""
    close = d.close
    news, earn, filings = ctx["news"], ctx["earn"], ctx["filings"]
    up, down, market_up = ctx["up"], ctx["down"], ctx["market_up"]
    chg5 = close.iloc[-1] / close.iloc[-6] - 1
    gap50 = close.iloc[-1] / last.sma50 - 1
    net = news_net(news)
    days = (earn - datetime.now().date()).days if earn else None
    want = 1 if sig == "ACHAT" else -1
    sgn = lambda x: (x > 0) - (x < 0)

    reasons = []
    if sig == "ACHAT":
        if last.cross_up:
            reasons.append("La moyenne des 20 derniers jours vient de passer au-dessus de celle des 50 derniers jours : la tendance se retourne à la hausse.")
        if last.rsi_up:
            reasons.append("L'action était très survendue (RSI sous 30) et repart à la hausse : un rebond est possible.")
        level = f"Le signal serait remis en cause sous {close.iloc[-10:].min():.2f} $ (plus bas des 10 derniers jours)."
    else:
        if last.cross_dn:
            reasons.append("La moyenne des 20 derniers jours vient de passer sous celle des 50 derniers jours : la tendance devient baissière.")
        if last.rsi_dn:
            reasons.append("L'action était très surachetée (RSI au-dessus de 70) et commence à redescendre : la hausse s'essouffle.")
        level = f"Le signal serait remis en cause au-dessus de {close.iloc[-10:].max():.2f} $ (plus haut des 10 derniers jours)."

    has200 = pd.notna(last.sma200)
    votes = [
        ("Technique", want, "le signal ci-dessus"),
        ("Actualité", 1 if net >= 1 else -1 if net <= -1 else 0,
         f"{len(news)} info(s), ton {tone(net)}" if news else "rien de marquant"),
        ("Analystes", sgn(up - down),
         f"{up} relèvement(s), {down} abaissement(s) en {ANALYST_DAYS} j" if up + down else "aucun changement récent"),
        ("Tendance de fond", (1 if last.close > last.sma200 else -1) if has200 else 0,
         ("cours au-dessus" if last.close > last.sma200 else "cours en dessous") + " de sa moyenne 200 jours" if has200 else "historique insuffisant"),
        ("Marché", 0 if market_up is None else (1 if market_up else -1),
         "indisponible" if market_up is None else "S&P 500 " + ("au-dessus" if market_up else "en dessous") + " de sa moyenne 50 jours"),
    ]
    agree = sum(v == want for _, v, _ in votes)
    clash = sum(v == -want for _, v, _ in votes)
    near = days is not None and days <= EARNINGS_CAUTION_DAYS
    if near or clash >= 2:
        conf = "faible"
    elif agree >= 4 and clash == 0:
        conf = "forte"
    elif agree >= 3:
        conf = "moyenne"
    else:
        conf = "faible"

    if conf == "faible":
        kind = "D'ACHAT" if sig == "ACHAT" else "DE VENTE"
        title = f"**🟡 SIGNAL {kind}, MAIS ATTENTION : {t}**"
    elif sig == "ACHAT":
        title = f"🟢 **ACHÈTE MAINTENANT : {t}**"
    else:
        title = f"🔴 **VENDS MAINTENANT : {t}** (si tu en détiens)"

    parts = [
        title,
        f"Prix : {last.close:.2f} $ (clôture du {day}, donc à l'ouverture de la prochaine séance)",
        f"**Confiance : {conf}** ({agree}/{len(votes)} sources d'accord, {clash} contre)",
        "**Pourquoi maintenant :**",
        *["• " + r for r in reasons],
        "**Vérification par les autres sources :**",
        *[f"{'✅' if v == want else '❌' if v == -want else '➖'} {name} : {txt}" for name, v, txt in votes],
        f"**Contexte :** {chg5:+.1%} sur 5 jours · RSI {last.rsi:.0f} · cours {gap50:+.1%} par rapport à sa moyenne 50 jours",
    ]
    if days is not None and days <= EARNINGS_WARN_DAYS:
        parts.append(f"⚠️ Résultats prévus le {earn:%d/%m} (dans {days} j) : le cours peut bouger très fort, risque élevé.")
    for f in filings[:2]:
        what = " ; ".join(f["items"]) or "événement déclaré"
        parts.append(f"📄 Dépôt officiel SEC (8-K) le {f['when']:%d/%m} : {what}")
    if news:
        parts.append("**Actus récentes :**")
        parts.extend(headlines(news))
    parts.append(level)
    return "\n".join(parts), conf


def current_votes(last, ctx):
    """Vote de chaque source sur l'état ACTUEL (+1 favorable à l'achat, -1 défavorable, 0 neutre)."""
    sgn = lambda x: (x > 0) - (x < 0)
    net = news_net(ctx["news"])
    tech = 1 if last.sma20 > last.sma50 else -1
    if last.rsi > 70:
        tech = min(tech, 0)   # déjà très monté : pas le moment d'acheter
    elif last.rsi < 30:
        tech = max(tech, 0)   # très survendu : la baisse est peut-être exagérée
    return {
        "tendance court terme": tech,
        "actu": 1 if net >= 1 else -1 if net <= -1 else 0,
        "analystes": sgn(ctx["up"] - ctx["down"]),
        "tendance 200 j": (1 if last.close > last.sma200 else -1) if pd.notna(last.sma200) else 0,
        "marché": 0 if ctx["market_up"] is None else (1 if ctx["market_up"] else -1),
    }


def make_verdict(sig, conf, votes, days):
    """Verdict simple : ACHETER / ATTENDRE / ÉVITER / VENDRE, avec la raison en une ligne."""
    total = sum(votes.values())
    pos = [k for k, v in votes.items() if v > 0]
    neg = [k for k, v in votes.items() if v < 0]
    reason = (("pour : " + ", ".join(pos)) if pos else "aucune source favorable") + \
             ((" · contre : " + ", ".join(neg)) if neg else "")
    near = days is not None and days <= EARNINGS_CAUTION_DAYS
    when = "aujourd'hui" if days == 0 else "demain" if days == 1 else f"dans {days} j"
    if sig == "VENTE":
        return "ÉVITER / VENDRE", "signal de vente du jour · " + reason
    if sig == "ACHAT" and conf in ("forte", "moyenne") and not near:
        return "ACHETER", f"signal d'achat du jour (confiance {conf}) · " + reason
    if sig == "ACHAT":
        return "ATTENDRE", "signal d'achat mais confiance faible · " + reason
    if near:
        return "ATTENDRE", f"⚠️ résultats {when}, trop risqué avant · " + reason
    if total >= 3:
        return "ACHETER", "conditions favorables · " + reason
    if total <= -3:
        return "ÉVITER / VENDRE", "conditions défavorables · " + reason
    return "ATTENDRE", "pas assez de sources d'accord · " + reason


def build_summary(rows):
    out = [f"📊 **RÉSUMÉ DU {datetime.now():%d/%m} : que faire aujourd'hui ?**"]
    for name, emoji in (("ACHETER", "🟢"), ("ATTENDRE", "🟡"), ("ÉVITER / VENDRE", "🔴")):
        grp = [r for r in rows if r["verdict"] == name]
        if grp:
            out.append(f"\n{emoji} **{name}**")
            out += [f"• **{r['t']}** ({r['price']:.2f} $) : {r['reason']}" for r in grp]
    return "\n".join(out)


SITE_DATA = Path(os.environ.get("SITE_DIR") or Path(__file__).parent) / "donnees.js"


def export_site(rows, market_up):
    """Écrit donnees.js à côté du script : c'est ce que le site (index.html) affiche."""
    data = {"updated": datetime.now().strftime("%d/%m/%Y %H:%M"), "market_up": market_up, "stocks": rows}
    SITE_DATA.write_text("window.BOT_DATA = " + json.dumps(data, ensure_ascii=False, default=str) + ";\n", encoding="utf-8")


def check_signals(force_summary=True):
    state, lines, rows = load_state(), [], []
    today, now = datetime.now().date(), datetime.now(timezone.utc)
    market_up = market_trend()
    for t in TICKERS:
        try:
            d = compute(fetch(t, "2y"))
        except Exception as e:
            print(f"{t}: ignoré ({e})")
            continue
        news, earn = get_news(t), next_earnings(t)
        up, down = analyst_changes(t)
        ctx = {"news": news, "earn": earn, "filings": sec_filings(t),
               "up": up, "down": down, "market_up": market_up}
        days = (earn - today).days if earn else None
        last, day = d.iloc[-1], str(d.index[-1].date())
        n_y = sum("Yahoo Finance" in x["sources"] for x in news)
        n_g = sum("Google Actualités" in x["sources"] for x in news)
        srcs = f"actus {len(news)} (Yahoo {n_y}, Google {n_g}) · SEC {len(ctx['filings'])} · analystes +{up}/-{down}"

        # Signal technique du jour, vérifié par les autres sources : un message détaillé
        sig, conf = None, None
        if last.buy != last.sell:
            sig = "ACHAT" if last.buy else "VENTE"
            key = f"{day}:{sig}"
            text, conf = explain(sig, t, last, d, day, ctx)
            if state.get(t) != key:
                lines.append(text)
                state[t] = key
        print(f"{t}: {'signal ' + sig if sig else 'pas de signal'} | {srcs}")

        # Verdict pour le résumé final
        votes = current_votes(last, ctx)
        verdict, reason = make_verdict(sig, conf, votes, days)
        net24 = news_net([n for n in news if now - n["when"] <= timedelta(hours=24)])
        if abs(net24) >= NEWS_ALERT_SCORE:
            reason += f" · 📰 grosse actu {tone(net24)}"
        rows.append({
            "t": t, "name": NAMES.get(t, t), "price": round(float(last.close), 2),
            "chg1": float(d.close.iloc[-1] / d.close.iloc[-2] - 1),
            "verdict": verdict, "reason": reason, "sig": sig, "conf": conf,
            "votes": {k: int(v) for k, v in votes.items()},
            "rsi": float(last.rsi), "support": float(d.close.iloc[-10:].min()),
            "resistance": float(d.close.iloc[-10:].max()),
            "closes": [round(float(x), 2) for x in d.close.iloc[-60:]],
            "news": [{"title": n["title"], "link": n["link"], "score": int(n["score"]),
                      "sources": len(n["sources"])} for n in news[:5]],
            "earn": earn.strftime("%d/%m/%Y") if earn else None, "days": days,
            "filings": [{"date": f["when"].strftime("%d/%m"), "items": f["items"], "link": f["link"]}
                        for f in ctx["filings"]],
        })

    if rows:
        export_site(rows, market_up)
    if not rows:
        raise SystemExit("Aucune action n'a pu être analysée (Yahoo bloque peut-être cette connexion).")
    summary = build_summary(rows)
    send_summary = bool(summary) and (force_summary or state.get("summary") != str(today))
    if send_summary:
        state["summary"] = str(today)
    msgs = lines + ([summary] if send_summary else [])
    for i, msg in enumerate(msgs):
        send_discord(msg + ("\n" + FOOTER if i == len(msgs) - 1 else ""))
    if msgs:
        STATE_FILE.write_text(json.dumps(state))
    print("\n" + (summary or "Aucune donnée."))
    print(f"\n{len(msgs)} message(s) envoyé(s) sur Discord.")


# ---------- Backtest (partie technique seulement : les autres sources passées ne sont pas disponibles) ----------
def backtest():
    print(f"{'Action':<7}{'Stratégie':>11}{'Acheter-garder':>16}{'Ordres':>8}")
    wins = 0
    for t in TICKERS:
        try:
            d = compute(fetch(t, "5y"))
        except Exception as e:
            print(f"{t:<7}ignoré ({e})")
            continue
        pos = pd.Series(np.nan, index=d.index)
        pos[d.sell] = 0.0
        pos[d.buy] = 1.0
        pos = pos.ffill().fillna(0.0).shift(1).fillna(0.0)  # on agit le lendemain du signal
        trades = pos.diff().abs().fillna(0.0)
        ret = d.close.pct_change().fillna(0.0) * pos - trades * FEE
        strat = (1 + ret).prod() - 1
        hold = d.close.iloc[-1] / d.close.iloc[0] - 1
        wins += strat > hold
        print(f"{t:<7}{strat:>10.0%}{hold:>15.0%}{int(trades.sum()):>9}")
    print(f"\nLa stratégie bat « acheter-garder » sur {wins}/{len(TICKERS)} actions.")
    print("Rappel : un bon résultat passé ne garantit rien pour la suite.")


# ---------- Programme principal ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true")
    p.add_argument("--backtest", action="store_true")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--every", type=int, default=60, help="minutes entre deux vérifications")
    a = p.parse_args()
    if a.test:
        send_discord("✅ Le bot de signaux est bien connecté à Discord.")
    elif a.backtest:
        backtest()
    elif a.loop:
        while True:
            check_signals(force_summary=False)  # résumé une seule fois par jour
            time.sleep(a.every * 60)
    else:
        check_signals()


if __name__ == "__main__":
    main()
