import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    HRFlowable, Table, TableStyle, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

OUTPUT_PATH = "outputs/research_note.pdf"
CHART_DIR   = "outputs"
os.makedirs(CHART_DIR, exist_ok=True)

doc = SimpleDocTemplate(
    OUTPUT_PATH,
    pagesize=A4,
    leftMargin=2.2*cm,
    rightMargin=2.2*cm,
    topMargin=2*cm,
    bottomMargin=2*cm
)

# Colour palette - clean and professional
DARK       = colors.HexColor("#1A1A2E")
ACCENT     = colors.HexColor("#185FA5")
LIGHT_GREY = colors.HexColor("#F5F5F5")
MID_GREY   = colors.HexColor("#888780")

styles = getSampleStyleSheet()

# Custom styles
title_style = ParagraphStyle(
    "CustomTitle",
    fontSize=18,
    fontName="Helvetica-Bold",
    textColor=DARK,
    spaceAfter=4,
    alignment=TA_LEFT
)
subtitle_style = ParagraphStyle(
    "Subtitle",
    fontSize=10,
    fontName="Helvetica",
    textColor=MID_GREY,
    spaceAfter=2,
    alignment=TA_LEFT
)
section_style = ParagraphStyle(
    "Section",
    fontSize=11,
    fontName="Helvetica-Bold",
    textColor=ACCENT,
    spaceBefore=14,
    spaceAfter=6,
    alignment=TA_LEFT
)
body_style = ParagraphStyle(
    "Body",
    fontSize=9.5,
    fontName="Helvetica",
    textColor=DARK,
    leading=15,
    spaceAfter=8,
    alignment=TA_JUSTIFY
)
caption_style = ParagraphStyle(
    "Caption",
    fontSize=8,
    fontName="Helvetica-Oblique",
    textColor=MID_GREY,
    spaceAfter=10,
    alignment=TA_CENTER
)
bullet_style = ParagraphStyle(
    "Bullet",
    fontSize=9.5,
    fontName="Helvetica",
    textColor=DARK,
    leading=15,
    leftIndent=14,
    spaceAfter=4,
    alignment=TA_LEFT
)

story = []

# Header line
story.append(HRFlowable(width="100%", thickness=2, color=ACCENT))
story.append(Spacer(1, 8))

# Title block
story.append(Paragraph("Detecting Credit Regimes in European Markets", title_style))
story.append(Paragraph(
    "A quantitative approach using Gaussian Mixture Models and lagged macro signals",
    subtitle_style
))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "[Your Name]  |  BSc Mathematics, Queen Mary University of London  |  June 2026",
    subtitle_style
))
story.append(Spacer(1, 6))
story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GREY))
story.append(Spacer(1, 10))


# Section 1: Motivation
story.append(Paragraph("1. Motivation", section_style))
story.append(Paragraph(
    "European credit markets do not behave uniformly over time. Periods of compressed "
    "spreads and subdued volatility, characteristic of the quantitative easing era, "
    "alternate with periods of stress and dislocation driven by macro shocks. "
    "Identifying which regime the market is in - and detecting transitions before they "
    "fully reprice - is central to active credit portfolio management.",
    body_style
))
story.append(Paragraph(
    "This note presents a systematic framework for detecting these regimes using "
    "publicly available data. The approach is motivated by the observation that "
    "the post-2022 environment represents a structural break from the decade of "
    "QE-suppressed volatility: central bank tightening, persistent inflation, and "
    "rising fiscal deficits have created conditions where tail risks in credit are "
    "underpriced relative to the macro backdrop. A regime detector that can identify "
    "these transitions early has direct implications for position sizing, hedging, "
    "and risk management in a liquid credit long/short strategy.",
    body_style
))


# Section 2: Data and Methodology
story.append(Paragraph("2. Data and Methodology", section_style))
story.append(Paragraph(
    "The analysis uses five daily time series from 2019 to 2024, sourced from the "
    "Federal Reserve Bank of St. Louis (FRED) and Yahoo Finance:",
    body_style
))

data_rows = [
    ["Variable", "Description", "Source"],
    ["EUR HY OAS", "Euro high yield option-adjusted spread (bps)", "ICE BofA / FRED"],
    ["EUR IG OAS", "Euro investment grade OAS (bps)", "ICE BofA / FRED"],
    ["VIX",        "CBOE volatility index", "FRED"],
    ["US 10y yield", "10-year Treasury yield (%)", "FRED"],
    ["EUR/USD",    "Euro/dollar exchange rate", "FRED"],
]
table = Table(data_rows, colWidths=[3.5*cm, 8*cm, 4*cm])
table.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, 0),  ACCENT),
    ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
    ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
    ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
    ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
    ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT_GREY, colors.white]),
    ("GRID",          (0, 0), (-1, -1), 0.3, MID_GREY),
    ("TOPPADDING",    (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING",   (0, 0), (-1, -1), 6),
]))
story.append(table)
story.append(Spacer(1, 8))

story.append(Paragraph(
    "Each series was tested for stationarity using the Augmented Dickey-Fuller (ADF) "
    "test. EUR HY OAS, EUR IG OAS, US 10y yield, and EUR/USD were found to be "
    "non-stationary in levels (p > 0.05) and were first-differenced prior to analysis. "
    "VIX was stationary in levels (p = 0.0003) due to its mean-reverting nature.",
    body_style
))
story.append(Paragraph(
    "Regime detection was performed using a Gaussian Mixture Model (GMM) with three "
    "components, fitted on two features: the EUR HY OAS level and log-transformed VIX. "
    "Three components were chosen to represent risk-on, transition, and stress regimes. "
    "The model was initialised with 20 random seeds to avoid local optima, with the "
    "best solution selected by log-likelihood. States were labelled by their spread "
    "level mean - higher spread level corresponds to the stress regime.",
    body_style
))
story.append(Paragraph(
    "<b>Data limitation:</b> The ICE BofA EUR HY series available on FRED provides "
    "reliable daily variation only from mid-2022 onwards. Prior to this the series is "
    "reported at lower frequency and forward-filled, which reduces the informativeness "
    "of daily spread changes in the earlier period. This is reflected in the regime "
    "output and is acknowledged as a constraint on the analysis.",
    body_style
))


# Section 3: Results
story.append(Paragraph("3. Results", section_style))
story.append(Paragraph(
    "The GMM converged successfully and identified three well-separated regimes. "
    "The regime distribution across the 2019-2024 sample is as follows:",
    body_style
))

regime_rows = [
    ["Regime",      "Days", "Share", "Avg Spread (bps)", "Interpretation"],
    ["Risk-on",     "1149", "73.5%", "Low / stable",    "QE-era compression, calm conditions"],
    ["Transition",  "141",  "9.0%",  "Moving",          "Regime shift in progress"],
    ["Stress",      "274",  "17.5%", "Wide / volatile", "Post-2022 repricing period"],
]
table2 = Table(regime_rows, colWidths=[2.5*cm, 1.5*cm, 1.5*cm, 3.5*cm, 6.5*cm])
table2.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, 0),  ACCENT),
    ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
    ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
    ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
    ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
    ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT_GREY, colors.white]),
    ("GRID",          (0, 0), (-1, -1), 0.3, MID_GREY),
    ("TOPPADDING",    (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING",   (0, 0), (-1, -1), 6),
]))
story.append(table2)
story.append(Spacer(1, 8))

# Insert regime chart
regime_chart = os.path.join(CHART_DIR, "regimes.png")
if os.path.exists(regime_chart):
    story.append(Image(regime_chart, width=16*cm, height=9*cm))
    story.append(Paragraph(
        "Figure 1. EUR HY OAS and VIX coloured by detected regime (green = risk-on, "
        "orange = transition, red = stress). The structural break in mid-2023 is "
        "clearly visible as the market shifts from the dominant risk-on regime.",
        caption_style
    ))

story.append(PageBreak())


# Section 4: Macro Linkage
story.append(Paragraph("4. Macro Leading Indicators", section_style))
story.append(Paragraph(
    "Having identified the regimes, the key analytical question is whether macro "
    "variables observed prior to a regime transition can predict it in advance. "
    "A logistic regression was fitted using lagged values of all five macro variables "
    "at horizons of 5, 10, and 20 trading days, with the GMM regime label as the "
    "dependent variable.",
    body_style
))
story.append(Paragraph("The key findings are:", body_style))
story.append(Paragraph(
    "- <b>Short lags dominate.</b> Predictive accuracy is highest at the 5-day lag "
    "(98.9%) and declines modestly at 10 days (98.6%) and 20 days (97.6%), suggesting "
    "regime signals are relatively short-lived and markets reprice quickly once a "
    "transition begins.",
    bullet_style
))
story.append(Paragraph(
    "- <b>VIX leads risk-on regimes.</b> The 5 and 10-day lagged VIX are the strongest "
    "predictors of calm credit conditions, with coefficients of 1.77 and 1.42 "
    "respectively. Low volatility in the preceding week is a reliable signal that "
    "credit spreads will remain compressed.",
    bullet_style
))
story.append(Paragraph(
    "- <b>Yield changes lead transitions.</b> The 5 and 10-day lagged yield is the "
    "strongest predictor of regime transitions, with coefficients of 1.40 and 1.27. "
    "This supports the view that rate moves precede credit regime shifts - a finding "
    "consistent with the post-2022 experience where aggressive central bank tightening "
    "drove the structural break in credit markets.",
    bullet_style
))
story.append(Paragraph(
    "- <b>Spread persistence predicts stress.</b> Lagged HY and IG spreads are the "
    "strongest predictors of the stress regime, confirming that credit stress is "
    "persistent - wide spreads today predict wide spreads in the near term.",
    bullet_style
))
story.append(Spacer(1, 6))

prob_chart = os.path.join(CHART_DIR, "regime_probabilities.png")
if os.path.exists(prob_chart):
    story.append(Image(prob_chart, width=16*cm, height=7*cm))
    story.append(Paragraph(
        "Figure 2. Stacked regime probabilities derived from lagged macro variables. "
        "The model assigns high confidence to risk-on through 2022, identifies a "
        "clear transition in mid-2023, and shifts to stress through 2024.",
        caption_style
    ))


# Section 5: Implications and limitations
story.append(Paragraph("5. Implications and Limitations", section_style))
story.append(Paragraph(
    "The regime framework has direct applications for a liquid credit long/short "
    "portfolio. A manager observing a transition signal - rising yields over the "
    "preceding week alongside elevated VIX - could reduce gross exposure, increase "
    "short credit positions via iTraxx index hedges, or shift the portfolio toward "
    "higher-quality credits before spreads fully reprice. The 5-day lag finding "
    "suggests this signal provides approximately one trading week of lead time.",
    body_style
))
story.append(Paragraph(
    "Several limitations constrain the current analysis. First, the in-sample "
    "accuracy figures overstate predictive power since the model is evaluated on "
    "the data it was trained on. A proper out-of-sample validation with a held-out "
    "test period would be the natural next step. Second, the data limitation in the "
    "EUR HY series prior to mid-2022 means the risk-on regime is partly identified "
    "by the absence of daily variation rather than genuine spread compression signals. "
    "Access to proper iTraxx Main and Crossover daily data would significantly improve "
    "the pre-2022 regime identification. Third, the GMM does not impose temporal "
    "ordering on regimes - a Hidden Markov Model with transition probabilities would "
    "better capture the sequential nature of regime changes and reduce day-to-day "
    "switching noise.",
    body_style
))
story.append(Paragraph(
    "Extensions to this framework could include incorporating PMI and ECB balance "
    "sheet data as additional macro factors, testing whether the regime signal adds "
    "value over a simple spread-level threshold rule, and applying the model in a "
    "rolling out-of-sample setting to simulate real-time regime detection.",
    body_style
))

story.append(Spacer(1, 10))
story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GREY))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "Data sources: ICE BofA Indices via FRED (Federal Reserve Bank of St. Louis), "
    "CBOE VIX via FRED. All analysis conducted in Python using pandas, scikit-learn, "
    "and matplotlib. Code available at: [your GitHub link]",
    caption_style
))

doc.build(story)
print(f"Research note saved to {OUTPUT_PATH}")