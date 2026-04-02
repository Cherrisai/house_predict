"""
app.py -- Real-Time House Price Prediction (India)
Streamlit UI  |  Random Forest / XGBoost  |  Future Projections  |  Buying Suggestions
"""

import pickle, os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────── page config ───────────────────────────
st.set_page_config(
    page_title="House Price Predictor ",
    page_icon="",
    layout="wide",
)

# ─────────────────────────── custom CSS ────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .header-band {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem; border-radius: 14px; margin-bottom: 1.8rem;
    }
    .header-band h1 {
        color: #fff; font-size: 2rem; margin: 0;
        font-weight: 700; letter-spacing: -0.5px;
    }
    .header-band p { color: #b0b0d0; margin: 0.3rem 0 0; font-size: 0.95rem; }

    .pred-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px; padding: 1.4rem 1.6rem; text-align: center;
    }
    .pred-card .label {
        color: #8888aa; font-size: 0.78rem;
        text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.35rem;
    }
    .pred-card .value     { font-size: 1.55rem; font-weight: 700; color: #7ef9a0; }
    .pred-card .value.accent { color: #64b5f6; }
    .pred-card .value.warn   { color: #ffd54f; }
    .pred-card .value.profit { color: #69f0ae; }
    .pred-card .value.future { color: #ce93d8; }

    .section-title {
        font-size: 1.05rem; font-weight: 700; color: #e0e0e0;
        margin: 1.5rem 0 0.8rem; padding-bottom: 0.4rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }

    /* suggestion cards */
    .sug-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 0.8rem;
    }
    .sug-card .sug-area {
        color: #e0e0e0; font-size: 1rem; font-weight: 700; margin-bottom: 0.3rem;
    }
    .sug-card .sug-row {
        display: flex; justify-content: space-between; margin: 0.15rem 0;
    }
    .sug-card .sug-label { color: #8888aa; font-size: 0.82rem; }
    .sug-card .sug-val   { color: #b0b0d0; font-size: 0.85rem; font-weight: 500; }
    .sug-card .sug-val.green  { color: #69f0ae; }
    .sug-card .sug-val.blue   { color: #64b5f6; }
    .sug-card .sug-val.purple { color: #ce93d8; }
    .sug-badge-fit {
        display: inline-block; background: rgba(105,240,174,0.15);
        color: #69f0ae; border-radius: 6px; padding: 2px 10px;
        font-size: 0.75rem; font-weight: 700; margin-top: 0.3rem;
    }
    .sug-badge-stretch {
        display: inline-block; background: rgba(255,213,79,0.15);
        color: #ffd54f; border-radius: 6px; padding: 2px 10px;
        font-size: 0.75rem; font-weight: 700; margin-top: 0.3rem;
    }
    .sug-badge-over {
        display: inline-block; background: rgba(239,154,154,0.15);
        color: #ef9a9a; border-radius: 6px; padding: 2px 10px;
        font-size: 0.75rem; font-weight: 700; margin-top: 0.3rem;
    }

    section[data-testid="stSidebar"] > div { background: #0f0c29; }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  FORMATTERS  —  all amounts in Cr / L / K
# ═══════════════════════════════════════════════════════════════════

def fmt(val):
    """Format rupee amount → 1.25 Cr / 45.3L / 980K / 5K."""
    val = float(val)
    if val >= 1e7:
        return f"{val/1e7:.2f} Cr"
    elif val >= 1e5:
        return f"{val/1e5:.1f}L"
    elif val >= 1e3:
        return f"{val/1e3:.0f}K"
    else:
        return f"{val:,.0f}"


def fmt_rs(val):
    """Same as fmt() but prefixed with rupee sign."""
    return f"\u20B9{fmt(val)}"


def fmt_chart(val):
    """Short label for chart bars / text annotations."""
    val = float(val)
    if val >= 1e7:
        return f"\u20B9{val/1e7:.1f}Cr"
    elif val >= 1e5:
        return f"\u20B9{val/1e5:.0f}L"
    elif val >= 1e3:
        return f"\u20B9{val/1e3:.0f}K"
    else:
        return f"\u20B9{val:,.0f}"


def fmt_k(val):
    """Format per-sqft prices (typically thousands range)."""
    val = float(val)
    if val >= 1e5:
        return f"\u20B9{val/1e5:.1f}L"
    elif val >= 1e3:
        return f"\u20B9{val/1e3:.1f}K"
    else:
        return f"\u20B9{val:,.0f}"


# ═══════════════════════════════════════════════════════════════════
#  DATA MAPPINGS
# ═══════════════════════════════════════════════════════════════════

DATA_PATH  = "data.csv"
MODEL_PATH = "model.pkl"

CITY_AREAS = {
    "Bangalore":  ["Whitefield", "Sarjapur Road", "Electronic City", "Hebbal", "Yelahanka"],
    "Chennai":    ["OMR", "Medavakkam", "Ambattur", "Chromepet", "Pallavaram"],
    "Mumbai":     ["Andheri East", "Borivali", "Chembur", "Worli", "Lower Parel"],
    "Hyderabad":  ["Gachibowli", "Kondapur", "Miyapur", "Kukatpally", "HITEC City"],
    "Delhi":      ["Dwarka", "Rohini", "Greater Kailash", "Uttam Nagar", "Saket"],
}

CITY_BASE_PRICE = {
    "Bangalore": 11000, "Chennai": 9300,
    "Mumbai": 12000, "Hyderabad": 6000, "Delhi": 8400,
}

CITY_APPRECIATION = {
    "Bangalore":  8.5, "Chennai": 6.5,
    "Mumbai":     5.0, "Hyderabad": 11.0, "Delhi": 7.0,
}

AREA_APPRECIATION_OFFSET = {
    "Bangalore": {
        "Whitefield": 1.5, "Sarjapur Road": 2.0, "Electronic City": 0.5,
        "Hebbal": 1.0, "Yelahanka": 1.8,
    },
    "Chennai": {
        "OMR": 2.0, "Medavakkam": 1.0, "Ambattur": 0.5,
        "Chromepet": 0.3, "Pallavaram": 0.8,
    },
    "Mumbai": {
        "Andheri East": 0.5, "Borivali": 1.0, "Chembur": 1.2,
        "Worli": -0.5, "Lower Parel": -0.3,
    },
    "Hyderabad": {
        "Gachibowli": 1.5, "Kondapur": 1.2, "Miyapur": 2.5,
        "Kukatpally": 0.8, "HITEC City": 1.0,
    },
    "Delhi": {
        "Dwarka": 1.0, "Rohini": 0.8, "Greater Kailash": -0.5,
        "Uttam Nagar": 2.0, "Saket": 0.5,
    },
}


# ═══════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def get_appreciation_rate(city, area):
    base = CITY_APPRECIATION[city]
    offset = AREA_APPRECIATION_OFFSET.get(city, {}).get(area, 0)
    return base + offset


def project_price(current_price, rate_pct, years):
    return current_price * ((1 + rate_pct / 100) ** years)


def build_projection_table(current_price, city, area, max_years=15):
    rate = get_appreciation_rate(city, area)
    rows = []
    for y in range(0, max_years + 1):
        future = project_price(current_price, rate, y)
        rows.append({
            "year": 2026 + y, "years_from_now": y,
            "projected_price": future,
            "total_gain": future - current_price,
            "gain_pct": ((future / current_price) - 1) * 100,
            "annual_rate": rate,
        })
    return pd.DataFrame(rows)


def build_city_comparison(sqft, bhk, bath, model_pipe, max_years=15):
    rows = []
    for c, areas in CITY_AREAS.items():
        for a in areas:
            inp = pd.DataFrame([{
                "city": c, "area": a,
                "sqft": sqft, "bhk": bhk, "bathrooms": bath,
            }])
            current = model_pipe.predict(inp)[0]
            rate = get_appreciation_rate(c, a)
            for y in range(0, max_years + 1):
                future = project_price(current, rate, y)
                rows.append({
                    "city": c, "area": a,
                    "year": 2026 + y, "years_from_now": y,
                    "current_price": current,
                    "projected_price": future,
                    "gain_pct": ((future / current) - 1) * 100,
                    "rate": rate,
                })
    return pd.DataFrame(rows)


def generate_buying_suggestions(budget, city, model_pipe, future_years=5):
    """
    For a given budget and city, predict prices for every BHK + area combo
    and return matches that are within budget, slight stretch, or over budget.
    """
    results = []
    areas = CITY_AREAS[city]

    for area in areas:
        for bhk in [1, 2, 3, 4, 5]:
            # pick typical sqft for bhk
            typical_sqft = {1: 550, 2: 950, 3: 1400, 4: 2100, 5: 3200}[bhk]
            typical_bath = min(bhk, bhk)

            inp = pd.DataFrame([{
                "city": city, "area": area,
                "sqft": typical_sqft, "bhk": bhk, "bathrooms": typical_bath,
            }])
            price = model_pipe.predict(inp)[0]

            rate = get_appreciation_rate(city, area)
            future = project_price(price, rate, future_years)
            profit = future - price
            profit_pct = ((future / price) - 1) * 100

            price_per_sqft = price / typical_sqft

            # affordability tag
            ratio = price / budget
            if ratio <= 1.0:
                tag = "WITHIN BUDGET"
                tag_class = "sug-badge-fit"
            elif ratio <= 1.15:
                tag = "SLIGHT STRETCH"
                tag_class = "sug-badge-stretch"
            else:
                tag = "OVER BUDGET"
                tag_class = "sug-badge-over"

            results.append({
                "area": area,
                "bhk": bhk,
                "sqft": typical_sqft,
                "bathrooms": typical_bath,
                "price": price,
                "price_per_sqft": price_per_sqft,
                "future_price": future,
                "profit": profit,
                "profit_pct": profit_pct,
                "rate": rate,
                "tag": tag,
                "tag_class": tag_class,
                "ratio": ratio,
            })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════
#  LOAD RESOURCES
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("model.pkl not found. Run `python train.py` first.")
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error("data.csv not found. Run `python generate_data.py` first.")
        st.stop()
    return pd.read_csv(DATA_PATH)


model = load_model()
df    = load_data()


# ═══════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="header-band">
    <h1>House Price Predictor</h1>
    <p>Future projections | Buying suggestions</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### Property Details")
    city = st.selectbox("City", list(CITY_AREAS.keys()))
    area = st.selectbox("Area / Micro-market", CITY_AREAS[city])

    sqft = st.slider("Area (sq ft)", 400, 5000, 1200, step=50)
    bhk  = st.selectbox("BHK", [1, 2, 3, 4, 5], index=2)
    bath = st.selectbox("Bathrooms", [1, 2, 3, 4, 5], index=1)

    st.markdown("---")
    st.markdown("### Future Projection")
    future_year = st.slider(
        "Project price up to year",
        min_value=2027, max_value=2041, value=2031, step=1,
    )
    max_years = future_year - 2026

    st.markdown("---")
    st.markdown("### Buying Suggestions (Optional)")
    enable_suggestions = st.checkbox("Enable buying suggestions", value=False)
    budget_lakhs = st.number_input(
        "Your budget (in Lakhs)",
        min_value=10.0, max_value=5000.0, value=80.0, step=5.0,
        disabled=not enable_suggestions,
    )
    budget = budget_lakhs * 1e5

    st.markdown("---")
    predict_btn = st.button("Predict Price", use_container_width=True, type="primary")


# ═══════════════════════════════════════════════════════════════════
#  PLOTLY DARK LAYOUT DEFAULTS
# ═══════════════════════════════════════════════════════════════════

LAYOUT_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)


# ═══════════════════════════════════════════════════════════════════
#  MAIN OUTPUT -- after Predict is clicked
# ═══════════════════════════════════════════════════════════════════

if predict_btn:

    input_df = pd.DataFrame([{
        "city": city, "area": area,
        "sqft": sqft, "bhk": bhk, "bathrooms": bath,
    }])
    predicted_price = model.predict(input_df)[0]
    price_per_sqft  = predicted_price / sqft
    city_avg        = CITY_BASE_PRICE[city]
    diff_pct        = ((price_per_sqft - city_avg) / city_avg) * 100

    rate         = get_appreciation_rate(city, area)
    future_price = project_price(predicted_price, rate, max_years)
    total_profit = future_price - predicted_price
    profit_pct   = ((future_price / predicted_price) - 1) * 100

    # ──────────────────────────────────────────────────────────────
    #  SECTION 1: Current Price Cards
    # ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Current Price Estimate</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="pred-card">
            <div class="label">Predicted Price (2026)</div>
            <div class="value">{fmt_rs(predicted_price)}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="pred-card">
            <div class="label">Price / sq ft</div>
            <div class="value accent">{fmt_k(price_per_sqft)}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        arrow = "+" if diff_pct >= 0 else ""
        st.markdown(f"""
        <div class="pred-card">
            <div class="label">vs City Avg ({fmt_k(city_avg)}/sqft)</div>
            <div class="value warn">{arrow}{diff_pct:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────
    #  SECTION 2: Future Price Cards
    # ──────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-title">Future Projection -- {area}, {city} in {future_year}</div>',
        unsafe_allow_html=True,
    )
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown(f"""
        <div class="pred-card">
            <div class="label">Projected Price ({future_year})</div>
            <div class="value future">{fmt_rs(future_price)}</div>
        </div>""", unsafe_allow_html=True)
    with f2:
        st.markdown(f"""
        <div class="pred-card">
            <div class="label">Total Profit if Bought Now</div>
            <div class="value profit">{fmt_rs(total_profit)}</div>
        </div>""", unsafe_allow_html=True)
    with f3:
        st.markdown(f"""
        <div class="pred-card">
            <div class="label">Return on Investment</div>
            <div class="value profit">+{profit_pct:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with f4:
        st.markdown(f"""
        <div class="pred-card">
            <div class="label">Annual Appreciation Rate</div>
            <div class="value accent">{rate:.1f}% / yr</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────
    #  SECTION 3: Buy Now vs Hold Chart
    # ──────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-title">Buy Now vs Hold -- {area}, {city}</div>',
        unsafe_allow_html=True,
    )

    proj_df = build_projection_table(predicted_price, city, area, max_years)

    fig_proj = go.Figure()
    fig_proj.add_trace(go.Scatter(
        x=proj_df["year"], y=proj_df["projected_price"],
        mode="lines+markers", name="Projected Price",
        line=dict(color="#ce93d8", width=3), marker=dict(size=6),
        hovertemplate="Year %{x}<br>Price: %{y:,.0f}<extra></extra>",
    ))
    fig_proj.add_hline(
        y=predicted_price, line_dash="dash", line_color="#7ef9a0",
        annotation_text=f"Buy Now: {fmt_rs(predicted_price)}",
        annotation_font_color="#7ef9a0", annotation_position="top left",
    )
    fig_proj.add_trace(go.Scatter(
        x=proj_df["year"], y=[predicted_price] * len(proj_df),
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_proj.add_trace(go.Scatter(
        x=proj_df["year"], y=proj_df["projected_price"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        fill="tonexty", fillcolor="rgba(206,147,216,0.15)",
    ))
    fig_proj.add_annotation(
        x=future_year, y=future_price,
        text=f"Profit: {fmt_rs(total_profit)} (+{profit_pct:.1f}%)",
        showarrow=True, arrowhead=2, arrowcolor="#69f0ae",
        font=dict(color="#69f0ae", size=12), ax=0, ay=-40,
    )
    fig_proj.update_layout(
        **LAYOUT_DARK,
        height=380, margin=dict(l=10, r=10, t=10, b=40),
        xaxis_title="Year", yaxis_title="Price (Rs)",
        yaxis=dict(tickformat=","),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_proj, use_container_width=True)

    # ──────────────────────────────────────────────────────────────
    #  SECTION 4: City Growth Comparison
    # ──────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-title">'
        f'City-wise Growth Comparison -- Same Property in {future_year}'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Showing what a {bhk}BHK, {sqft} sqft property would cost across all cities "
        f"and how much it appreciates by {future_year}."
    )

    comp_df = build_city_comparison(sqft, bhk, bath, model, max_years)

    year_options = sorted(comp_df["year"].unique())
    selected_filter_year = st.select_slider(
        "Filter by year", options=year_options, value=future_year,
    )

    snap = comp_df[comp_df["year"] == selected_filter_year].copy()

    city_snap = (
        snap.groupby("city")
        .agg(
            avg_current=("current_price", "mean"),
            avg_projected=("projected_price", "mean"),
            avg_gain_pct=("gain_pct", "mean"),
            avg_rate=("rate", "mean"),
        )
        .reset_index()
        .sort_values("avg_gain_pct", ascending=True)
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"**City Avg Price -- Now vs {selected_filter_year}**")
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            y=city_snap["city"], x=city_snap["avg_current"],
            orientation="h", name="Current (2026)",
            marker_color="#3d3d6b",
            text=city_snap["avg_current"].apply(fmt_chart),
            textposition="inside",
        ))
        fig_comp.add_trace(go.Bar(
            y=city_snap["city"], x=city_snap["avg_projected"],
            orientation="h", name=f"Projected ({selected_filter_year})",
            marker_color="#ce93d8",
            text=city_snap["avg_projected"].apply(fmt_chart),
            textposition="inside",
        ))
        fig_comp.update_layout(
            barmode="group", **LAYOUT_DARK,
            height=350, margin=dict(l=0, r=10, t=10, b=10),
            xaxis=dict(visible=False),
            yaxis=dict(tickfont=dict(size=13)),
            legend=dict(orientation="h", y=-0.12),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    with col_b:
        st.markdown(f"**Growth % by City -- by {selected_filter_year}**")
        bar_colors = ["#7ef9a0" if c == city else "#64b5f6"
                      for c in city_snap["city"]]
        fig_growth = go.Figure(go.Bar(
            y=city_snap["city"], x=city_snap["avg_gain_pct"],
            orientation="h", marker_color=bar_colors,
            text=city_snap["avg_gain_pct"].apply(lambda v: f"+{v:.1f}%"),
            textposition="outside",
        ))
        fig_growth.update_layout(
            **LAYOUT_DARK,
            height=350, margin=dict(l=0, r=60, t=10, b=10),
            xaxis=dict(visible=False),
            yaxis=dict(tickfont=dict(size=13)),
        )
        st.plotly_chart(fig_growth, use_container_width=True)

    # ──────────────────────────────────────────────────────────────
    #  SECTION 5: Area-wise Future within City
    # ──────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-title">'
        f'Area-wise Projection -- {city} in {selected_filter_year}'
        f'</div>',
        unsafe_allow_html=True,
    )

    area_snap = (
        snap[snap["city"] == city]
        .sort_values("projected_price", ascending=True)
        .copy()
    )

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Current vs Projected by Area**")
        fig_area = go.Figure()
        fig_area.add_trace(go.Bar(
            y=area_snap["area"], x=area_snap["current_price"],
            orientation="h", name="Current (2026)",
            marker_color="#3d3d6b",
            text=area_snap["current_price"].apply(fmt_chart),
            textposition="inside",
        ))
        fig_area.add_trace(go.Bar(
            y=area_snap["area"], x=area_snap["projected_price"],
            orientation="h", name=f"Projected ({selected_filter_year})",
            marker_color="#64b5f6",
            text=area_snap["projected_price"].apply(fmt_chart),
            textposition="inside",
        ))
        fig_area.update_layout(
            barmode="group", **LAYOUT_DARK,
            height=350, margin=dict(l=0, r=10, t=10, b=10),
            xaxis=dict(visible=False),
            yaxis=dict(tickfont=dict(size=12)),
            legend=dict(orientation="h", y=-0.12),
        )
        st.plotly_chart(fig_area, use_container_width=True)

    with col_r:
        st.markdown("**Profit if Bought Now (by Area)**")
        area_snap["profit"] = area_snap["projected_price"] - area_snap["current_price"]
        p_colors = ["#69f0ae" if a == area else "#4db6ac"
                    for a in area_snap["area"]]
        fig_profit = go.Figure(go.Bar(
            y=area_snap["area"], x=area_snap["profit"],
            orientation="h", marker_color=p_colors,
            text=area_snap["profit"].apply(fmt_chart),
            textposition="outside",
        ))
        fig_profit.update_layout(
            **LAYOUT_DARK,
            height=350, margin=dict(l=0, r=70, t=10, b=10),
            xaxis=dict(visible=False),
            yaxis=dict(tickfont=dict(size=12)),
        )
        st.plotly_chart(fig_profit, use_container_width=True)

    # ──────────────────────────────────────────────────────────────
    #  SECTION 6: Year-by-Year Growth Lines (All Cities)
    # ──────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-title">Year-by-Year Growth Trend -- All Cities</div>',
        unsafe_allow_html=True,
    )

    city_yearly = (
        comp_df.groupby(["city", "year"])["projected_price"]
        .mean().reset_index()
    )

    fig_lines = px.line(
        city_yearly, x="year", y="projected_price", color="city",
        labels={"projected_price": "Avg Projected Price (Rs)", "year": "Year"},
        color_discrete_sequence=[
            "#7ef9a0", "#64b5f6", "#ce93d8", "#ffd54f", "#ef9a9a"
        ],
    )
    fig_lines.update_layout(
        **LAYOUT_DARK,
        height=380, margin=dict(l=10, r=10, t=10, b=40),
        yaxis=dict(tickformat=","),
        legend=dict(orientation="h", y=-0.15, title_text=""),
    )
    fig_lines.update_traces(line=dict(width=2.5))
    st.plotly_chart(fig_lines, use_container_width=True)

    # ──────────────────────────────────────────────────────────────
    #  SECTION 7: Price Distribution
    # ──────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-title">Price Distribution -- {city}</div>',
        unsafe_allow_html=True,
    )
    city_prices = df[df["city"] == city]["price"] / 1e5
    fig_hist = px.histogram(
        city_prices, nbins=40,
        labels={"value": "Price (Lakhs)", "count": "Properties"},
        color_discrete_sequence=["#7e57c2"],
    )
    fig_hist.add_vline(
        x=predicted_price / 1e5,
        line_dash="dash", line_color="#7ef9a0",
        annotation_text=f"Your prediction: {fmt_rs(predicted_price)}",
        annotation_font_color="#7ef9a0",
    )
    fig_hist.update_layout(
        **LAYOUT_DARK,
        height=300, margin=dict(l=0, r=0, t=10, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ──────────────────────────────────────────────────────────────
    #  SECTION 8: BUYING SUGGESTIONS (optional)
    # ──────────────────────────────────────────────────────────────
    if enable_suggestions:
        st.markdown(
            f'<div class="section-title">'
            f'Buying Suggestions -- {city} | Budget: {fmt_rs(budget)}'
            f'</div>',
            unsafe_allow_html=True,
        )

        sug_df = generate_buying_suggestions(budget, city, model, max_years)

        # ── Summary cards ──
        within   = sug_df[sug_df["tag"] == "WITHIN BUDGET"]
        stretch  = sug_df[sug_df["tag"] == "SLIGHT STRETCH"]

        affordable_count = len(within)
        stretch_count    = len(stretch)

        cheapest = sug_df.loc[sug_df["price"].idxmin()]
        best_roi = sug_df[sug_df["tag"] == "WITHIN BUDGET"]
        if len(best_roi) > 0:
            best_roi = best_roi.loc[best_roi["profit_pct"].idxmax()]
        else:
            best_roi = sug_df.loc[sug_df["profit_pct"].idxmax()]

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(f"""
            <div class="pred-card">
                <div class="label">Within Budget</div>
                <div class="value">{affordable_count} options</div>
            </div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""
            <div class="pred-card">
                <div class="label">Slight Stretch</div>
                <div class="value warn">{stretch_count} options</div>
            </div>""", unsafe_allow_html=True)
        with s3:
            st.markdown(f"""
            <div class="pred-card">
                <div class="label">Cheapest Option</div>
                <div class="value accent">{cheapest['bhk']}BHK {cheapest['area']}<br>
                    <span style="font-size:0.9rem">{fmt_rs(cheapest['price'])}</span>
                </div>
            </div>""", unsafe_allow_html=True)
        with s4:
            st.markdown(f"""
            <div class="pred-card">
                <div class="label">Best ROI (in budget)</div>
                <div class="value profit">{best_roi['bhk']}BHK {best_roi['area']}<br>
                    <span style="font-size:0.9rem">+{best_roi['profit_pct']:.1f}% in {max_years}yr</span>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Filter controls ──
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            filter_tag = st.multiselect(
                "Filter by affordability",
                ["WITHIN BUDGET", "SLIGHT STRETCH", "OVER BUDGET"],
                default=["WITHIN BUDGET", "SLIGHT STRETCH"],
            )
        with filter_col2:
            filter_bhk = st.multiselect(
                "Filter by BHK",
                [1, 2, 3, 4, 5],
                default=[1, 2, 3],
            )

        filtered = sug_df[
            (sug_df["tag"].isin(filter_tag)) &
            (sug_df["bhk"].isin(filter_bhk))
        ].sort_values("price", ascending=True)

        if len(filtered) == 0:
            st.warning("No properties match these filters. Try expanding your budget or BHK selection.")
        else:
            # ── Suggestion cards ──
            st.markdown(f"**Showing {len(filtered)} property options in {city}**")

            # render in 2-column grid
            card_cols = st.columns(2)
            for idx, (_, row) in enumerate(filtered.iterrows()):
                with card_cols[idx % 2]:
                    st.markdown(f"""
                    <div class="sug-card">
                        <div class="sug-area">{row['bhk']}BHK in {row['area']}</div>
                        <div class="sug-row">
                            <span class="sug-label">Estimated Price</span>
                            <span class="sug-val green">{fmt_rs(row['price'])}</span>
                        </div>
                        <div class="sug-row">
                            <span class="sug-label">Size</span>
                            <span class="sug-val">{row['sqft']} sqft  |  {row['bathrooms']} bath</span>
                        </div>
                        <div class="sug-row">
                            <span class="sug-label">Price / sqft</span>
                            <span class="sug-val">{fmt_k(row['price_per_sqft'])}</span>
                        </div>
                        <div class="sug-row">
                            <span class="sug-label">Future Price ({future_year})</span>
                            <span class="sug-val purple">{fmt_rs(row['future_price'])}</span>
                        </div>
                        <div class="sug-row">
                            <span class="sug-label">Profit in {max_years} yrs</span>
                            <span class="sug-val green">{fmt_rs(row['profit'])} (+{row['profit_pct']:.1f}%)</span>
                        </div>
                        <div class="sug-row">
                            <span class="sug-label">Appreciation</span>
                            <span class="sug-val blue">{row['rate']:.1f}% / yr</span>
                        </div>
                        <span class="{row['tag_class']}">{row['tag']}</span>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Budget vs Price chart ──
            st.markdown(f"**Budget ({fmt_rs(budget)}) vs Property Prices -- {city}**")

            chart_data = filtered.copy()
            chart_data["label"] = chart_data.apply(
                lambda r: f"{r['bhk']}BHK {r['area']}", axis=1
            )
            chart_data = chart_data.sort_values("price", ascending=True)

            bar_c = []
            for _, r in chart_data.iterrows():
                if r["tag"] == "WITHIN BUDGET":
                    bar_c.append("#69f0ae")
                elif r["tag"] == "SLIGHT STRETCH":
                    bar_c.append("#ffd54f")
                else:
                    bar_c.append("#ef9a9a")

            fig_sug = go.Figure()
            fig_sug.add_trace(go.Bar(
                y=chart_data["label"], x=chart_data["price"],
                orientation="h", marker_color=bar_c,
                text=chart_data["price"].apply(fmt_chart),
                textposition="outside",
                name="Estimated Price",
            ))
            fig_sug.add_vline(
                x=budget, line_dash="dash", line_color="#ffffff", line_width=2,
                annotation_text=f"Budget: {fmt_rs(budget)}",
                annotation_font_color="#ffffff",
                annotation_position="top right",
            )
            fig_sug.update_layout(
                **LAYOUT_DARK,
                height=max(300, len(chart_data) * 35 + 80),
                margin=dict(l=0, r=80, t=10, b=10),
                xaxis=dict(visible=False),
                yaxis=dict(tickfont=dict(size=11)),
                showlegend=False,
            )
            st.plotly_chart(fig_sug, use_container_width=True)

            # ── Profit comparison chart ──
            st.markdown(f"**Projected Profit by {future_year} -- Within Budget Options**")

            budget_opts = filtered[filtered["tag"] == "WITHIN BUDGET"].copy()
            if len(budget_opts) == 0:
                st.info("No options within budget to show profit comparison.")
            else:
                budget_opts["label"] = budget_opts.apply(
                    lambda r: f"{r['bhk']}BHK {r['area']}", axis=1
                )
                budget_opts = budget_opts.sort_values("profit", ascending=True)

                fig_sp = go.Figure()
                fig_sp.add_trace(go.Bar(
                    y=budget_opts["label"], x=budget_opts["profit"],
                    orientation="h",
                    marker_color="#69f0ae",
                    text=budget_opts.apply(
                        lambda r: f"{fmt_chart(r['profit'])} (+{r['profit_pct']:.0f}%)",
                        axis=1,
                    ),
                    textposition="outside",
                ))
                fig_sp.update_layout(
                    **LAYOUT_DARK,
                    height=max(280, len(budget_opts) * 35 + 80),
                    margin=dict(l=0, r=120, t=10, b=10),
                    xaxis=dict(visible=False),
                    yaxis=dict(tickfont=dict(size=11)),
                )
                st.plotly_chart(fig_sp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
#  LANDING STATE -- before Predict is clicked
# ═══════════════════════════════════════════════════════════════════
else:
    st.info(
        "Configure property details in the sidebar and click "
        "**Predict Price** to get started."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cities", "5")
    m2.metric("Micro-markets", "25")
    m3.metric("Training Samples", f"{len(df):,}")
    m4.metric("ML Model", "RF / XGB")

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Average Price / sq ft by City</div>',
        unsafe_allow_html=True,
    )
    summary = (
        df.assign(price_sqft=df["price"] / df["sqft"])
          .groupby("city")["price_sqft"]
          .mean().reset_index()
          .rename(columns={"price_sqft": "avg_price_sqft"})
          .sort_values("avg_price_sqft", ascending=False)
    )
    fig0 = px.bar(
        summary, x="city", y="avg_price_sqft",
        color="avg_price_sqft", color_continuous_scale="Viridis",
        labels={"avg_price_sqft": "Avg Rs/sqft", "city": ""},
        text=summary["avg_price_sqft"].apply(fmt_k),
    )
    fig0.update_layout(
        **LAYOUT_DARK,
        height=350, margin=dict(l=0, r=0, t=10, b=10),
        coloraxis_showscale=False,
    )
    fig0.update_traces(textposition="outside")
    st.plotly_chart(fig0, use_container_width=True)

    st.markdown(
        '<div class="section-title">Annual Appreciation Rate by City</div>',
        unsafe_allow_html=True,
    )
    rate_df = (
        pd.DataFrame([
            {"city": c, "rate": r} for c, r in CITY_APPRECIATION.items()
        ]).sort_values("rate", ascending=True)
    )
    fig_rate = go.Figure(go.Bar(
        y=rate_df["city"], x=rate_df["rate"],
        orientation="h",
        marker_color=["#64b5f6", "#7ef9a0", "#ce93d8", "#ffd54f", "#ef9a9a"],
        text=rate_df["rate"].apply(lambda v: f"{v:.1f}% / yr"),
        textposition="outside",
    ))
    fig_rate.update_layout(
        **LAYOUT_DARK,
        height=300, margin=dict(l=0, r=80, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(tickfont=dict(size=13)),
    )
    st.plotly_chart(fig_rate, use_container_width=True)
