"""
components.py – Reusable, presentation-ready UI components for ScanWise.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Clinical Color Palette
# ---------------------------------------------------------------------------
GRADE_COLORS = {
    "No DR":             "#10b981",  # Emerald
    "Mild DR":           "#34d399",  # Mint
    "Moderate DR":       "#f59e0b",  # Amber
    "Severe DR":         "#f97316",  # Orange
    "Proliferative DR":  "#ef4444",  # Crimson
    "No ME Risk":        "#10b981",
    "Moderate ME Risk":  "#f59e0b",
    "High ME Risk":      "#ef4444",
}

PRIMARY   = "#6366f1"
SECONDARY = "#8b5cf6"
SURFACE   = "#1e1e2e"
CARD_BG   = "#242438"

# ---------------------------------------------------------------------------
# Chart Builders
# ---------------------------------------------------------------------------

def class_distribution_chart(
    counts: Dict[str, int],
    title: str = "Class Distribution",
) -> go.Figure:
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [GRADE_COLORS.get(l, PRIMARY) for l in labels]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=values,
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#f1f5f9")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        margin=dict(t=45, b=10, l=10, r=10),
        height=300,
    )
    return fig


def confidence_gauge(probability: float, label: str) -> go.Figure:
    pct = probability * 100
    color = "#10b981" if pct >= 75 else "#f59e0b" if pct >= 50 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        title={"text": f"Model Confidence", "font": {"color": "#94a3b8", "size": 13}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748b"},
            "bar": {"color": color, "thickness": 0.8},
            "bgcolor": "rgba(255,255,255,0.05)",
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0,  50], "color": "rgba(239,68,68,0.12)"},
                {"range": [50, 75], "color": "rgba(245,158,11,0.12)"},
                {"range": [75, 100], "color": "rgba(16,185,129,0.12)"},
            ],
        },
        number={"suffix": "%", "font": {"color": "#ffffff", "size": 28}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        height=210,
        margin=dict(t=30, b=10, l=20, r=20),
    )
    return fig


def probability_bar_chart(probabilities: Dict[str, float]) -> go.Figure:
    labels = list(probabilities.keys())
    values = [probabilities[l] * 100 for l in labels]
    colors = [GRADE_COLORS.get(l, PRIMARY) for l in labels]

    fig = go.Figure(go.Bar(
        y=labels,
        x=values,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        xaxis=dict(range=[0, 115], title="Posterior Probability (%)", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(autorange="reversed"),
        margin=dict(t=10, b=30, l=10, r=60),
        height=260,
    )
    return fig


def confusion_matrix_chart(
    cm: List[List[int]],
    class_names: List[str],
    title: str = "Normalized Confusion Matrix",
) -> go.Figure:
    z     = np.array(cm, dtype=float)
    z_pct = z / (z.sum(axis=1, keepdims=True) + 1e-8)

    text = [[f"{int(z[i][j])}<br><b>({z_pct[i][j]*100:.0f}%)</b>"
             for j in range(len(class_names))]
            for i in range(len(class_names))]

    fig = go.Figure(go.Heatmap(
        z=z_pct,
        x=[f"Pred: {n}" for n in class_names],
        y=[f"True: {n}" for n in class_names],
        text=text,
        texttemplate="%{text}",
        colorscale="Blues",
        hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{text}<extra></extra>",
        showscale=False,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#f1f5f9")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        margin=dict(t=50, b=10, l=10, r=10),
        height=380,
        xaxis=dict(side="bottom"),
    )
    return fig


def loss_curve_chart(history: Dict) -> go.Figure:
    epochs = list(range(1, len(history["loss"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=history["loss"], mode="lines+markers",
        name="Train Loss", line=dict(color=PRIMARY, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=history["val_loss"], mode="lines+markers",
        name="Val Loss", line=dict(color="#f97316", width=2.5, dash="dash"),
    ))
    fig.update_layout(
        title=dict(text="Loss Convergence (Cross-Entropy)", font=dict(size=15, color="#f1f5f9")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
        xaxis=dict(title="Epoch", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Loss",  gridcolor="rgba(255,255,255,0.06)"),
        margin=dict(t=45, b=30, l=40, r=10),
        height=300,
    )
    return fig


def accuracy_curve_chart(history: Dict) -> go.Figure:
    epochs = list(range(1, len(history["acc"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=[v*100 for v in history["acc"]], mode="lines+markers",
        name="Train Accuracy", line=dict(color="#10b981", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=[v*100 for v in history["val_acc"]], mode="lines+markers",
        name="Val Accuracy", line=dict(color=SECONDARY, width=2.5, dash="dash"),
    ))
    fig.update_layout(
        title=dict(text="Accuracy Convergence (%)", font=dict(size=15, color="#f1f5f9")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
        xaxis=dict(title="Epoch", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Accuracy (%)", gridcolor="rgba(255,255,255,0.06)"),
        margin=dict(t=45, b=30, l=40, r=10),
        height=300,
    )
    return fig


# ---------------------------------------------------------------------------
# Clinical Stat & Presentation Cards
# ---------------------------------------------------------------------------

def metric_card(label: str, value: str, subtext: str = "", color: str = PRIMARY) -> None:
    st.markdown(f"""
    <div style="
        background:{CARD_BG};
        border-top: 3px solid {color};
        border-radius:10px;
        padding:16px 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    ">
        <p style="color:#94a3b8;font-size:12px;font-weight:500;margin:0 0 4px;text-transform:uppercase;letter-spacing:0.5px">{label}</p>
        <p style="color:#ffffff;font-size:26px;font-weight:700;margin:0">{value}</p>
        {f'<p style="color:{color};font-size:12px;margin:4px 0 0;font-weight:600">{subtext}</p>' if subtext else ""}
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"""
    <div style="margin: 20px 0 16px; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 10px;">
        <h2 style="color:#f8fafc;margin:0;font-size:22px;font-weight:700">{title}</h2>
        {f'<p style="color:#94a3b8;margin:4px 0 0;font-size:14px">{subtitle}</p>' if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)


def clinical_action_card(pred_label: str) -> None:
    """Provides actionable clinical recommendations based on ICDR guidelines."""
    actions = {
        "No DR": {
            "urgency": "Routine Screening",
            "timeline": "Annual Follow-up (12 months)",
            "action": "Maintain optimal glycaemic and blood pressure control. Repeat dilated fundus photography annually.",
            "color": "#10b981",
            "referable": False,
        },
        "Mild DR": {
            "urgency": "Clinical Monitoring",
            "timeline": "Review in 6–12 months",
            "action": "Small microaneurysms detected. Optimize diabetes management with primary care physician. Repeat retinal screening in 6-12 months.",
            "color": "#34d399",
            "referable": False,
        },
        "Moderate DR": {
            "urgency": "Specialist Referral",
            "timeline": "Referral within 3–6 months",
            "action": "Multiple microaneurysms, blot haemorrhages, or hard exudates present. Refer to an ophthalmologist for comprehensive examination.",
            "color": "#f59e0b",
            "referable": True,
        },
        "Severe DR": {
            "urgency": "Urgent Referral",
            "timeline": "Referral within 2–4 weeks",
            "action": "Severe NPDR (4-2-1 rule criteria). High risk of rapid progression to vision-threatening proliferative retinopathy.",
            "color": "#f97316",
            "referable": True,
        },
        "Proliferative DR": {
            "urgency": "Emergent Referral",
            "timeline": "Immediate Referral (< 1 week)",
            "action": "Neovascularisation or pre-retinal haemorrhage suspected. Urgent vitreoretinal specialist intervention (panretinal photocoagulation / anti-VEGF).",
            "color": "#ef4444",
            "referable": True,
        },
    }
    info = actions.get(pred_label, actions["No DR"])
    ref_tag = "🔴 Referable DR (Specialist Care Needed)" if info["referable"] else "🟢 Non-Referable DR (Primary Screening)"

    st.markdown(f"""
    <div style="
        background: rgba(30,30,46,0.85);
        border: 1px solid rgba(255,255,255,0.08);
        border-left: 5px solid {info['color']};
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 12px;
    ">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
            <span style="color:{info['color']};font-weight:700;font-size:14px;text-transform:uppercase;letter-spacing:0.5px">
                ⚡ Clinical Triage: {info['urgency']}
            </span>
            <span style="background:rgba(255,255,255,0.08);padding:3px 10px;border-radius:12px;font-size:11px;color:#e2e8f0;font-weight:600">
                {ref_tag}
            </span>
        </div>
        <p style="color:#ffffff;font-size:15px;font-weight:600;margin:0 0 6px">Recommended Timeline: {info['timeline']}</p>
        <p style="color:#cbd5e1;font-size:13px;line-height:1.5;margin:0">{info['action']}</p>
    </div>
    """, unsafe_allow_html=True)


def disclaimer_footer() -> None:
    st.markdown("""
    <div style="
        border-top: 1px solid rgba(255,255,255,0.06);
        padding-top: 16px;
        margin-top: 40px;
        text-align: center;
        color: #64748b;
        font-size: 11px;
    ">
        ⚕️ <b>ScanWise Clinical Research Prototype</b> | Built for academic evaluation & demonstration on the IDRiD Retinal Dataset.
        This system is designed as an AI decision-support prototype and does not replace diagnostic assessment by an ophthalmologist.
    </div>
    """, unsafe_allow_html=True)
