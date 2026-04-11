"""
LungScreen — Public Equity Dashboard
=====================================
Aggregate-only Dash app for sharing lung cancer screening adherence
findings with clinical stakeholders, grant reviewers, and the public.

All data is pre-aggregated with cell sizes >= 20 per All of Us policy.
No individual-level data is included.

Deploy: Render (render.com) with gunicorn.
"""

import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, callback, Output, Input

# ---------------------------------------------------------------------------
# Load aggregate data
# ---------------------------------------------------------------------------
with open("data.json") as f:
    bundle = json.load(f)

fairness = pd.DataFrame(bundle['fairness_metrics'])
gaps = pd.DataFrame(bundle['fairness_gaps'])
models = pd.DataFrame(bundle['model_comparison'])
ins_pcp = pd.DataFrame(bundle['insurance_pcp'])
summary = bundle['cohort_summary']

# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------
NAVY    = '#1B4965'
TEAL    = '#62B6CB'
GREEN   = '#2A9D8F'
RED     = '#E63946'
GRAY    = '#6C757D'
LIGHT   = '#F8F9FA'
WHITE   = '#FFFFFF'

# ---------------------------------------------------------------------------
# Insurance display helpers
# ---------------------------------------------------------------------------
INS_ORDER = ['va_other_gov', 'aca_marketplace', 'medicare',
             'private_employer', 'medicaid', 'unknown', 'uninsured_or_selfpay']
INS_LABELS = {
    'va_other_gov': 'VA / other gov',
    'aca_marketplace': 'ACA marketplace',
    'medicare': 'Medicare',
    'private_employer': 'Private (employer)',
    'medicaid': 'Medicaid',
    'unknown': 'Unknown',
    'uninsured_or_selfpay': 'Uninsured / self-pay',
}

# ---------------------------------------------------------------------------
# Build figures
# ---------------------------------------------------------------------------

def make_stat_cards():
    fig = make_subplots(
        rows=1, cols=5,
        specs=[[{"type": "indicator"}] * 5],
        subplot_titles=["USPSTF Eligible", "Screened", "Screening Rate",
                        "PCP Engaged", "Flagged for Outreach"],
    )
    vals = [
        (summary['n_total'], NAVY, ''),
        (summary['n_screened'], GREEN, ''),
        (summary['screening_rate'], RED, '%'),
        (summary['n_pcp'], TEAL, ''),
        (summary['n_outreach'], RED, ''),
    ]
    for i, (v, c, s) in enumerate(vals):
        fig.add_trace(go.Indicator(
            mode="number", value=v,
            number=dict(font=dict(size=34, color=c), suffix=s),
        ), row=1, col=i + 1)
    fig.update_layout(height=170, margin=dict(t=40, b=10, l=10, r=10),
                      paper_bgcolor=WHITE, plot_bgcolor=WHITE)
    return fig


def make_model_table():
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Model', 'Estimator', 'AUC', 'AP', 'Brier', 'n', 'Pos Rate (%)'],
            fill_color=NAVY, font=dict(color=WHITE, size=12), align='left',
        ),
        cells=dict(
            values=[models[c] for c in models.columns],
            fill_color=[LIGHT], align='left', font=dict(size=11),
        ),
    )])
    fig.update_layout(title="Model Comparison (5-fold CV)", height=230,
                      margin=dict(t=40, b=10, l=10, r=10),
                      paper_bgcolor=WHITE)
    return fig


def make_heatmap():
    pivot = ins_pcp.pivot(index='insurance_status', columns='has_pcp', values='rate_pct')
    pivot.columns = ['No PCP visit', 'Has PCP visit']
    n_pivot = ins_pcp.pivot(index='insurance_status', columns='has_pcp', values='n')

    ordered = [i for i in INS_ORDER if i in pivot.index]
    pivot = pivot.reindex(ordered)
    n_pivot = n_pivot.reindex(ordered)

    annotations = []
    for idx in pivot.index:
        for col in pivot.columns:
            v = pivot.loc[idx, col]
            n = n_pivot.loc[idx, col] if col in n_pivot.columns else 0
            if pd.notna(v) and pd.notna(n):
                annotations.append(f"{v:.1f}%\nn={int(n):,}")
            else:
                annotations.append("")

    z = pivot.values
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=list(pivot.columns),
        y=[INS_LABELS.get(i, i) for i in pivot.index],
        text=list(map(list, zip(*[iter(annotations)] * len(pivot.columns)))),
        texttemplate="%{text}",
        colorscale=[[0, WHITE], [0.33, '#FFD166'], [0.66, '#EF476F'], [1.0, NAVY]],
        zmin=0, zmax=30,
        colorbar=dict(title="Screening %"),
    ))
    fig.update_layout(
        title="Screening Rate: Insurance × PCP Engagement (strict USPSTF cohort)",
        height=380, margin=dict(t=50, b=20, l=160, r=20),
        paper_bgcolor=WHITE, plot_bgcolor=WHITE,
    )
    return fig


def make_disparity_chart(variable):
    subset = fairness[fairness['variable'] == variable].copy()
    if subset.empty:
        return go.Figure()
    subset['rate_pct'] = (subset['prevalence'] * 100).round(1)
    subset = subset.sort_values('rate_pct', ascending=True)
    overall = summary['screening_rate']

    colors = [RED if r < overall else GREEN for r in subset['rate_pct']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=subset['group'], x=subset['rate_pct'],
        orientation='h', marker_color=colors,
        text=[f"{r:.1f}% (n={n:,})" for r, n in
              zip(subset['rate_pct'], subset['n'])],
        textposition='outside',
    ))
    fig.add_vline(x=overall, line_dash="dash", line_color=GRAY,
                  annotation_text=f"Overall: {overall:.1f}%")
    fig.update_layout(
        title=f'Screening Rate by {variable.replace("_", " ").title()}',
        xaxis_title='Screening Rate (%)',
        height=max(300, len(subset) * 42 + 100),
        showlegend=False,
        margin=dict(l=180, r=80, t=50, b=40),
        paper_bgcolor=WHITE, plot_bgcolor=WHITE,
    )
    return fig


def make_auc_chart(variable):
    subset = fairness[(fairness['variable'] == variable) &
                       fairness['auc'].notna()].copy()
    if subset.empty:
        return go.Figure()

    error_y = None
    if 'auc_ci_hi' in subset.columns and 'auc_ci_lo' in subset.columns:
        error_y = dict(
            type='data', symmetric=False,
            array=(subset['auc_ci_hi'] - subset['auc']).values,
            arrayminus=(subset['auc'] - subset['auc_ci_lo']).values,
        )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=subset['group'], y=subset['auc'],
        marker_color=TEAL,
        text=[f"{a:.3f}" for a in subset['auc']],
        textposition='outside',
        error_y=error_y,
    ))
    fig.update_layout(
        title=f'Model AUC by {variable.replace("_", " ").title()}',
        yaxis_title='AUC-ROC', yaxis_range=[0.5, 1.05],
        height=350,
        margin=dict(t=50, b=40, l=60, r=20),
        paper_bgcolor=WHITE, plot_bgcolor=WHITE,
    )
    return fig


def make_gaps_table():
    if gaps.empty:
        return go.Figure()

    cell_colors = []
    for col in gaps.columns:
        col_colors = []
        for v in gaps[col]:
            if v == 'PASS':
                col_colors.append('#E8F5E9')
            elif v == 'FLAG':
                col_colors.append('#FFEBEE')
            else:
                col_colors.append(LIGHT)
        cell_colors.append(col_colors)

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[c.replace('_', ' ').title() for c in gaps.columns],
            fill_color=NAVY, font=dict(color=WHITE, size=11), align='left',
        ),
        cells=dict(
            values=[gaps[c] for c in gaps.columns],
            fill_color=cell_colors,
            align='left', font=dict(size=11),
            format=[None, None, '.3f', '.3f', '.3f', '.3f', '.3f', None],
        ),
    )])
    fig.update_layout(title="Fairness Gaps Across Stratification Axes",
                      height=280, margin=dict(t=40, b=10, l=10, r=10),
                      paper_bgcolor=WHITE)
    return fig


def make_subgroup_table(variable):
    subset = fairness[fairness['variable'] == variable].copy()
    if subset.empty:
        return go.Figure()

    cols = ['group', 'n', 'prevalence', 'auc', 'auc_ci_lo', 'auc_ci_hi',
            'tpr', 'fpr', 'ppv', 'cal_slope']
    available = [c for c in cols if c in subset.columns]
    subset = subset[available]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[c.replace('_', ' ').title() for c in available],
            fill_color=NAVY, font=dict(color=WHITE, size=11), align='left',
        ),
        cells=dict(
            values=[subset[c] for c in available],
            fill_color=[LIGHT], align='left', font=dict(size=11),
        ),
    )])
    fig.update_layout(height=max(200, 40 * len(subset) + 80),
                      margin=dict(t=10, b=10, l=10, r=10),
                      paper_bgcolor=WHITE)
    return fig


# ---------------------------------------------------------------------------
# Dropdown options
# ---------------------------------------------------------------------------
STRAT_OPTIONS = []
for v in ['race_ethnicity', 'insurance_status', 'sex', 'education_level',
          'income_level', 'state']:
    if v in fairness['variable'].values:
        STRAT_OPTIONS.append({'label': v.replace('_', ' ').title(), 'value': v})


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------
app = Dash(__name__)
server = app.server  # for gunicorn

app.title = "LungScreen — Equity Dashboard"

app.layout = html.Div(style={'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                              'maxWidth': '1100px', 'margin': '0 auto',
                              'padding': '20px', 'backgroundColor': WHITE}, children=[

    # Header
    html.H1("LungScreen — Equity Dashboard",
            style={'color': NAVY, 'borderBottom': f'3px solid {TEAL}',
                   'paddingBottom': '10px'}),
    html.P([
        "Fairness-aware lung cancer screening adherence analysis using the NIH ",
        html.Em("All of Us"), " Research Program (CDR v8 R2024Q3R8). ",
        "Strict USPSTF cohort: adults aged 50–80 with ≥20 pack-years smoking "
        "history who currently smoke or quit within 15 years.",
    ], style={'color': GRAY, 'fontSize': '14px', 'marginBottom': '20px'}),

    html.P([
        "All statistics are aggregate with minimum cell sizes ≥ 20 per ",
        html.Em("All of Us"), " data use policy. No individual-level data is displayed.",
    ], style={'color': GRAY, 'fontSize': '12px', 'fontStyle': 'italic',
              'marginBottom': '30px'}),

    # Tabs
    dcc.Tabs(id='tabs', value='overview', children=[

        # ==== Tab 1: Overview ====
        dcc.Tab(label='Population Overview', value='overview', children=[
            html.Div(style={'padding': '20px 0'}, children=[
                dcc.Graph(figure=make_stat_cards(), config={'displayModeBar': False}),
                dcc.Graph(figure=make_model_table(), config={'displayModeBar': False}),
                dcc.Graph(figure=make_heatmap()),
            ]),
        ]),

        # ==== Tab 2: Disparities ====
        dcc.Tab(label='Disparity Explorer', value='disparities', children=[
            html.Div(style={'padding': '20px 0'}, children=[
                html.Label("Stratify by:", style={'fontWeight': 'bold',
                                                   'marginRight': '10px'}),
                dcc.Dropdown(
                    id='strat-dropdown',
                    options=STRAT_OPTIONS,
                    value='race_ethnicity',
                    style={'width': '300px', 'marginBottom': '20px'},
                ),
                dcc.Graph(id='disparity-chart'),
                dcc.Graph(id='auc-chart'),
            ]),
        ]),

        # ==== Tab 3: Fairness Report ====
        dcc.Tab(label='Fairness Report', value='fairness', children=[
            html.Div(style={'padding': '20px 0'}, children=[
                dcc.Graph(figure=make_gaps_table(), config={'displayModeBar': False}),
                html.H3("Detailed Metrics by Subgroup",
                         style={'color': NAVY, 'marginTop': '30px'}),
                html.Label("Select axis:", style={'fontWeight': 'bold',
                                                   'marginRight': '10px'}),
                dcc.Dropdown(
                    id='fairness-dropdown',
                    options=STRAT_OPTIONS,
                    value='race_ethnicity',
                    style={'width': '300px', 'marginBottom': '20px'},
                ),
                dcc.Graph(id='subgroup-table'),
            ]),
        ]),
    ]),

    # Footer
    html.Div(style={'marginTop': '50px', 'paddingTop': '20px',
                     'borderTop': '1px solid #ddd', 'color': '#999',
                     'fontSize': '11px'}, children=[
        html.P([
            "LungScreen: Fairness-Aware Lung Cancer Screening Adherence Pipeline. ",
            "Project lead: Dan Zimmerman (FAU Center for Connected Autonomy & AI). ",
            "Clinical lead: Dr. Mark Block, MD (Memorial Healthcare). ",
        ]),
        html.P([
            "Data source: NIH All of Us Research Program, CDR v8 R2024Q3R8. ",
            "All aggregate statistics comply with the All of Us Data Use Agreement "
            "(minimum cell size ≥ 20). No individual-level data is displayed or stored.",
        ]),
    ]),
])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@callback(
    Output('disparity-chart', 'figure'),
    Output('auc-chart', 'figure'),
    Input('strat-dropdown', 'value'),
)
def update_disparity(variable):
    return make_disparity_chart(variable), make_auc_chart(variable)


@callback(
    Output('subgroup-table', 'figure'),
    Input('fairness-dropdown', 'value'),
)
def update_subgroup(variable):
    return make_subgroup_table(variable)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
