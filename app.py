"""
LungScreen — Public Equity Dashboard
=====================================
Mobile-first, aggregate-only Dash app for sharing lung cancer screening
adherence findings with clinical stakeholders.

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
# Colors
# ---------------------------------------------------------------------------
NAVY    = '#1B4965'
TEAL    = '#62B6CB'
GREEN   = '#2A9D8F'
RED     = '#E63946'
GRAY    = '#6C757D'
LIGHT   = '#F8F9FA'
WHITE   = '#FFFFFF'

# ---------------------------------------------------------------------------
# Insurance helpers
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
# Responsive chart config — disable modebar on mobile, allow scroll zoom
# ---------------------------------------------------------------------------
CHART_CONFIG = {
    'displayModeBar': False,
    'responsive': True,
    'scrollZoom': False,
}

RESPONSIVE_MARGIN = dict(l=10, r=10, t=45, b=30)


# ---------------------------------------------------------------------------
# Stat card component (mobile-friendly)
# ---------------------------------------------------------------------------
def stat_card(label, value, color=NAVY, suffix=''):
    return html.Div(style={
        'backgroundColor': WHITE,
        'border': f'1px solid #e0e0e0',
        'borderLeft': f'4px solid {color}',
        'borderRadius': '6px',
        'padding': '14px 16px',
        'textAlign': 'center',
        'flex': '1 1 140px',
        'minWidth': '120px',
    }, children=[
        html.Div(f"{value}{suffix}",
                 style={'fontSize': '26px', 'fontWeight': '700', 'color': color}),
        html.Div(label,
                 style={'fontSize': '11px', 'color': GRAY, 'textTransform': 'uppercase',
                        'letterSpacing': '0.5px', 'marginTop': '4px'}),
    ])


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def html_table(df, title=None, status_col=None):
    """Build a responsive HTML table from a DataFrame.
    Wrapped in a scrollable div so it reflows on narrow screens.
    If status_col is given, PASS cells are green-tinted and FLAG cells red-tinted."""
    header = html.Thead(html.Tr([
        html.Th(c, style={
            'backgroundColor': NAVY, 'color': WHITE,
            'padding': '8px 10px', 'textAlign': 'left',
            'fontSize': '11px', 'fontWeight': '600',
            'position': 'sticky', 'top': '0',
        }) for c in df.columns
    ]))

    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, float):
                txt = f"{v:.3f}"
            else:
                txt = str(v)
            cell_style = {
                'padding': '7px 10px',
                'borderBottom': '1px solid #eaeaea',
                'fontSize': '12px',
            }
            if status_col and c == status_col:
                if v == 'PASS':
                    cell_style['backgroundColor'] = '#E8F5E9'
                    cell_style['color'] = GREEN
                    cell_style['fontWeight'] = '700'
                elif v == 'FLAG':
                    cell_style['backgroundColor'] = '#FFEBEE'
                    cell_style['color'] = RED
                    cell_style['fontWeight'] = '700'
            cells.append(html.Td(txt, style=cell_style))
        rows.append(html.Tr(cells))

    table = html.Table([header, html.Tbody(rows)], style={
        'width': '100%',
        'borderCollapse': 'collapse',
        'backgroundColor': WHITE,
    })

    wrapper = html.Div([
        html.H4(title, style={
            'color': NAVY, 'fontSize': '14px', 'margin': '20px 0 8px 0',
        }) if title else None,
        html.Div(table, style={
            'overflowX': 'auto',
            'WebkitOverflowScrolling': 'touch',
            'border': '1px solid #e0e0e0',
            'borderRadius': '4px',
        }),
    ])
    return wrapper


def make_model_table_html():
    display_models = models.copy()
    display_models.columns = ['Model', 'Est.', 'AUC', 'AP', 'Brier', 'n', 'Pos %']
    return html_table(display_models, title="Model Comparison (5-fold CV)")


def make_heatmap():
    pivot = ins_pcp.pivot(index='insurance_status', columns='has_pcp', values='rate_pct')
    n_pivot = ins_pcp.pivot(index='insurance_status', columns='has_pcp', values='n')

    ordered = [i for i in INS_ORDER if i in pivot.index]
    pivot = pivot.reindex(ordered)
    n_pivot = n_pivot.reindex(ordered)

    # Build annotations using the original integer columns (0, 1), THEN rename
    # the display columns. Previous bug: renaming pivot.columns before the
    # loop made the lookup in n_pivot (still [0, 1]) always fail → n=0 everywhere.
    annotations = []
    for idx in pivot.index:
        for col in pivot.columns:  # integers 0, 1
            v = pivot.loc[idx, col]
            n = n_pivot.loc[idx, col] if col in n_pivot.columns else 0
            if pd.notna(v) and pd.notna(n):
                annotations.append(f"{v:.1f}%<br>n={int(n):,}")
            else:
                annotations.append("")

    # Rename columns now, for display only
    pivot.columns = ['No PCP', 'Has PCP']

    z = pivot.values
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=list(pivot.columns),
        y=[INS_LABELS.get(i, i) for i in pivot.index],
        text=list(map(list, zip(*[iter(annotations)] * len(pivot.columns)))),
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorscale=[[0, WHITE], [0.33, '#FFD166'], [0.66, '#EF476F'], [1.0, NAVY]],
        zmin=0, zmax=30,
        colorbar=dict(
            title=dict(text="Rate %", font=dict(size=10)),
            tickfont=dict(size=9), len=0.8,
        ),
    ))
    fig.update_layout(
        title="Screening: Insurance × PCP Engagement",
        title_font_size=14,
        height=340,
        margin=dict(l=130, r=10, t=45, b=20),
        paper_bgcolor=WHITE, plot_bgcolor=WHITE, autosize=True,
        yaxis=dict(tickfont=dict(size=10)),
        xaxis=dict(tickfont=dict(size=11)),
    )
    return fig


# Ordinal orderings (low-to-high / "unknown" first so it sits at the bottom
# of the horizontal bar chart). For variables not in this dict, the chart
# sorts by screening rate — that's the informative ordering for nominal axes
# like race and insurance.
ORDINAL_ORDERS = {
    'income_level': ['unknown', '<10k', '10k-25k', '25k-35k', '35k-50k',
                     '50k-75k', '75k-100k', '100k-150k', '150k-200k', '>200k'],
    'education_level': ['unknown', 'Less than HS', 'HS',
                        'Vocational training', 'College or grad'],
}


def make_disparity_chart(variable):
    subset = fairness[fairness['variable'] == variable].copy()
    if subset.empty:
        return go.Figure()
    subset['rate_pct'] = (subset['prevalence'] * 100).round(1)

    if variable in ORDINAL_ORDERS:
        order_map = {g: i for i, g in enumerate(ORDINAL_ORDERS[variable])}
        subset['__order'] = subset['group'].map(order_map).fillna(999)
        subset = subset.sort_values('__order').drop(columns='__order')
    else:
        subset = subset.sort_values('rate_pct', ascending=True)

    overall = summary['screening_rate']
    colors = [RED if r < overall else GREEN for r in subset['rate_pct']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=subset['group'], x=subset['rate_pct'],
        orientation='h', marker_color=colors,
        text=[f"{r:.1f}% (n={int(n):,})" for r, n in
              zip(subset['rate_pct'], subset['n'])],
        textposition='outside',
        textfont=dict(size=10),
    ))
    fig.add_vline(x=overall, line_dash="dash", line_color=GRAY,
                  annotation_text=f"Overall: {overall:.1f}%",
                  annotation_font_size=10)
    fig.update_layout(
        title=f'Screening Rate: {variable.replace("_", " ").title()}',
        title_font_size=14,
        xaxis_title='Rate (%)',
        height=max(260, len(subset) * 38 + 80),
        showlegend=False,
        margin=dict(l=150, r=100, t=45, b=35),
        paper_bgcolor=WHITE, plot_bgcolor=WHITE, autosize=True,
        yaxis=dict(tickfont=dict(size=10)),
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
        textfont=dict(size=10),
        error_y=error_y,
    ))
    fig.update_layout(
        title=f'Model AUC: {variable.replace("_", " ").title()}',
        title_font_size=14,
        yaxis_title='AUC-ROC', yaxis_range=[0.5, 1.05],
        height=300,
        margin=dict(t=45, b=40, l=50, r=10),
        paper_bgcolor=WHITE, plot_bgcolor=WHITE, autosize=True,
        xaxis=dict(tickfont=dict(size=9), tickangle=-30),
    )
    return fig


def make_gaps_table_html():
    if gaps.empty:
        return html.Div()
    display = gaps[['variable', 'n_subgroups', 'auc_gap', 'tpr_gap',
                     'eq_odds_gap', 'screening_rate_gap', 'auc_pass_lt_0_1']].copy()
    display.columns = ['Axis', 'Groups', 'AUC Gap', 'TPR Gap',
                        'EO Gap', 'Rate Gap', 'Status']
    return html_table(display, title="Fairness Gaps Across Stratification Axes",
                       status_col='Status')


def make_subgroup_table_html(variable):
    subset = fairness[fairness['variable'] == variable].copy()
    if subset.empty:
        return html.Div("No data for this subgroup.", style={'padding': '20px'})
    cols = ['group', 'n', 'prevalence', 'auc', 'tpr', 'fpr', 'cal_slope']
    available = [c for c in cols if c in subset.columns]
    display = subset[available].copy()
    display.columns = [c.replace('_', ' ').title() for c in available]
    return html_table(display)


# ---------------------------------------------------------------------------
# Dropdown options
# ---------------------------------------------------------------------------
STRAT_OPTIONS = []
for v in ['race_ethnicity', 'insurance_status', 'sex', 'education_level',
          'income_level', 'state']:
    if v in fairness['variable'].values:
        STRAT_OPTIONS.append({'label': v.replace('_', ' ').title(), 'value': v})


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Dash(__name__)
server = app.server

app.title = "LungScreen — Equity Dashboard"

# Inject viewport meta tag for mobile rendering
app.index_string = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        * { box-sizing: border-box; -webkit-text-size-adjust: 100%; }
        body { margin: 0; padding: 0; background: ''' + LIGHT + '''; overflow-x: hidden; }
        .Select-control { min-height: 44px !important; }
        .Select-input > input { font-size: 16px !important; }
        /* Force plotly charts to respect container width */
        .js-plotly-plot, .plotly, .plot-container { width: 100% !important; }
        .js-plotly-plot .main-svg { width: 100% !important; }
        /* Hide plotly modebar on all screens — clutter on mobile */
        .modebar-container { display: none !important; }
        /* Horizontal scroll for wide tables */
        .dash-table-container { overflow-x: auto !important; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>'''

# ---------------------------------------------------------------------------
# Layout — single column, stacked, mobile-first
# ---------------------------------------------------------------------------
app.layout = html.Div(style={
    'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    'maxWidth': '800px',
    'margin': '0 auto',
    'padding': '16px',
    'backgroundColor': LIGHT,
}, children=[

    # Header
    html.H1("LungScreen", style={
        'color': NAVY, 'fontSize': '28px', 'marginBottom': '4px',
        'borderBottom': f'3px solid {TEAL}', 'paddingBottom': '8px',
    }),
    html.P("Fairness-Aware Lung Cancer Screening Adherence",
           style={'color': GRAY, 'fontSize': '14px', 'marginTop': '0',
                  'marginBottom': '4px'}),
    html.P([
        "NIH ", html.Em("All of Us"), " CDR v8 | Strict USPSTF cohort | "
        "All statistics n ≥ 20",
    ], style={'color': GRAY, 'fontSize': '11px', 'marginBottom': '20px'}),

    # Stat cards — CSS grid, 2 columns on mobile, 5 on desktop
    html.Div(style={
        'display': 'grid',
        'gridTemplateColumns': 'repeat(auto-fit, minmax(140px, 1fr))',
        'gap': '8px',
        'marginBottom': '20px',
    }, children=[
        stat_card("Eligible", f"{summary['n_total']:,}", NAVY),
        stat_card("Screened", f"{summary['n_screened']:,}", GREEN),
        stat_card("Rate", f"{summary['screening_rate']}", RED, suffix='%'),
        stat_card("PCP Engaged", f"{summary['n_pcp']:,}", TEAL),
        stat_card("Outreach", f"{summary['n_outreach']:,}", RED),
    ]),

    # Tabs
    dcc.Tabs(id='tabs', value='overview', style={'fontSize': '13px'},
             children=[

        # ==== Tab 1: Overview ====
        dcc.Tab(label='Overview', value='overview',
                style={'padding': '8px 12px'},
                selected_style={'padding': '8px 12px', 'borderTop': f'3px solid {NAVY}'},
                children=[
            html.Div(style={'padding': '12px 0'}, children=[
                make_model_table_html(),
                dcc.Graph(figure=make_heatmap(), config=CHART_CONFIG,
                          style={'width': '100%', 'marginTop': '20px'}),

                # Key finding callout
                html.Div(style={
                    'backgroundColor': '#FFF7E6',
                    'borderLeft': f'4px solid #F4A261',
                    'padding': '12px 14px',
                    'margin': '12px 0',
                    'fontSize': '13px',
                    'lineHeight': '1.5',
                }, children=[
                    html.Strong("Key finding: "),
                    f"VA patients screen at 29.5% vs Medicaid at 11.7% among "
                    f"equally-eligible, PCP-engaged patients — a 2.5× gap driven "
                    f"by system-level differences, not patient characteristics. "
                    f"Without any PCP visit, screening is ~1% regardless of insurance.",
                ]),
            ]),
        ]),

        # ==== Tab 2: Disparities ====
        dcc.Tab(label='Disparities', value='disparities',
                style={'padding': '8px 12px'},
                selected_style={'padding': '8px 12px', 'borderTop': f'3px solid {NAVY}'},
                children=[
            html.Div(style={'padding': '12px 0'}, children=[
                html.Label("Stratify by:", style={
                    'fontWeight': 'bold', 'fontSize': '13px',
                    'marginBottom': '6px', 'display': 'block',
                }),
                dcc.Dropdown(
                    id='strat-dropdown',
                    options=STRAT_OPTIONS,
                    value='race_ethnicity',
                    style={'marginBottom': '16px', 'fontSize': '14px'},
                    clearable=False,
                ),
                dcc.Graph(id='disparity-chart', config=CHART_CONFIG,
                          style={'width': '100%'}),
                dcc.Graph(id='auc-chart', config=CHART_CONFIG,
                          style={'width': '100%'}),
            ]),
        ]),

        # ==== Tab 3: Fairness ====
        dcc.Tab(label='Fairness', value='fairness',
                style={'padding': '8px 12px'},
                selected_style={'padding': '8px 12px', 'borderTop': f'3px solid {NAVY}'},
                children=[
            html.Div(style={'padding': '12px 0'}, children=[
                make_gaps_table_html(),

                html.H3("Subgroup Metrics", style={
                    'color': NAVY, 'fontSize': '16px', 'marginTop': '24px',
                }),
                html.Label("Select axis:", style={
                    'fontWeight': 'bold', 'fontSize': '13px',
                    'marginBottom': '6px', 'display': 'block',
                }),
                dcc.Dropdown(
                    id='fairness-dropdown',
                    options=STRAT_OPTIONS,
                    value='race_ethnicity',
                    style={'marginBottom': '16px', 'fontSize': '14px'},
                    clearable=False,
                ),
                html.Div(id='subgroup-table-container'),
            ]),
        ]),
    ]),

    # Footer
    html.Div(style={
        'marginTop': '30px', 'paddingTop': '14px',
        'borderTop': '1px solid #ddd', 'fontSize': '10px',
        'color': '#999', 'lineHeight': '1.5',
    }, children=[
        html.P([
            "LungScreen — FAU Center for Connected Autonomy & AI. ",
            "Clinical lead: Dr. Mark Block, MD (Memorial Healthcare). ",
        ]),
        html.P([
            "Data: NIH All of Us CDR v8 R2024Q3R8. ",
            "All statistics ≥ 20 participants per All of Us data use policy. ",
            "No individual-level data displayed.",
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
    Output('subgroup-table-container', 'children'),
    Input('fairness-dropdown', 'value'),
)
def update_subgroup(variable):
    return make_subgroup_table_html(variable)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
