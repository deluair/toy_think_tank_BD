import pandas as pd
import plotly.graph_objects as go

# Load the data
df = pd.read_csv("implementation_timeline.csv")

# Create the chart with budget data
fig = go.Figure()

# Add budget line (using brand color #1FB8CD)
fig.add_trace(go.Scatter(
    x=df['Unnamed: 0'],
    y=df['budget_usd'],
    mode='lines+markers',
    name='Budget',
    line=dict(color='#1FB8CD', width=3),
    marker=dict(size=8, color='#1FB8CD'),
    hovertemplate='<b>%{x}</b><br>Budget: $%{y:,.0f}<extra></extra>',
    cliponaxis=False
))

# Update layout
fig.update_layout(
    title='Implementation Budget Timeline',
    xaxis_title='Phase',
    yaxis_title='Budget (USD)',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
    hovermode='x unified'
)

# Format y-axis to show currency with abbreviated values
fig.update_yaxes(
    tickformat='$,.0f'
)

# Update x-axis
fig.update_xaxes()

# Save the chart
fig.write_image("implementation_timeline_chart.png")
print("Chart saved successfully!")