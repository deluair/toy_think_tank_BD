import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the data
df = pd.read_csv("ai_agent_specializations.csv")

# Rename columns for easier handling
df = df.rename(columns={'Unnamed: 0': 'specialization'})

# Create color mapping based on complexity levels
color_map = {
    'Medium': '#5D878F',  # Cyan (blue-ish from brand colors)
    'High': '#FFC185',    # Light orange
    'Very High': '#B4413C'  # Moderate red
}

# Sort by count for better visualization
df_sorted = df.sort_values('count', ascending=True)

# Create horizontal bar chart
fig = go.Figure()

# Add bars for each complexity level
for complexity in df_sorted['complexity'].unique():
    subset = df_sorted[df_sorted['complexity'] == complexity]
    
    fig.add_trace(go.Bar(
        x=subset['count'],
        y=subset['specialization'],
        orientation='h',
        name=complexity,
        marker_color=color_map[complexity],
        text=subset['count'],
        textposition='outside',
        cliponaxis=False
    ))

# Update layout
fig.update_layout(
    title='AI Agents by Specialization',
    xaxis_title='Count',
    yaxis_title='Specialization',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
    barmode='group'
)

# Update y-axis to show full specialization names but truncated if needed
fig.update_yaxes(tickmode='linear')
fig.update_xaxes(tickmode='linear')

# Save the chart
fig.write_image("ai_agents_chart.png")
print("Chart saved successfully!")