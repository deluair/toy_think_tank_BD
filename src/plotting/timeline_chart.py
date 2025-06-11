import pandas as pd
import plotly.graph_objects as go
import os

DATA_INPUT_DIR = "outputs/data"
FIGURE_OUTPUT_DIR = "outputs/figures"

def generate_timeline_chart():
    """Generates and saves the implementation budget timeline chart."""
    os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_INPUT_DIR, "implementation_timeline.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please generate data first.")
        return

    df = pd.read_csv(csv_path)
    df = df.rename(columns={'Unnamed: 0': 'Phase'})

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Phase'],
        y=df['budget_usd'],
        mode='lines+markers',
        name='Budget',
        line=dict(color='#1FB8CD', width=3),
        marker=dict(size=8, color='#1FB8CD'),
        hovertemplate='<b>%{x}</b><br>Budget: $%{y:,.0f}<extra></extra>',
        cliponaxis=False
    ))

    fig.update_layout(
        title='Implementation Budget Timeline',
        xaxis_title='Phase',
        yaxis_title='Budget (USD)',
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
        hovermode='x unified'
    )

    fig.update_yaxes(tickformat='$,.0f')

    output_path = os.path.join(FIGURE_OUTPUT_DIR, "implementation_timeline_chart.png")
    fig.write_image(output_path)
    print(f"Chart saved successfully to {output_path}!")

if __name__ == '__main__':
    # Create dummy data for direct script execution (optional, for testing)
    # Ensure 'outputs/data' directory exists and 'implementation_timeline.csv' is present
    # Example: from src.data_generation.policy_and_agents import generate_policy_and_agents_data
    # generate_policy_and_agents_data()
    generate_timeline_chart()
