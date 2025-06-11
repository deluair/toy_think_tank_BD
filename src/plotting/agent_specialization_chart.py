import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

DATA_INPUT_DIR = "outputs/data"
FIGURE_OUTPUT_DIR = "outputs/figures"

def generate_agent_specialization_chart():
    """Generates and saves the AI agent specialization chart."""
    os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_INPUT_DIR, "ai_agent_specializations.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please generate data first.")
        return

    df = pd.read_csv(csv_path)
    df = df.rename(columns={'Unnamed: 0': 'specialization'})

    color_map = {
        'Medium': '#5D878F',
        'High': '#FFC185',
        'Very High': '#B4413C'
    }

    df_sorted = df.sort_values('count', ascending=True)

    fig = go.Figure()

    for complexity_level in df_sorted['complexity'].unique():
        subset = df_sorted[df_sorted['complexity'] == complexity_level]
        fig.add_trace(go.Bar(
            x=subset['count'],
            y=subset['specialization'],
            orientation='h',
            name=complexity_level,
            marker_color=color_map.get(complexity_level, '#CCCCCC'), # Default color if complexity not in map
            text=subset['count'],
            textposition='outside',
            cliponaxis=False
        ))

    fig.update_layout(
        title='AI Agents by Specialization',
        xaxis_title='Count',
        yaxis_title='Specialization',
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
        barmode='group',
        height=max(600, len(df_sorted['specialization']) * 40) # Adjust height based on number of specializations
    )

    fig.update_yaxes(tickmode='linear')
    fig.update_xaxes(tickmode='linear')

    output_path = os.path.join(FIGURE_OUTPUT_DIR, "ai_agents_chart.png")
    fig.write_image(output_path)
    print(f"Chart saved successfully to {output_path}!")

if __name__ == '__main__':
    # Create dummy data for direct script execution (optional, for testing)
    # Ensure 'outputs/data' directory exists and 'ai_agent_specializations.csv' is present
    # Example: from src.data_generation.policy_and_agents import generate_policy_and_agents_data
    # generate_policy_and_agents_data() 
    generate_agent_specialization_chart()
