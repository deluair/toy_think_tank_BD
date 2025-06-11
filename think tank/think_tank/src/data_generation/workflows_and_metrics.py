import pandas as pd
import os
import yaml
from typing import Dict, Any

# Define constants for paths, assuming script is run from project root
CONFIG_DIR = "config"
DATA_OUTPUT_DIR = "outputs/data"

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Loads a YAML file and returns its content."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def generate_workflows_and_metrics_data():
    """Generates and saves data for workflows and metrics from YAML configs."""
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

    configs_to_process = {
        'agent_workflows': 'agent_workflows.yaml',
        'stakeholder_engagement': 'stakeholder_engagement.yaml',
        'innovation_pipeline': 'innovation_pipeline.yaml',
        'performance_metrics': 'performance_metrics.yaml',
        'revenue_diversification': 'revenue_diversification.yaml'
    }

    for df_name, file_name in configs_to_process.items():
        config_data = load_yaml_config(os.path.join(CONFIG_DIR, file_name))
        df = pd.DataFrame.from_dict(config_data, orient='index')
        csv_path = os.path.join(DATA_OUTPUT_DIR, f"{df_name}.csv")
        df.to_csv(csv_path)
        print(f"Saved {os.path.basename(csv_path)} to {DATA_OUTPUT_DIR}")

    print("\nAll operational frameworks saved to CSV files!")

if __name__ == '__main__':
    generate_workflows_and_metrics_data()
