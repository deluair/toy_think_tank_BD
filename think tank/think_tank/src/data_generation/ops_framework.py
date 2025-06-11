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

def generate_ops_framework_data():
    """Generates and saves data for the operational framework from YAML configs."""
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

    # --- Process Technology Infrastructure ---
    tech_infra_config = load_yaml_config(os.path.join(CONFIG_DIR, "tech_infrastructure.yaml"))
    tech_df = pd.DataFrame.from_dict(tech_infra_config, orient='index')
    tech_df.to_csv(os.path.join(DATA_OUTPUT_DIR, 'technology_infrastructure.csv'))
    print(f"Saved technology_infrastructure.csv to {DATA_OUTPUT_DIR}")

    # --- Process Human Team Structure ---
    human_team_config = load_yaml_config(os.path.join(CONFIG_DIR, "human_team.yaml"))
    team_df = pd.DataFrame.from_dict(human_team_config, orient='index')
    team_df.to_csv(os.path.join(DATA_OUTPUT_DIR, 'human_team_structure.csv'))
    print(f"Saved human_team_structure.csv to {DATA_OUTPUT_DIR}")

    # --- Process Research Outputs ---
    research_outputs_config = load_yaml_config(os.path.join(CONFIG_DIR, "research_outputs.yaml"))
    outputs_df = pd.DataFrame.from_dict(research_outputs_config, orient='index')
    outputs_df.to_csv(os.path.join(DATA_OUTPUT_DIR, 'research_outputs_framework.csv'))
    print(f"Saved research_outputs_framework.csv to {DATA_OUTPUT_DIR}")

    # --- Process Partnership Strategy ---
    partnerships_config = load_yaml_config(os.path.join(CONFIG_DIR, "partnership_strategy.yaml"))
    partnerships_df = pd.DataFrame.from_dict(partnerships_config, orient='index')
    partnerships_df.to_csv(os.path.join(DATA_OUTPUT_DIR, 'partnership_strategy.csv'))
    print(f"Saved partnership_strategy.csv to {DATA_OUTPUT_DIR}")

    # --- Calculate and Print Summaries ---
    monthly_tech_cost = sum([item.get('monthly_cost_usd', 0) for item in tech_infra_config.values()])
    annual_tech_cost = monthly_tech_cost * 12
    annual_human_cost = sum([team.get('count', 0) * team.get('annual_salary_usd', 0) for team in human_team_config.values()])

    # Note: Operational costs are not in config yet, keeping them here for now.
    # We can move this to a YAML file as well in the next step.
    operational_costs = {
        'Office space': 24000, 'Legal and compliance': 15000, 'Insurance': 8000,
        'Marketing and events': 20000, 'Travel and conferences': 25000, 'Miscellaneous': 10000
    }
    total_annual_operational = sum(operational_costs.values())
    total_annual_cost = annual_tech_cost + annual_human_cost + total_annual_operational

    print("\n=== BANGLADESH AI THINK TANK - OPERATIONAL FRAMEWORK SUMMARY ===")
    print(f"Annual Cost Breakdown:")
    print(f"  - Technology Infrastructure: ${annual_tech_cost:,}")
    print(f"  - Human Resources: ${annual_human_cost:,}")
    print(f"  - Operational Costs: ${total_annual_operational:,}")
    print(f"  - TOTAL ANNUAL COST: ${total_annual_cost:,}")

    total_outputs = sum([output.get('annual_target', 0) for output in research_outputs_config.values()])
    total_review_hours = sum([output.get('annual_target', 0) * output.get('human_review_hours', 0) for output in research_outputs_config.values()])
    print(f"\nAnnual Research Output:")
    print(f"  - Total Publications: {total_outputs}")
    print(f"  - Total Human Review Hours: {total_review_hours:,}")

if __name__ == '__main__':
    generate_ops_framework_data()
