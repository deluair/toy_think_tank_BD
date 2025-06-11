import argparse
import os

from src.data_generation.policy_and_agents import generate_policy_and_agents_data
from src.data_generation.ops_framework import generate_ops_framework_data
from src.data_generation.workflows_and_metrics import generate_workflows_and_metrics_data
from src.plotting.agent_specialization_chart import generate_agent_specialization_chart
from src.plotting.timeline_chart import generate_timeline_chart

# Define output directories
DATA_OUTPUT_DIR = "outputs/data"
FIGURE_OUTPUT_DIR = "outputs/figures"

def main():
    parser = argparse.ArgumentParser(description="Generate datasets and plots for the AI Think Tank project.")
    parser.add_argument("--generate-data", action="store_true", help="Generate all CSV datasets.")
    parser.add_argument("--generate-plots", action="store_true", help="Generate all plots.")
    parser.add_argument("--all", action="store_true", help="Generate all datasets and plots.")

    args = parser.parse_args()

    # Create output directories if they don't exist
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)

    if args.all or args.generate_data:
        print("--- Generating Policy and Agents Data ---")
        generate_policy_and_agents_data()
        print("\n--- Generating Operational Framework Data ---")
        generate_ops_framework_data()
        print("\n--- Generating Workflows and Metrics Data ---")
        generate_workflows_and_metrics_data()
        print("\nAll data generation complete.")

    if args.all or args.generate_plots:
        print("\n--- Generating Agent Specialization Chart ---")
        generate_agent_specialization_chart()
        print("\n--- Generating Implementation Timeline Chart ---")
        generate_timeline_chart()
        print("\nAll plot generation complete.")

    if not (args.generate_data or args.generate_plots or args.all):
        print("No action specified. Use --generate-data, --generate-plots, or --all.")
        parser.print_help()

if __name__ == "__main__":
    main()
