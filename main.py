import argparse
import os
import sys

# Add the new location of src to the Python path
# Assumes main.py is in the root and src is in think_tank/src
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, "think_tank"))

from think_tank.src.data_generation.policy_and_agents import generate_policy_and_agents_data
from think_tank.src.data_generation.ops_framework import generate_ops_framework_data
from think_tank.src.data_generation.workflows_and_metrics import generate_workflows_and_metrics_data
from think_tank.src.plotting.agent_specialization_chart import generate_agent_specialization_chart
from think_tank.src.plotting.timeline_chart import generate_timeline_chart

# Define output directories (still at the root level)
DATA_OUTPUT_DIR = os.path.join(project_root, "outputs/data")
FIGURE_OUTPUT_DIR = os.path.join(project_root, "outputs/figures")

def main():
    parser = argparse.ArgumentParser(description="Generate datasets and plots for the AI Think Tank project.")
    parser.add_argument("--generate-data", action="store_true", help="Generate all CSV datasets.")
    parser.add_argument("--generate-plots", action="store_true", help="Generate all plots.")
    parser.add_argument("--all", action="store_true", help="Generate all datasets and plots.")

    args = parser.parse_args()

    # Create output directories if they don't exist
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)

    # Adjust internal paths for modules if they also write to 'outputs'
    # This assumes the modules' internal DATA_OUTPUT_DIR and FIGURE_OUTPUT_DIR
    # are relative to the project root when main.py is run.
    # If modules expect to be run from 'think_tank/src', their paths might need adjustment too.
    # For now, we assume they use paths relative to where main.py is executed or absolute paths.

    if args.all or args.generate_data:
        print("--- Generating Policy and Agents Data ---")
        generate_policy_and_agents_data() # Modules will use their own output dir constants
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
