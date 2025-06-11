import os
import sys
import pytest

# Add project root to path to allow importing from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, "think_tank"))

from src.data_generation.policy_and_agents import generate_policy_and_agents_data
from src.data_generation.ops_framework import generate_ops_framework_data
from src.data_generation.workflows_and_metrics import generate_workflows_and_metrics_data

# Define expected output files
DATA_DIR = os.path.join(project_root, 'outputs', 'data')
EXPECTED_CSV_FILES = [
    'bangladesh_policy_priorities.csv',
    'ai_agent_specializations.csv',
    'technology_infrastructure.csv',
    'human_team_structure.csv',
    'research_outputs_framework.csv',
    'partnership_strategy.csv',
    'agent_workflows.csv',
    'stakeholder_engagement.csv',
    'innovation_pipeline.csv',
    'performance_metrics.csv',
    'revenue_diversification.csv'
]

@pytest.fixture(scope="module")
def run_data_generation():
    """Fixture to run all data generation functions once per test module."""
    # Ensure the output directory is clean before running
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    for f in os.listdir(DATA_DIR):
        os.remove(os.path.join(DATA_DIR, f))
    
    generate_policy_and_agents_data()
    generate_ops_framework_data()
    generate_workflows_and_metrics_data()

@pytest.mark.parametrize("filename", EXPECTED_CSV_FILES)
def test_csv_file_is_created(run_data_generation, filename):
    """Test that each expected CSV file is created."""
    file_path = os.path.join(DATA_DIR, filename)
    assert os.path.exists(file_path), f"{filename} was not created in {DATA_DIR}"
    assert os.path.getsize(file_path) > 0, f"{filename} is an empty file."

def test_all_expected_files_are_present(run_data_generation):
    """Test that all expected files, and only them, are in the directory."""
    actual_files = set(os.listdir(DATA_DIR))
    expected_files = set(EXPECTED_CSV_FILES)
    assert actual_files == expected_files, f"Mismatch in generated files.\nExpected: {expected_files}\nGot: {actual_files}"
