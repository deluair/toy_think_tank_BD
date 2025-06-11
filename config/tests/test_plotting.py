import os
import sys
import pytest

# Add project root to path to allow importing from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, "think_tank"))

# Run data generation first to ensure plots have data
from src.data_generation.policy_and_agents import generate_policy_and_agents_data
from src.plotting.agent_specialization_chart import generate_agent_specialization_chart
from src.plotting.timeline_chart import generate_timeline_chart

# Define expected output files
FIGURES_DIR = os.path.join(project_root, 'outputs', 'figures')
EXPECTED_PNG_FILES = [
    'ai_agents_chart.png',
    'implementation_timeline_chart.png'
]

@pytest.fixture(scope="module")
def run_plotting_logic():
    """Fixture to run all plotting functions once per test module."""
    # Ensure data dependencies exist
    generate_policy_and_agents_data()

    # Ensure the output directory is clean before running
    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)
    for f in os.listdir(FIGURES_DIR):
        os.remove(os.path.join(FIGURES_DIR, f))

    generate_agent_specialization_chart()
    generate_timeline_chart()

@pytest.mark.parametrize("filename", EXPECTED_PNG_FILES)
def test_png_file_is_created(run_plotting_logic, filename):
    """Test that each expected PNG file is created."""
    file_path = os.path.join(FIGURES_DIR, filename)
    assert os.path.exists(file_path), f"{filename} was not created in {FIGURES_DIR}"
    assert os.path.getsize(file_path) > 0, f"{filename} is an empty file."
