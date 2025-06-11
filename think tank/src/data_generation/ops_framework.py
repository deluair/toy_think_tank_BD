import pandas as pd
import os

DATA_OUTPUT_DIR = "outputs/data"

def generate_ops_framework_data():
    """Generates and saves data for the operational framework."""
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

    tech_infrastructure = {
        'Cloud Computing': {
            'provider': 'AWS/Azure/Google Cloud',
            'monthly_cost_usd': 15000,
            'components': ['Compute instances', 'Storage', 'AI/ML services', 'Data pipeline'],
            'justification': 'Scalable infrastructure for 100 AI agents'
        },
        'AI/ML Platforms': {
            'provider': 'OpenAI API, Anthropic, Custom Models',
            'monthly_cost_usd': 25000,
            'components': ['LLM access', 'Custom model training', 'Fine-tuning', 'API calls'],
            'justification': 'Core AI capabilities for specialized agents'
        },
        'Data Sources': {
            'provider': 'Multiple vendors',
            'monthly_cost_usd': 8000,
            'components': ['Economic databases', 'News feeds', 'Government data', 'Research portals'],
            'justification': 'Comprehensive data access for analysis'
        },
        'Security & Compliance': {
            'provider': 'Enterprise security solutions',
            'monthly_cost_usd': 5000,
            'components': ['Encryption', 'Access control', 'Monitoring', 'Backup'],
            'justification': 'Protecting sensitive policy research'
        },
        'Development Tools': {
            'provider': 'Software licenses',
            'monthly_cost_usd': 3000,
            'components': ['IDEs', 'Version control', 'Testing tools', 'Monitoring'],
            'justification': 'AI agent development and maintenance'
        },
        'Communication & Collaboration': {
            'provider': 'Multiple platforms',
            'monthly_cost_usd': 2000,
            'components': ['Video conferencing', 'Document sharing', 'Project management'],
            'justification': 'Stakeholder engagement and collaboration'
        }
    }
    tech_df = pd.DataFrame.from_dict(tech_infrastructure, orient='index')
    tech_df.to_csv(os.path.join(DATA_OUTPUT_DIR, 'technology_infrastructure.csv'))
    print(f"Saved technology_infrastructure.csv to {DATA_OUTPUT_DIR}")

    human_team = {
        'Executive Director': {
            'count': 1,
            'annual_salary_usd': 120000,
            'responsibilities': ['Strategic oversight', 'Stakeholder relations', 'Quality control', 'Public representation'],
            'required_skills': ['Policy expertise', 'AI literacy', 'Leadership', 'Communication']
        },
        'Technical Lead': {
            'count': 1,
            'annual_salary_usd': 100000,
            'responsibilities': ['AI agent development', 'System architecture', 'Technical oversight', 'Innovation'],
            'required_skills': ['AI/ML expertise', 'Software development', 'System design', 'Problem solving']
        },
        'Research Coordinator': {
            'count': 1,
            'annual_salary_usd': 70000,
            'responsibilities': ['Research planning', 'Quality assurance', 'Methodology oversight', 'Output coordination'],
            'required_skills': ['Research methods', 'Data analysis', 'Policy knowledge', 'Project management']
        },
        'Data Manager': {
            'count': 1,
            'annual_salary_usd': 80000,
            'responsibilities': ['Data governance', 'Source management', 'Pipeline maintenance', 'Quality control'],
            'required_skills': ['Data engineering', 'Database management', 'Data quality', 'Automation']
        },
        'Communications Specialist': {
            'count': 1,
            'annual_salary_usd': 60000,
            'responsibilities': ['Report writing', 'Stakeholder communication', 'Public engagement', 'Dissemination'],
            'required_skills': ['Writing', 'Communication', 'Digital marketing', 'Stakeholder management']
        }
    }
    team_df = pd.DataFrame.from_dict(human_team, orient='index')
    team_df.to_csv(os.path.join(DATA_OUTPUT_DIR, 'human_team_structure.csv'))
    print(f"Saved human_team_structure.csv to {DATA_OUTPUT_DIR}")

    research_outputs = {
        'Policy Briefs': {
            'frequency': 'Weekly',
            'annual_target': 52,
            'length_pages': '4-6',
            'ai_agents_involved': 15,
            'human_review_hours': 8
        },
        'Comprehensive Reports': {
            'frequency': 'Monthly',
            'annual_target': 12,
            'length_pages': '30-50',
            'ai_agents_involved': 40,
            'human_review_hours': 40
        },
        'Rapid Analysis': {
            'frequency': 'As needed',
            'annual_target': 100,
            'length_pages': '2-3',
            'ai_agents_involved': 8,
            'human_review_hours': 3
        },
        'Annual Outlook': {
            'frequency': 'Annually',
            'annual_target': 1,
            'length_pages': '100-150',
            'ai_agents_involved': 80,
            'human_review_hours': 200
        },
        'Sectoral Deep Dives': {
            'frequency': 'Quarterly',
            'annual_target': 4,
            'length_pages': '60-80',
            'ai_agents_involved': 60,
            'human_review_hours': 120
        }
    }
    outputs_df = pd.DataFrame.from_dict(research_outputs, orient='index')
    outputs_df.to_csv(os.path.join(DATA_OUTPUT_DIR, 'research_outputs_framework.csv'))
    print(f"Saved research_outputs_framework.csv to {DATA_OUTPUT_DIR}")

    partnerships = {
        'Government Agencies': {
            'target_partners': ['Prime Minister\'s Office', 'Planning Commission', 'Ministry of Finance', 'ICT Division'],
            'collaboration_type': 'Data sharing, policy consultation',
            'value_proposition': 'Evidence-based policy recommendations'
        },
        'International Organizations': {
            'target_partners': ['World Bank', 'ADB', 'UNDP', 'IMF'],
            'collaboration_type': 'Research collaboration, funding',
            'value_proposition': 'Advanced analytics capabilities'
        },
        'Universities': {
            'target_partners': ['University of Dhaka', 'BUET', 'BRAC University', 'NSU'],
            'collaboration_type': 'Research partnerships, talent pipeline',
            'value_proposition': 'Cutting-edge research methods'
        },
        'Think Tanks': {
            'target_partners': ['CPD', 'BIISS', 'PRI', 'BIDS'],
            'collaboration_type': 'Joint research, knowledge sharing',
            'value_proposition': 'Complementary expertise'
        },
        'Technology Partners': {
            'target_partners': ['OpenAI', 'Microsoft', 'Google', 'Local tech companies'],
            'collaboration_type': 'Technology access, innovation',
            'value_proposition': 'Pioneer AI applications in policy'
        }
    }
    partnerships_df = pd.DataFrame.from_dict(partnerships, orient='index')
    partnerships_df.to_csv(os.path.join(DATA_OUTPUT_DIR, 'partnership_strategy.csv'))
    print(f"Saved partnership_strategy.csv to {DATA_OUTPUT_DIR}")

    monthly_tech_cost = sum([item['monthly_cost_usd'] for item in tech_infrastructure.values()])
    annual_tech_cost = monthly_tech_cost * 12
    annual_human_cost = sum([team['count'] * team['annual_salary_usd'] for team in human_team.values()])

    operational_costs = {
        'Office space': 24000,  # Annual
        'Legal and compliance': 15000,
        'Insurance': 8000,
        'Marketing and events': 20000,
        'Travel and conferences': 25000,
        'Miscellaneous': 10000
    }
    total_annual_operational = sum(operational_costs.values())
    total_annual_cost = annual_tech_cost + annual_human_cost + total_annual_operational

    print("\n=== BANGLADESH AI THINK TANK - OPERATIONAL FRAMEWORK ===")
    print(f"Annual Cost Breakdown:")
    print(f"Technology Infrastructure: ${annual_tech_cost:,}")
    print(f"Human Resources: ${annual_human_cost:,}")
    print(f"Operational Costs: ${total_annual_operational:,}")
    print(f"TOTAL ANNUAL COST: ${total_annual_cost:,}")
    print(f"Monthly Technology Cost: ${monthly_tech_cost:,}")
    print(f"Team Size: {sum([team['count'] for team in human_team.values()])} people")

    total_outputs = sum([output['annual_target'] for output in research_outputs.values()])
    total_review_hours = sum([output['annual_target'] * output['human_review_hours'] for output in research_outputs.values()])
    print(f"\nAnnual Research Output:")
    print(f"Total Publications: {total_outputs}")
    print(f"Total Human Review Hours: {total_review_hours:,}")
    print(f"Average Hours per Output: {total_review_hours/total_outputs:.1f}")

if __name__ == '__main__':
    generate_ops_framework_data()
