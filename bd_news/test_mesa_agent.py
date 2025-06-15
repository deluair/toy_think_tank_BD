import mesa

try:
    # Test basic Mesa Agent creation
    model = mesa.Model()
    agent = mesa.Agent('test', model)
    print('✓ Mesa Agent creation works')
    print(f'Agent ID: {agent.unique_id}')
    print(f'Agent model: {agent.model}')
except Exception as e:
    print(f'✗ Mesa Agent creation failed: {e}')