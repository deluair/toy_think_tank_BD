import mesa
import inspect

print(f"Mesa version: {mesa.__version__}")
print("\nMesa Model attributes:")
for attr in dir(mesa.Model):
    if 'agents' in attr.lower():
        print(f"  {attr}")
        
print("\nMesa Model properties:")
for name, obj in inspect.getmembers(mesa.Model):
    if isinstance(obj, property):
        print(f"  {name}: {obj}")
        if 'agents' in name.lower():
            print(f"    -> This is a property with agents!")