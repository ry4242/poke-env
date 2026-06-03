import subprocess
import sys

for script in [
    "scripts/update_learnset.py",
    "scripts/update_moves.py",
    "scripts/update_pokedex.py",
]:
    subprocess.run([sys.executable, script], check=True)
