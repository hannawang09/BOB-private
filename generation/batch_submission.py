import os

for i in range(50):
    os.system(f"sbatch run.sh {i}")