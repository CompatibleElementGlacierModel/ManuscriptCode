import subprocess

for j in range(0,25):
    print(j)
    subprocess.call(["python","./forward_mtw.py",f"{j}"])
