import subprocess

n_runs = 25

for j in range(0,n_runs):
    #print(j)
    subprocess.call(["python","./forward.py",f"{j}","0","1"])

for j in range(0,n_runs):
    #print(j)
    subprocess.call(["python","./forward.py",f"{j}","0","0"])

for j in range(0,n_runs):
    #print(j)
    subprocess.call(["python","./forward.py",f"{j}","1","0"])

for j in range(0,n_runs):
    #print(j)
    subprocess.call(["python","./forward.py",f"{j}","1","1"])
