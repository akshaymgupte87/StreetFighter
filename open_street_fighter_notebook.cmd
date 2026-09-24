@echo off
setlocal
cd /d "%~dp0"
echo Opening the Street Fighter notebook with Python 3.14 in WSL2...
wsl.exe -d Ubuntu -- bash -lc "cd /mnt/c/Users/aksha/PycharmProjects/StreetFighter && /home/akshay/.venvs/streetfighter314/bin/python -m notebook street_fighter_mixed_moves.ipynb"
endlocal
