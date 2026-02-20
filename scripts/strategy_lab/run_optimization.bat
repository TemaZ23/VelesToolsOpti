@echo off
cd /d C:\Users\temay\Desktop\veles-tools-main
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe -u -m scripts.strategy_lab.run --years 2 --target 15 --rounds 10 --trials 300 --splits 5 --train-ratio 0.5 --min-trades 30 --patience 3 > output\strategy_lab\opt_stdout.txt 2> output\strategy_lab\opt_stderr.txt
echo.
echo === OPTIMIZATION COMPLETE ===
echo Exit code: %ERRORLEVEL%
pause
