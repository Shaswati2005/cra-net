@echo off
REM One command for the full batch-size sweep: GAIT-IT, INIT-6, INIT-8 x batch 4, 8, 16.
REM Double-click this file, or run "run_all.bat" from Command Prompt.
REM If it stops (crash, closed window), run it again: finished runs are skipped.
cd /d "%~dp0"
set PY=C:\Users\Asus\AppData\Local\Programs\Python\Python313\python.exe
set PYTHONIOENCODING=utf-8
REM Keep the laptop awake on mains power while the sweep runs (restored at the end).
powercfg /change standby-timeout-ac 0
"%PY%" -u run_all_experiments.py %*
powercfg /change standby-timeout-ac 30
echo.
echo Sweep finished. Results: %~dp0sweep_results\SWEEP_SUMMARY.md
pause
