@echo off
rem Double-clickable launcher: opens the lidar-align GUI with no console window.
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0ui\refine_gui.py"
