@echo off
rem 每個交易日美股收盤後（台灣時間早上 6 點以後）跑一次：記帳＋算出下一個交易日的單
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
B:\qfin-main\B\.venv-gpu\Scripts\python.exe B:\qfin-main\D\paper_trade.py
pause
