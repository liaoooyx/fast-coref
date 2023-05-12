@echo off
set /a startS=%time:~6,2%
set /a startM=%time:~3,2%
echo Start at %time%
python E:\myproject\vscode_workspace\fast-coref\src\main.py ^
infra=win experiment=win_exp_7a ^
paths.model_name=model_7a01
set /a endS=%time:~6,2%
set /a endM=%time:~3,2%
echo Finish at %time%
set /a diffS_=%endS%-%startS%
set /a diffM_=%endM%-%startM%
echo Time Cost:%diffM_%min %diffS_%s
pause