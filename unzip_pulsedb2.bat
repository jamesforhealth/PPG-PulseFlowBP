@echo off
setlocal enabledelayedexpansion

:: 設定目錄路徑
set "BASE_DIR=D:\PulseDB\PulseDB2_0"
set "MIMIC_DIR=%BASE_DIR%\PulseDB_MIMIC"
set "VITAL_DIR=%BASE_DIR%\PulseDB_Vital"

:: 創建目標目錄（如果不存在）
if not exist "%MIMIC_DIR%" mkdir "%MIMIC_DIR%"
if not exist "%VITAL_DIR%" mkdir "%VITAL_DIR%"

echo 開始整理 PulseDB 2.0 檔案...
echo 基礎目錄: %BASE_DIR%

:: 解壓縮所有 zip 檔案
for %%f in ("%BASE_DIR%\*.zip") do (
    echo 正在解壓縮: %%~nxf
    powershell -command "Expand-Archive -Path '%%f' -DestinationPath '%BASE_DIR%' -Force"
)

:: 移動 MIMIC 的 mat 檔案
echo 移動 MIMIC 檔案...
for /r "%BASE_DIR%\PulseDB_MIMIC" %%f in (*.mat) do (
    move "%%f" "%MIMIC_DIR%"
)

:: 移動 Vital 的 mat 檔案
echo 移動 Vital 檔案...
for /r "%BASE_DIR%\PulseDB_Vital" %%f in (*.mat) do (
    move "%%f" "%VITAL_DIR%"
)

:: 清理臨時解壓縮的目錄
echo 清理臨時檔案...
if exist "%BASE_DIR%\PulseDB_MIMIC" rd /s /q "%BASE_DIR%\PulseDB_MIMIC"
if exist "%BASE_DIR%\PulseDB_Vital" rd /s /q "%BASE_DIR%\PulseDB_Vital"

:: 顯示結果
echo.
echo 整理完成！
echo MIMIC 檔案位置: %MIMIC_DIR%
echo Vital 檔案位置: %VITAL_DIR%
echo.

:: 計算檔案數量
set /a mimic_count=0
set /a vital_count=0
for %%f in ("%MIMIC_DIR%\*.mat") do set /a mimic_count+=1
for %%f in ("%VITAL_DIR%\*.mat") do set /a vital_count+=1

echo MIMIC mat 檔案數量: %mimic_count%
echo Vital mat 檔案數量: %vital_count%

pause