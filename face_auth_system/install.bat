@echo off
chcp 65001 >nul
echo ================================================
echo   人脸识别认证系统 - 依赖安装
echo ================================================
echo.
echo 请选择安装版本:
echo   1. 轻量版 (推荐，快速安装)
echo   2. 标准版 (高精度，需要编译环境)
echo.

set /p choice="输入选项 (1/2): "

if "%choice%"=="2" (
    echo.
    echo 正在安装标准版依赖...
    pip install -r requirements.txt
) else (
    echo.
    echo 正在安装轻量版依赖...
    pip install -r requirements_lite.txt
)

echo.
echo ================================================
echo   安装完成！
echo ================================================
echo.
echo 下一步:
echo   1. 录入人员：python enroll_lite.py  (或 enroll.py)
echo   2. 启动识别：python recognize_lite.py  (或 recognize.py)
echo.
pause
