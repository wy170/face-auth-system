# 下载人脸识别所需模型文件
# 运行：.\download_models.ps1

$modelsDir = $PSScriptRoot
Set-Location $modelsDir

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "  人脸识别模型下载脚本" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan

# OpenFace 人脸识别模型 (12MB)
$openfaceUrl = "https://raw.githubusercontent.com/pyannote/pyannote-data/master/openface.nn4.small2.v1.t7"
$openfaceFile = "openface.nn4.small2.v1.t7"

# SSD 人脸检测模型 (11MB)
$ssdModelUrl = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face/master/res10_300x300_ssd_iter_140000.caffemodel"
$ssdModelFile = "res10_300x300_ssd_iter_140000.caffemodel"

# SSD 检测器配置文件
$ssdProtoUrl = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
$ssdProtoFile = "deploy.prototxt"

function Download-File {
    param($url, $output)
    
    if (Test-Path $output) {
        Write-Host "[SKIP] $output 已存在" -ForegroundColor Yellow
        return $true
    }
    
    Write-Host "[INFO] 下载 $output ..." -ForegroundColor Cyan
    
    try {
        Invoke-WebRequest -Uri $url -OutFile $output -UseBasicParsing
        if (Test-Path $output) {
            $size = (Get-Item $output).Length / 1MB
            Write-Host "[OK] 下载完成：{0:F2}MB" -f $size -ForegroundColor Green
            return $true
        }
    } catch {
        Write-Host "[ERROR] 下载失败：$_" -ForegroundColor Red
        return $false
    }
    
    return $false
}

Write-Host "`n[INFO] 开始下载模型文件..." -ForegroundColor White

$success = 0
$total = 3

if (Download-File $openfaceUrl $openfaceFile) { $success++ }
if (Download-File $ssdModelUrl $ssdModelFile) { $success++ }
if (Download-File $ssdProtoUrl $ssdProtoFile) { $success++ }

Write-Host "`n==================================" -ForegroundColor Cyan
Write-Host "  下载完成：$success/$total" -ForegroundColor $(if($success -eq $total){"Green"}else{"Yellow"})
Write-Host "==================================" -ForegroundColor Cyan

if ($success -eq $total) {
    Write-Host "`n[INFO] 现在可以运行录入和识别程序了！" -ForegroundColor Green
    Write-Host "  1. 重新录入人员：python enroll_lite.py" -ForegroundColor White
    Write-Host "  2. 启动识别系统：python recognize_lite.py" -ForegroundColor White
} else {
    Write-Host "`n[WARN] 部分模型下载失败，但程序仍可运行（使用备选方案）" -ForegroundColor Yellow
    Write-Host "  - 如果没有 OpenFace 模型，将无法生成特征编码" -ForegroundColor Gray
    Write-Host "  - 如果没有 SSD 模型，将使用 Haar 级联检测器（速度更快但精度略低）" -ForegroundColor Gray
}

Write-Host "`n按任意键退出..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
