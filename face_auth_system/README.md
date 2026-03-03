# 人脸识别认证系统

考场/考务人员身份验证系统 - 实时检测人脸并显示身份信息

## 📦 两个版本

| 版本 | 优点 | 缺点 | 推荐场景 |
|------|------|------|----------|
| **标准版** | 识别精度高，API 简单 | 需要安装 dlib (较复杂) | 追求精度，有技术能力 |
| **轻量版** | 安装简单，快速上手 | 精度略低 | 快速部署，普通场景 |

## 🚀 快速开始

### 方案 A: 轻量版 (推荐首选)

**1. 安装依赖**
```bash
cd face_auth_system
pip install -r requirements_lite.txt
```

**2. 录入人员**
```bash
python enroll_lite.py
```

**3. 启动识别**
```bash
python recognize_lite.py
```

---

### 方案 B: 标准版 (高精度)

**1. 安装依赖**
```bash
cd face_auth_system
pip install -r requirements.txt
```
> ⚠️ **注意**: `face-recognition` 依赖 `dlib`，需要 CMake 和 C++ 编译环境。
> 如遇安装问题，请使用轻量版。

### 2. 录入人员信息

```bash
python enroll.py
```

操作步骤：
1. 输入姓名、编号/考号、身份证号
2. 按 `c` 键开始采集人脸（采集 10 张照片）
3. 保持正脸，光线充足
4. 采集完成后自动保存

### 3. 启动实时识别

```bash
python recognize.py
```

- 摄像头会自动打开
- 人脸进入画面时自动识别并显示姓名
- 按 `q` 退出，按 `s` 截图

## 📁 文件说明

```
face_auth_system/
├── enroll.py          # 人员录入程序
├── recognize.py       # 实时识别程序
├── database.json      # 人员信息数据库（自动生成）
├── known_faces/       # 人脸编码存储目录（自动生成）
├── requirements.txt   # Python 依赖
└── README.md          # 本说明文件
```

## ⚙️ 配置选项

在 `recognize.py` 顶部可调整：

```python
WEBCAM_INDEX = 0           # 摄像头索引（多摄像头时修改）
DETECTION_INTERVAL = 3     # 检测间隔帧数（越大越快但可能漏检）
CONFIDENCE_THRESHOLD = 0.6 # 识别置信度阈值（0-1，越高越严格）
```

## 🔧 常见问题

### Q1: `dlib` 安装失败

**Windows 解决方案：**
```bash
# 方法 1: 使用预编译 wheel
pip install dlib-19.24.0-cp314-cp314-win_amd64.whl

# 方法 2: 安装 Visual C++ Build Tools
# 下载：https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### Q2: 摄像头无法打开

- 检查摄像头是否被其他程序占用
- 尝试修改 `WEBCAM_INDEX` 为 1 或 2
- 检查摄像头权限设置

### Q3: 识别速度慢

- 增加 `DETECTION_INTERVAL`（如改为 5）
- 降低摄像头分辨率
- 确保光线充足

### Q4: 识别不准确

- 重新录入人脸（确保光线好、正脸）
- 降低 `CONFIDENCE_THRESHOLD`（如改为 0.5）
- 录入时多采集几张照片

## 🎯 使用场景

- ✅ 考场考生身份验证
- ✅ 考务人员签到
- ✅ 会议室签到
- ✅ 门禁系统

## ⚠️ 当前局限与注意事项

1. **无活体检测** — 无法区分真人和照片/视频，考场场景存在作弊风险
2. **环境依赖** — 光线不足、侧脸、遮挡（口罩/墨镜）会显著降低识别准确率
3. **距离敏感** — 最佳识别距离 0.8-1.2 米，过远或过近可能识别失败
4. **本地存储** — 人脸特征和数据存储于本地，多设备无法同步，人员需重新录入
5. **规模限制** — 适合 100 人以内场景，人员过多时识别速度和精度会下降

**适用场景：** 小型考场、会议室签到、内部人员验证等低安全风险场景

**不适用场景：** 高安全门禁、大规模考场、无人值守验证等

## 📝 数据库格式

`database.json` 示例：

```json
{
  "persons": [
    {
      "name": "张三",
      "id": "2024001",
      "id_card": "110101199001011234",
      "encoding_file": "2024001_20260228_120000.npy",
      "enrolled_at": "2026-02-28T12:00:00"
    }
  ],
  "created": "2026-02-28T12:00:00"
}
```

## 🚀 扩展建议

- 添加活体检测（防照片/视频作弊）
- 添加识别日志记录
- 添加 Web 管理界面
- 支持批量导入 Excel 名单

---

**版本**: v1.0  
**创建**: 2026-02-28
