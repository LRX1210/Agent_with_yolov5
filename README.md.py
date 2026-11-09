## 3. 详细部署步骤

### 3.1 环境要求
- Python >= 3.9
- Git（克隆项目使用）
- 可用USB/内置摄像头设备
- 网络可访问 Agent 后端服务地址

### 3.2 获取项目源码
#### 方式1：git克隆（推荐）
本项目使用 `git submodule` 引入YOLOv5官方代码，**克隆必须带上 `--recursive` 参数**，否则 yolov5 文件夹为空。
```bash
git clone --recursive https://github.com/你的用户名/Agent_with_yolo.git
cd Agent_with_yolo
##如果克隆时忘记加 `--recursive`，进入项目目录执行下面命令修复子模块：
##git submodule update --init --recursive
