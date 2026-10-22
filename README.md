# 视场与遮挡查看

桌面程序，用来打开 URDF，查看相机和激光雷达的视场，并计算哪些表面被挡住。遮挡在点击「计算遮挡」后才计算。有 NVIDIA 显卡时，射线与网格的求交走 GPU，否则自动用 CPU。

需要 64 位 Python 3.10 或更高版本。

## Windows

在 Windows 上从源码运行。GPU 加速需要本机有 NVIDIA 驱动，以及用来编译的 CUDA Toolkit 和 MSVC。不需要再安装 CuPy 或 PyTorch。

1. 安装 [64 位 Python](https://www.python.org/downloads/windows/)，勾选 **Add python.exe to PATH**。
2. 安装当前显卡的 NVIDIA 驱动。程序只用第 0 块 NVIDIA 显卡。AMD 和核显会一直用 CPU。
3. 安装 [Visual Studio 2022 生成工具](https://visualstudio.microsoft.com/visual-cpp-build-tools/)，工作负载选择 **使用 C++ 的桌面开发**。`nvcc` 要调用其中的 `cl.exe`。
4. 安装 [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)。版本要被当前驱动支持；驱动很新时，装较新的 Toolkit。50 系显卡需要支持该架构的新 Toolkit，否则第一次计算会失败并改用 CPU。
5. 安装完成后重新打开终端，让 `nvcc` 和 `cl` 进入 PATH。

在仓库根目录执行：

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m view_simulate
```

若 PowerShell 禁止运行脚本，改用 `cmd`，执行 `.venv\Scripts\activate.bat` 后再运行上面的 `pip` 和 `python` 命令。

第一次点「计算遮挡」时，程序会用 `nvcc` 把 `view_simulate/ray_mesh.cu` 编译成同目录下的 `view_simulate/_raymesh.dll`。这一步可能要一两分钟。成功后状态栏显示显卡名称，例如 `NVIDIA GeForce RTX 2080 SUPER`。之后只要 `.cu` 没有改过，就不会再编译。

`nvcc` 不在默认位置时，先设置再启动：

```powershell
$env:CUDA_NVCC = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe"
python -m view_simulate
```

状态栏一直显示 `CPU` 时，在 `view_simulate` 目录手动编译，可以看到具体错误：

```powershell
nvcc -O3 -shared -o _raymesh.dll ray_mesh.cu
```

常见原因是终端里找不到 `cl.exe`（重新打开「x64 Native Tools」或装好 C++ 生成工具后的新终端），或者 Toolkit 太旧、编出来的代码不能在这块显卡上启动。

## Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m view_simulate
```

已有 NVIDIA 驱动和 CUDA Toolkit 时，第一次点「计算遮挡」会编译 `view_simulate/_raymesh.so`。本机若已经把依赖装在 `.deps` 里，也可以不建虚拟环境，在仓库根目录执行 `PYTHONPATH=.deps python3 -m view_simulate`。

## 打开机器人模型

用「载入 URDF」打开 `view_simulate/data/` 里的文件。网格路径相对 URDF 所在目录，机体网格在 `view_simulate/data/meshes/`，喷杆和药箱在 `view_simulate/data/10.21/`。

URDF 里仍是 `package://` 路径时，在界面的包搜索框填包的上一级目录。源码树对应 `weve-embedded-interface-emulator/src`。
