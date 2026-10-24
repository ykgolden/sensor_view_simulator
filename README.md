# 视场与遮挡查看

桌面程序，用来打开 URDF，查看相机和激光雷达的视场，并计算哪些表面被挡住。遮挡在点击「计算遮挡」后才计算。Windows 和 Linux 上有 NVIDIA 显卡时，射线与网格的求交走 CUDA；macOS 上走系统 Metal，用的是这台 Mac 的 GPU。没有可用 GPU 时自动用 CPU。

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

## macOS

Mac 上不需要 NVIDIA 显卡、CUDA 或 Xcode。Apple 芯片和仍支持 Metal 的 Intel Mac 都用下面的步骤。把仓库拷到那台 Mac 上，用它自己的 Python 3.10 或更高版本安装。macOS 和 Homebrew 的 Python 通常不允许往系统环境里装包，所以先建虚拟环境。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m view_simulate
```

`pip` 在 macOS 上会额外安装 `pyobjc-framework-Metal`。第一次点「计算遮挡」时，系统 Metal 会编译 `view_simulate/ray_mesh.metal`，一般几秒钟。成功后状态栏显示显卡名称，例如 `Apple M2`。这次编译只留在当前进程里，下次重新打开程序会再编一次。

Metal 用不了时，终端会打印原因，状态栏保持 `CPU`，遮挡仍会算完。

那台 Mac 不能上网时，在另一台能上网、芯片相同的 Mac 上执行 `pip download -r requirements.txt -d wheels`，把 `wheels` 目录一起拷过去。对方在已经激活的虚拟环境里执行：

```bash
pip install --no-index --find-links wheels -r requirements.txt
python -m view_simulate
```

要在 Mac 上下载这些包。Linux 上的安装包不能在 Mac 上用，而且在 Linux 上执行 `pip download` 不会带上只在 macOS 安装的 Metal 依赖。

## 使用

启动后左侧是参数，右侧是三维视图。默认打开示例机器人和两个传感器：装在 `head_link` 上的头部相机、顶部雷达。左键旋转，中键平移，滚轮缩放。「复位视角」把画面重新对准机器人。

视场体会跟着关节和传感器参数立刻更新。遮挡着色和回波不会，改完之后要点「计算遮挡」。

### 机器人

「载入 URDF…」打开 `view_simulate/data/` 里的文件。网格路径相对 URDF 所在目录，机体网格在 `view_simulate/data/meshes/`，喷杆和药箱在 `view_simulate/data/10.21/`。URDF 里仍是 `package://` 路径时，先在包搜索框填包的上一级目录，或点「添加目录」；多个目录用分号分隔。源码树对应 `weve-embedded-interface-emulator/src`。xacro 要先展开成 URDF 再载入。

「示例机器人」回到自带的简单模型。「新建底座」换成一个长方体底座和一枚前视相机。「添加连杆…」在选中的父连杆上加盒子、圆柱或球，关节可以是固定、旋转或直线。「删除末端」删掉连杆列表里当前选中的、没有子连杆的那一节。

「关节」里列出可动关节。旋转关节的单位是度，直线关节的单位是米。拖动后机体和视场体马上跟着变。

### 传感器

在列表里选中一个传感器，再改下面的参数。点「添加相机」或「添加雷达」会在根连杆上放一个新传感器。「删除传感器」删掉当前选中的那个。

位置相对安装连杆，单位米。姿态是相对父连杆的 rpy，顺序与 URDF 相同，单位度。传感器坐标轴里红色是视线（+X），绿色是 +Y，蓝色是 +Z。水平角 0 朝前、正方向朝左；垂直角 0 水平、正方向朝上。雷达水平范围从 -180 到 180 是一整圈，垂直范围从 0 到 90 是上半球。

「水平采样」和「垂直采样」是回波射线的条数，只影响「传感器回波」，不影响表面着色的疏密。最远距离是这只传感器的视场尽头。

「保存布置…」和「载入布置…」读写传感器列表的 JSON，不包含机器人模型和关节角度。

### 遮挡

勾选「遮挡着色」或「传感器回波」后，点「计算遮挡」。默认一次算当前显示的全部传感器。勾了「只显示当前传感器视场」时只算列表里选中的那一个。没勾这两项就点计算，状态栏会提示先勾选。

着色点的颜色：

- 绿色：在视场内，且没有被挡住
- 红色：在视场内，但被机体、地面或围墙挡住
- 灰色：采样到了，但不在这只传感器的视场里

视场体用传感器各自的颜色画出，图例在参数区底部。状态栏在算完后显示绿色占用网格和红色占用网格的个数，以及这次用的是哪块 GPU 或 `CPU`。最高高度为 0 时没有占用网格，这两个数都是 0。格子按传感器合并：任一传感器能看见的格子算绿色，只被挡住、没有传感器看见的格子算红色。

这些选项的默认和含义：


| 选项         | 默认  | 作用                    |
| ---------- | --- | --------------------- |
| 视场体        | 开   | 画出相机锥体或雷达壳。关掉不影响遮挡结果  |
| 遮挡着色       | 开   | 在机体和地面上标出可见、被挡、视场外    |
| 传感器回波      | 关   | 画出打到表面上的射线落点          |
| 地面参与遮挡     | 开   | 地面会挡住视线，地面上的点也参与着色    |
| 四周围墙       | 开   | 在机器人四周放四面墙，用来看竖直方向的遮挡 |
| 只显示当前传感器视场 | 关   | 只画、只算列表中选中的那一只        |


数值里，「近处忽略」是传感器原点附近不计入的距离，避免装进壳体内部时整片被挡住。「表面间距」和「地面间距」越小，着色点越密，计算越慢。「地面高度」相对机器人根连杆，单位米。「围墙距离」是墙到原点的水平距离，机体比这个更大时墙会退到机体外侧。「围墙高度」从地面向上量。「最高高度」为 0 时不画占用网格；大于 0 时，从地面向上按格子显示遮挡，不再单独铺地面上的着色点。「网格大小」是每个格子的边长，格子太多时会自动加大。「网格不透明度」和「刻度范围」改完立刻生效，不用重新计算。刻度从 `-N` 画到 `N`，机器人更大时会自动再扩大。

关节、传感器参数，以及除视场体、不透明度、刻度范围以外的显示选项，改过之后都要再点一次「计算遮挡」，否则状态栏会保持「点计算遮挡更新」。

### 读取正在运行的窗口

程序启动后会在本机 `127.0.0.1` 上打开一个端口，窗口标题里能看到具体端口。这个端口只接受本机连接，用来读当前关节角度、传感器布置、显示选项，以及和画面一致的遮挡结果。参数改过还没重新计算时，遮挡标记为过期，绿格和红格按画面计为 0。

Cursor 通过项目里的 `.cursor/mcp.json` 启动 `python3 -m view_simulate.mcp_server`。可用的工具：

- `get_live_state`：当前参数和遮挡摘要
- `get_occlusion_cells`：给定长方体里、画面上实际画出的绿格和红格中心

改完 MCP 配置后要在 Cursor 里重新加载一次。窗口必须是这次改动之后重新打开的，旧进程没有这个端口。
