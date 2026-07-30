# FR5 + AG-160-95 组合 URDF

这是一个可直接在 PyBullet 中加载的法奥 FR5 与大寰 AG-160-95 二指夹爪组合模型。
主文件为 `urdf/fr5_ag160_95.urdf`。

## 已实现内容

- FR5 六轴运动链与原始关节限位；
- AG-160-95 完整基座、左右平行连杆、手指和指垫；
- 夹爪质量按厂商参数归一化为 1 kg；
- 夹爪命令范围：`0 rad` 为完全张开，`0.93 rad` 为闭合；
- 固定安装变换：FR5 `j6_Link` 到夹爪安装面为 `xyz="0 0 0.12"`、绕 Z 轴旋转 π；
- 标准抓取 TCP：`tcp_link`，位于夹爪安装面前方 0.190 m；
- 尺寸参考帧：`tcp_open_front_link`（0.188141 m）与
  `tcp_closed_front_link`（0.203700 m）；
- 优化视觉网格、碰撞网格、统一控制函数和自动验证脚本。

## 直接验证

在 PowerShell 中运行：

```powershell
cd "D:\pycharm\Project\FR5_AG-160-95"
& "D:\Anaconda\envs\gapg\python.exe" tools\build_model.py
& "D:\Anaconda\envs\gapg\python.exe" tools\validate_model.py
```

验证会检查 URDF 资源、21 个关节/固定帧、开合端点、自碰撞过滤和 TCP
逆运动学，并把报告与渲染图保存到 `validation`。

## 交互查看

```powershell
cd "D:\pycharm\Project\FR5_AG-160-95"
& "D:\Anaconda\envs\gapg\python.exe" examples\view_in_pybullet.py
```

PyBullet 窗口中的 `AG closure` 滑块范围为 0～1。

## 在场景代码中使用

```python
import pybullet as p
from tools.pybullet_model import load_robot, set_gripper

p.connect(p.DIRECT)
robot = load_robot()
set_gripper(robot, closure=0.0, use_motor=False)  # 张开
set_gripper(robot, closure=1.0)                   # 闭合电机命令
```

PyBullet 不会自动执行 URDF 的 `<mimic>` 约束，因此不能只设置一个夹爪关节。
`set_gripper()` 会同步设置六个连杆关节。`load_robot()` 同时关闭夹爪内部由于
URDF 无法表达闭环机构而产生的虚假自碰撞，但夹爪与场景物体之间的碰撞仍然保留。

## 模型边界

当前模型适合点云渲染、抓取规划、碰撞检测和二指夹持实验。AG-160-95 的真实机构
是闭环自适应连杆；标准 URDF 只能表达树结构，因此此处采用同步关节近似。若后续要
研究夹爪内部连杆受力或被动包络自适应，需要在 SDF/Gazebo、MuJoCo equality
constraint 或 PyBullet 用户约束中补上闭环约束。

原始 STEP、展开后的夹爪参考 URDF、FR5 原始 URDF 和第三方许可均保存在 `source`。
来源说明见 `NOTICE.md`。
