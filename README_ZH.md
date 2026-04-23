# ComfyUI-BodyRatioMapper

[中文](./README_ZH.md) | [English](./README.md)

一个基于 SDPose 的姿势对齐与人体比例映射项目。

## 项目简介

`ComfyUI-BodyRatioMapper` 的核心目标是使“目标动作”与“参考体型”相匹配，对目标动作进行相较于参考图的对齐和比例缩放。本项目也是首个支持多人场景的此类项目。

适用场景：

- 图像或视频的人体动作迁移前处理
- 需要统一角色体型比例的关键点序列生成

## 功能特性

- 姿势对齐（Pose Alignment）
- 人体比例映射（Body Ratio Mapping）
- 提供关键点渲染节点，便于可视化检查
- 支持多人场景

## 效果展示

### 单图示例

- 工作流示例：

![工作流示例](docs/images/image_001.png)

- 对比结果：

![对比结果](docs/images/image_002.png)

### 视频示例


- 单人对比：[video_001.mp4](docs/video/video_001.mp4)、[video_002.mp4](docs/video/video_002.mp4)
- 多人对比：[video_003.mp4](docs/video/video_003.mp4)

### 对比结论

实测表明：将人体骨骼与参考图对齐后，可明显提升动作迁移中的人物一致性。  
在与同类方案（如 One To All animation、ProportionChanger）对比时，本项目表现更优。  
测试使用动作迁移模型为 SteadyDancer，且包含随机种子在内的参数保持一致。


## 安装方式

将本仓库放入 ComfyUI 的 `custom_nodes` 目录并安装依赖：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/wuwukaka/ComfyUI-BodyRatioMapper.git
cd ComfyUI-BodyRatioMapper
pip install -r requirements.txt
```

安装完成后重启 ComfyUI。

## 快速开始

1. 启动 ComfyUI 并加载本插件节点。
2. 导入示例工作流：
`example_workflows/bodyratiomapper_officail_sdpose_image.json`
`example_workflows/bodyratiomapper_officail_sdpose_video.json`

## 节点说明

### 1. BodyRatioMapper Proportion Transfer

- 节点名：`BodyRatioMapperProportionTransfer`
- 作用：执行姿势对齐与人体比例映射，输出变换后的关键点和锚点关键点。
- 主要输入：
`pose_keypoint`
`ref_pose_keypoint`
`manual_anchor_pose`
- 常用参数：
`alignment_mode`、`hand_scaling`、`foot_scaling`、`offset_stabilizer`、`confidence_threshold`、`output_absolute_coordinates`
- 输出：
`changed_pose_keypoint`、`anchor_pose_keypoint`

### 2. BodyRatioMapper SDPose Render

- 节点名：`BodyRatioMapperSDPoseRender`
- 作用：将关键点渲染为骨架图
- 常用参数：
`resolution_x`、`score_threshold`、`stick_width`、`face_point_size`、`draw_face`、`draw_hands`、`draw_feet`
- 输出：`IMAGE`

### 3. pose_keypoint input

- 节点名：`PoseJSONToPoseKeypoint`
- 作用：将 JSON 字符串转换为 `POSE_KEYPOINT`，便于手工调试或粘贴外部关键点。

### 4. pose_keypoint preview

- 节点名：`PoseKeypointPreview`
- 作用：将 `POSE_KEYPOINT` 转为 JSON 文本并显示在节点上，便于复制、检查和回灌。

## 项目结构

```text
ComfyUI-BodyRatioMapper/
├─ body_ratio_mapper/
│  ├─ core_modules/
│  ├─ proportion_transfer_node.py
│  └─ render_nodes.py
├─ web/js/poseKeypointPreview.js
├─ example_workflows/
├─ docs/
├─ nodes.py
├─ requirements.txt
├─ pyproject.toml
└─ __init__.py
```

## 常见问题（FAQ）

### 1. 安装后看不到节点

- 确认仓库路径在 `ComfyUI/custom_nodes/ComfyUI-BodyRatioMapper`
- 检查依赖是否安装成功
- 重启 ComfyUI 并查看启动日志是否有报错

### 2. 关键点显示异常或为空

- 检查输入是否为有效 `POSE_KEYPOINT`
- 适当降低 `confidence_threshold`
- 用 `pose_keypoint preview` 先检查关键点 JSON 是否完整

## 致谢

- 部分代码来自 `grmchn/ComfyUI-ProportionChanger`：
https://github.com/grmchn/ComfyUI-ProportionChanger
- 感谢朋友望星铭（https://space.bilibili.com/13066617）和阿临（https://space.bilibili.com/20848068）提供 OC 测试素材。

## 许可证

因部分代码来源于 GPL v3.0 项目 `ComfyUI-ProportionChanger`，因此本项目也采用 **GPL v3.0** 协议发布。  


