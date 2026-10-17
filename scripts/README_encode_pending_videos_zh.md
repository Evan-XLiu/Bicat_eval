# 将录制时暂存的 PNG 编码成 LeRobot 视频

适用于本仓库 LeRobot v3.0 录制的数据：`meta/info.json` 中相机特征的
`dtype` 为 `video`，但末尾已保存的 episode 仍只有 `images/` 下的 PNG。
如果相机特征实际为 `image`，或只有图片文件夹而没有 episode、动作和时间戳元数据，
本脚本不适用。普通 PNG 转 MP4 不会自动生成完整的机器人训练数据集。

`--dataset.video_encoding_batch_size=120` 表示累积 120 个 episode 后才编码，
不是每 120 帧编码。正常退出时，`VideoEncodingManager` 也会编码剩余 episode；
强制结束进程、编码报错等情况可能留下 PNG。只有已经保存且元数据完整的 episode
可以恢复，未保存的录制片段不在本脚本处理范围内。

在仓库根目录执行，先停止录制，再检查：

```bash
.venv/bin/python scripts/encode_pending_videos.py \
  --repo-id local/cloth \
  --root /mnt/ssd/Evan/datasets/clothv1 \
  --dry-run
```

确认显示有待编码 episode 后，生成一个新的完整数据集：

```bash
.venv/bin/python scripts/encode_pending_videos.py \
  --repo-id local/cloth \
  --root /mnt/ssd/Evan/datasets/clothv1 \
  --output-root /mnt/ssd/Evan/datasets/clothv1_video
```

输出目录必须尚不存在，磁盘需容纳源数据副本及新视频。脚本保留源数据，
在副本中调用仓库原有编码器生成 MP4，并更新 episode 视频文件索引和时间偏移。
同一相机的多个 episode 可能共享一个视频文件，这是 LeRobot v3 的正常结构。
成功编码后，副本中对应 episode 的临时 PNG 会由原编码器删除。

脚本只处理所有相机都尚未编码的连续尾部 episode；已编码 episode 会跳过。
帧缺失、视频元数据部分写入、部分相机已经编码等情况会报错，避免重复追加。
若转换中途失败，源目录仍保留；请使用另一个新的输出目录重新从源数据转换，
不要直接对失败的副本重跑。

以后希望每个 episode 保存后就生成视频，可将录制参数改为：

```bash
--dataset.video=true \
--dataset.video_encoding_batch_size=1
```

这样会在每个 episode 保存时等待编码完成，可能增加 episode 之间的等待时间。
脚本使用仓库内部 API，升级 LeRobot 后需要重新验证兼容性。
