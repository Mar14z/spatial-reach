import sys
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator as ea

a = ea.EventAccumulator(str(Path(sys.argv[1]) / sys.argv[2]))
a.Reload()
tags = a.Tags()["scalars"]
loss = next((t for t in tags if t.endswith("损失")), None)
print("tags:", len(tags), "| 损失 last step:", a.Scalars(loss)[-1].step if loss else "n/a")
for k in [
    "1-主指标(验证集)/平均像素误差(px,软解码)",
    "1-主指标(验证集)/PCK@3px(软解码)",
    "1-主指标(验证集)/95分位像素误差(px,argmax)",
    "1-主指标(验证集)/NME归一化误差",
]:
    if k in tags:
        s = a.Scalars(k)
        print(f"{k.split('/')[-1]:28s} " + "  ".join(f"{x.step}:{x.value:.3f}" for x in s[-3:]))
names = [t.split("/")[1] for t in tags if t.startswith("2-逐点/")]
print("逐点:", names)
for n in names:
    px = a.Scalars(f"2-逐点/{n}/像素误差(px,软解码)")[-1].value
    bv = a.Scalars(f"2-逐点/{n}/纵向偏差v(px)")[-1].value
    print(f"  {n:6s} {px:5.2f}px  偏差v {bv:+.2f}")
