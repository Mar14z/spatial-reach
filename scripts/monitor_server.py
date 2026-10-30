"""实验监控台：零依赖本地 web 服务，实时看评测结果与运行日志。

用法（arm_sim）：python scripts/monitor_server.py --port 8765
页面每 3 秒拉一次 /api/state：自动发现 outputs/eval/*.json 的新结果、
blind_views 证据图、outputs/*.log 最新日志。跑实验时保持页面打开即可。
"""
import argparse
import json
import math
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs/eval"
VIEWS = EVAL / "blind_views"

# 卡片标题的中文名：原始文件名对不熟悉评测脚本的人没有信息量
EXP_NAMES = {
    "closed_loop_scale.json": "闭环抓取 v1 · 无拒识",
    "closed_loop_scale_v2.json": "闭环抓取 v2 · 拒识上线",
    "cl_gap_val.json": "退化解诊断 · val",
    "cl_gap_val_z.json": "退化解诊断 · 含物理拒识",
    "closed_loop.json": "闭环抓取 · 早期基线",
    "closed_loop_az0.json": "闭环抓取 · az=0 消融",
    "closed_loop_det0.json": "闭环抓取 · det=0 消融",
    "closed_loop_det15.json": "闭环抓取 · det=15 消融",
    "closed_loop_v3.json": "抓取代理 · 完整任务统计",
    "memory_closed_loop_v3.json": "记忆抓取代理 · 完整任务统计",
    "memory_closed_loop.json": "记忆闭环 L2 · A记B抓",
    "memory_closed_loop_nosim.json": "记忆闭环 L2 · 纯几何（无仿真）",
    "blind_view.json": "失明视角搜索 L1",
}

# 记忆闭环分组的中文名：同一张卡里混着「该抓到」和「该抓空」，不分组会看不懂
COND_NAMES = {"mem": "靠记忆抓", "stale": "过期记忆（应抓空）",
              "reobs": "B 理想重观测（代理）", "memory": "记忆精度"}


STATUS_NAMES = {"executed": "已执行", "geometry_only": "仅几何",
                "execution_failed": "执行异常", "geometry_failed": "几何失败",
                "missing_frame": "缺帧", "pose_failed": "位姿失败",
                "calibration_skipped": "标定数据不足", "calibration_failed": "标定失败",
                "calibration_rejected": "标定拒识", "a_memory_unavailable": "无有效记忆",
                "b_calibration_unavailable": "B 标定不可用", "target_not_visible": "目标不可见",
                "visibility_unverified": "可见性未验证"}


def _status_text(totals):
    return " · ".join(f"{STATUS_NAMES.get(k, k)} {v}"
                      for k, v in (totals or {}).get("status_counts", {}).items())


def _num(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(v) else None
    return None


def _pct(xs, q):
    if not xs:
        return None
    ys = sorted(xs)
    return ys[min(len(ys) - 1, int(q * len(ys)))]


def summarize(path: Path):
    """通用评测汇总：dict/list 里有 err_mm 行的都收（closed_loop*/cl_gap*）。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError, OSError):
        return None
    rows = data.get("rows") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return None
    rows = [r for r in rows if isinstance(r, dict)]
    if not any("err_mm" in r or "status" in r for r in rows):
        return None
    totals = data.get("task_summary") if isinstance(data, dict) else None
    grouped_totals = data.get("cond_summary", {}) if isinstance(data, dict) else {}
    legacy = not isinstance(data, dict) or data.get("protocol_version", 0) < 3
    errs = [e for e in (_num(r.get("err_mm")) for r in rows) if e is not None]
    tagged = [r for r in rows if "ok_grasp" in r]
    grasped = sum(bool(r["ok_grasp"]) for r in tagged)
    conds = None
    if any("cond" in r for r in rows):
        agg: dict = {}
        for r in rows:
            d = agg.setdefault(r.get("cond", "-"), {"n": 0, "t": 0, "grasp": 0, "errs": []})
            d["n"] += 1
            e = _num(r.get("err_mm"))
            if e is not None:
                d["errs"].append(e)
            if "ok_grasp" in r:
                d["t"] += 1
                d["grasp"] += bool(r["ok_grasp"])
        conds = [{"cond": COND_NAMES.get(k, k), "n": v["n"],
                  "grasp_rate": (_num(grouped_totals[k].get("attempt_success_rate"))
                      if k in grouped_totals and "n_attempted" in grouped_totals[k]
                      else round(v["grasp"] / v["t"], 4) if v["t"] else None),
                  "status_text": _status_text(grouped_totals.get(k)),
                  "n_planned": grouped_totals.get(k, {}).get("n_planned"),
                  "n_attempted": grouped_totals.get(k, {}).get("n_attempted"),
                  "end_to_end_rate": _num(grouped_totals.get(k, {}).get("end_to_end_success_rate")),
                  "err_med": _pct(v["errs"], 0.5)} for k, v in agg.items()]
    return {
        "file": path.name,
        "label": EXP_NAMES.get(path.name, f"评测结果 · {path.stem}"),
        "mtime": time.strftime("%m-%d %H:%M", time.localtime(path.stat().st_mtime)),
        "n": len(rows),
        "n_tagged": len(tagged), "legacy": legacy, "status_text": _status_text(totals),
        "n_planned": totals.get("n_planned") if totals else None,
        "n_attempted": totals.get("n_attempted") if totals else None,
        "end_to_end_rate": _num(totals.get("end_to_end_success_rate")) if totals else None,
        "grasp_rate": (_num(totals.get("attempt_success_rate")) if totals else
                       round(grasped / len(tagged), 4) if tagged else None),
        "err_med": _pct(errs, 0.5),
        "err_p95": _pct(errs, 0.95),
        "err_max": max(errs) if errs else None,
        "conds": conds,
        "pts": [[_num(r.get("err_mm")), _num(r.get("rms_px")),
                 None if "ok_grasp" not in r else bool(r["ok_grasp"])]
                for r in rows],
    }


def blind_summary():
    path = EVAL / "blind_view.json"
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(d, dict) or not isinstance(d.get("rows"), list):
        return None
    ok = [r for r in d["rows"] if isinstance(r, dict) and "failed" not in r]
    yaws = [r["yaw_gap_deg"] for r in ok]
    return {
        "n": d["n"], "ok": d["ok"],
        "occluded": sum(r["mode"] == "occluded" for r in ok),
        "out_frame": sum(r["mode"] == "out" for r in ok),
        "yaw_med": _pct(yaws, 0.5),
        "margin_min": min((r["kp_margin_px"] for r in ok), default=None),
        # 只给文件名：evidence 存的是 Windows 路径，直接拼 URL 会 404
        "images": [Path(r["evidence"]).name for r in ok],
    }


def recent_logs(limit: int = 3):
    logs = sorted(ROOT.glob("outputs/*.log"),
                  key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    out = []
    for p in logs:
        try:
            raw = p.read_bytes()
            encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
            lines = raw.decode(encoding, errors="replace").splitlines()
        except OSError:
            continue
        out.append({"name": p.name, "lines": len(lines),
                    "age_s": int(time.time() - p.stat().st_mtime),
                    "tail": lines[-5:]})
    return out


def build_state():
    exps = []
    for p in sorted(EVAL.glob("*.json"), key=lambda p: p.stat().st_mtime,
                    reverse=True):
        s = summarize(p)
        if s:
            exps.append(s)
        if len(exps) >= 6:
            break
    return {"now": time.strftime("%H:%M:%S"), "experiments": exps,
            "blind": blind_summary(), "logs": recent_logs()}


PAGE = """<!doctype html><meta charset="utf-8"><title>Touch 实验监控</title>
<style>
:root{--bg:#0f1419;--card:#182027;--line:#26323c;--tx:#d7e2ea;--dim:#7d8f9c;
--ok:#2ec27e;--bad:#f66151;--acc:#62a0ea}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);
font:14px/1.5 "Segoe UI",system-ui,sans-serif;padding:20px}
h1{font-size:18px;margin:0 0 4px}h1 span{color:var(--acc)}
.sub{color:var(--dim);font-size:12px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card h3{margin:0 0 8px;font-size:13px;color:var(--acc);font-weight:600}
.kv{display:flex;justify-content:space-between;font-size:12px;padding:2px 0}
.kv b{font-weight:600}
.big{font-size:22px;font-weight:700}
.ok{color:var(--ok)}.bad{color:var(--bad)}.dim{color:var(--dim)}
section{margin-top:18px}h2{font-size:14px;margin:0 0 10px;color:var(--dim);
font-weight:600;text-transform:uppercase;letter-spacing:.06em}
canvas{width:100%;max-width:860px;background:var(--card);border:1px solid var(--line);
border-radius:10px}
.thumb{width:132px;border:1px solid var(--line);border-radius:8px;cursor:zoom-in;
display:block}
.views{display:flex;flex-wrap:wrap;gap:8px}
textarea,pre{width:100%;background:var(--card);color:var(--tx);border:1px solid var(--line);
border-radius:8px;font:12px/1.5 Consolas,monospace;padding:10px;white-space:pre-wrap}
</style>
<h1>Touch <span>实验监控</span></h1>
<div class="sub">每 3 秒自动刷新 · <span id="now">-</span> · 数据源 outputs/eval/*.json</div>
<div class="grid" id="cards"></div>
<section><h2>误差分布（x = 取点误差 mm，对数轴 · y = 重投影 rms px · 绿=抓到 红=失败）</h2>
<canvas id="sc" width="860" height="320"></canvas></section>
<section><h2>失明视角验证（L1 记忆实验）</h2><div id="blind"></div>
<div class="views" id="views"></div></section>
<section><h2>最近运行日志</h2><div id="logs"></div></section>
<script>
const fmt=(v,d=2)=>v==null?"-":Number(v).toFixed(d);
const pct=v=>v==null?'-':(v*100).toFixed(1)+'%';
const esc=v=>String(v??'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
let current=null;
function cards(exps){
  const el=document.getElementById("cards");
  if(current===null||current>=exps.length)current=exps.length?0:null;
  el.innerHTML=exps.map((e,i)=>{
    // 有分组（conds）时不显示汇总百分比：记忆闭环里「该抓空」的条件混进来会误导，
    // 而且 0% 在这里是好事，不能上红色。改成一行一个条件、不加语义色。
    const body=e.conds?e.conds.map(c=>
      `<div class="kv"><span>${esc(c.cond)}</span><b>执行后 ${pct(c.grasp_rate)}</b></div>
       <div class="dim">${fmt(c.err_med,1)}mm · 全任务 ${pct(c.end_to_end_rate)} · 执行/计划 ${c.n_attempted??'-'}/${c.n_planned??'-'}</div>
       <div class="dim">${esc(c.status_text)}</div>`).join("")
      :`<div class="big">${pct(e.grasp_rate)}</div>
        <div class="dim">全任务 ${pct(e.end_to_end_rate)} · 执行/计划 ${e.n_attempted??'-'}/${e.n_planned??'-'}</div>`;
    const cap=(e.conds?"分条件":(e.n_tagged?"执行后抓到率":"样本"))+` · 记录 ${e.n} 条 · ${e.mtime}`
      +(e.legacy?' · 历史口径，完整分母未确认':'');
    return `<div class="card" onclick="pick(${i})"
    style="${i===current?'border-color:var(--acc)':'cursor:pointer'}">
    <h3>${esc(e.label)}</h3>${body}
    <div class="dim" style="font-size:11px">${esc(cap)}</div>
    <div class="dim">${esc(e.status_text)}</div>
    <div class="kv"><span>误差中位</span><b>${fmt(e.err_med)} mm</b></div>
    <div class="kv"><span>p95 / 最大</span><b>${fmt(e.err_p95)} / ${fmt(e.err_max)}</b></div>
  </div>`;}).join("");
  draw(exps[current]);
  window._exps=exps;
}
function pick(i){current=i;cards(window._exps);}
function draw(e){
  const c=document.getElementById("sc"),g=c.getContext("2d"),W=c.width,H=c.height;
  g.clearRect(0,0,W,H);const L=52,R=12,T=12,B=34;
  const pts=(e?.pts||[]).filter(p=>Number.isFinite(p[0])&&Number.isFinite(p[1])&&p[0]>0);
  if(!pts.length){g.fillStyle='#7d8f9c';g.font='14px sans-serif';
    g.fillText('暂无可绘制数据：需要有限误差和重投影残差（零误差不显示在对数轴）',L,48);return;}
  const xs=pts.map(p=>Math.log10(p[0])),ys=pts.map(p=>p[1]);
  const x0=Math.min(...xs),x1=Math.max(...xs),y0=0,y1=Math.max(...ys)*1.05;
  const X=v=>L+(v-x0)/(x1-x0||1)*(W-L-R),Y=v=>H-B-(v-y0)/(y1-y0||1)*(H-T-B);
  g.strokeStyle="#26323c";g.fillStyle="#7d8f9c";g.font="11px Consolas";
  for(let t=Math.floor(x0);t<=Math.ceil(x1);t++){const x=X(t);
    g.beginPath();g.moveTo(x,T);g.lineTo(x,H-B);g.stroke();
    g.fillText(Math.pow(10,t)<1?Math.pow(10,t).toFixed(1):Math.pow(10,t).toFixed(0),x-6,H-14);}
  for(let i=0;i<=4;i++){const y=Y(y1*i/4);
    g.beginPath();g.moveTo(L,y);g.lineTo(W-R,y);g.stroke();
    g.fillText((y1*i/4).toFixed(1),8,y+4);}
  pts.forEach(p=>{const ok=p[2]===null?"#62a0ea":(p[2]?"#2ec27e":"#f66151");
    g.fillStyle=ok;g.globalAlpha=.75;g.beginPath();
    g.arc(X(Math.log10(p[0])),Y(p[1]),2.4,0,7);g.fill();});
  g.globalAlpha=1;g.fillStyle="#d7e2ea";
  g.fillText(e.label,W-260,20);
}
function blind(b){
  const el=document.getElementById("blind");
  if(!b){el.innerHTML='<div class="dim">尚无 blind_view.json</div>';return;}
  el.innerHTML=`<div class="card" style="max-width:520px">
   <h3>找不到方块的 B 相机：${b.ok}/${b.n}</h3>
   <div class="kv"><span>失明机制</span><b>被臂挡 ${b.occluded} · 出框 ${b.out_frame}</b></div>
   <div class="kv"><span>B-A 视角差 中位</span><b>${fmt(b.yaw_med,1)}°</b></div>
   <div class="kv"><span>臂关键点最小边距</span><b>${fmt(b.margin_min,1)} px</b></div></div>`;
  document.getElementById("views").innerHTML=b.images.map(p=>
    `<a href="/img/${encodeURIComponent(p)}" target="_blank">
     <img class="thumb" src="/img/${encodeURIComponent(p)}"></a>`).join("");
}
function logs(ls){
  document.getElementById("logs").innerHTML=ls.map(l=>`<pre>${l.name}
(${l.lines} 行 · ${l.age_s}s 前更新)
${(l.tail||[]).join("\\n").replace(/</g,"&lt;")}</pre>`).join("");
}
async function tick(){
  try{const r=await fetch("/api/state");const s=await r.json();
    document.getElementById("now").textContent=s.now;
    cards(s.experiments);blind(s.blind);logs(s.logs);
  }catch(e){document.getElementById("now").textContent="连接失败";}
}
tick();setInterval(tick,3000);
</script>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = unquote(self.path.split("?")[0])
        if path == "/":
            body, ctype = PAGE.encode("utf-8"), "text/html; charset=utf-8"
        elif path == "/api/state":
            body = json.dumps(build_state(), ensure_ascii=False).encode("utf-8")
            ctype = "application/json; charset=utf-8"
        elif path.startswith("/img/"):
            f = (VIEWS / path[len("/img/"):]).resolve()
            if not f.is_relative_to(VIEWS.resolve()) or f.suffix != ".png" or not f.is_file():
                self.send_error(404)
                return
            body, ctype = f.read_bytes(), "image/png"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="实验监控台（零依赖）")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"monitor: http://127.0.0.1:{a.port}/  (Ctrl+C 停止)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
