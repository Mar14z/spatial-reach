"""监控数据和图片访问边界；不启动服务，不需要浏览器。"""
import io
import json

import pytest

from scripts import monitor_server as monitor


@pytest.mark.parametrize("route", ["../secret.png", "..%5Csecret.png", "%2e%2e/secret.png"])
def test_image_route_blocks_parent(tmp_path, monkeypatch, route):
    views = tmp_path / "views"
    views.mkdir()
    (tmp_path / "secret.png").write_bytes(b"private")
    monkeypatch.setattr(monitor, "VIEWS", views)
    handler = object.__new__(monitor.Handler)
    handler.path = "/img/" + route
    handler.wfile = io.BytesIO()
    errors = []
    handler.send_error = errors.append
    handler.send_response = lambda *a: None
    handler.send_header = lambda *a: None
    handler.end_headers = lambda: None
    handler.do_GET()
    assert errors == [404]
    assert handler.wfile.getvalue() == b""


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16"])
def test_log_encoding(tmp_path, monkeypatch, encoding):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "test.log").write_text("中文进度\n已完成", encoding=encoding)
    monkeypatch.setattr(monitor, "ROOT", tmp_path)
    assert monitor.recent_logs()[0]["tail"] == ["中文进度", "已完成"]


def test_all_rejected_experiment_is_visible(tmp_path):
    path = tmp_path / "failed.json"
    path.write_text(json.dumps({"protocol_version": 3, "rows": [
        {"status": "calibration_rejected", "err_mm": None},
        {"status": "missing_frame", "err_mm": None}]}), encoding="utf-8")
    result = monitor.summarize(path)
    assert result is not None and result["n"] == 2
    assert result["err_med"] is None


def test_grouped_counts_and_legacy_are_distinct(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"protocol_version": 3, "cond_summary": {"mem": {
        "n_planned": 3, "n_attempted": 2, "attempt_success_rate": 0.5,
        "end_to_end_success_rate": 1 / 3, "status_counts": {"calibration_rejected": 1}}},
        "rows": [{"cond": "mem", "err_mm": None, "status": "calibration_rejected"}]}), encoding="utf-8")
    result = monitor.summarize(path)
    assert not result["legacy"]
    assert result["conds"][0]["grasp_rate"] == 0.5
    assert result["conds"][0]["n_planned"] == 3
    assert result["conds"][0]["end_to_end_rate"] == pytest.approx(1 / 3)
    assert "标定拒识" in result["conds"][0]["status_text"]


def test_frontend_empty_data_and_selection():
    import shutil
    import subprocess
    node = shutil.which("node")
    if node is None:
        pytest.skip("没有 Node，前端脚本检查需单独运行")
    js = monitor.PAGE.split("<script>")[1].split("</script>")[0]
    js = js.replace("tick();setInterval(tick,3000);", "")
    harness = """
const assert=require('node:assert/strict');
const elements={};
const g=new Proxy({}, {get:()=>()=>{}});
global.window={};
global.document={getElementById:id=>elements[id]??=(
  {innerHTML:'',width:860,height:320,getContext:()=>g})};
"""
    checks = """
cards([]);
current=9;
cards([{label:'<b>测试</b>',pts:[],n:0,conds:null,mtime:'',n_tagged:0}]);
assert.equal(current,0);
assert.ok(elements.cards.innerHTML.includes('&lt;b&gt;'));
assert.ok(elements.cards.innerHTML.includes('执行/计划'));
draw({pts:[[null,1,null],[0,0,true]],label:'空图'});
cards([]);
assert.equal(current,null);
"""
    result = subprocess.run([node], input=harness + js + checks, text=True,
                            capture_output=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
