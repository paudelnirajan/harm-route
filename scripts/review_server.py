"""Browser UI for reviewing machine translations (ok / fix / drop). Same decisions file as review_translations.py.

    uv run python scripts/review_server.py --config configs/experiments/e01_translate.yaml --lang npi_Deva
    # then open the printed http://127.0.0.1:8765 link

Why a browser: terminals can't shape complex scripts (Devanagari conjuncts, vowel signs); browsers can.
Standard library only (http.server). Serves PROMPTS INCLUDING HARMFUL ONES:
  - default host 127.0.0.1 = only this computer can open it
  - any other host (sharing on a network) REQUIRES --token; every request must carry it (?t=...)
Decisions are appended to {out_dir}/review/{system}/{lang}__{reviewer}.jsonl (newest decision per key wins).
"""

import argparse
import json
import re
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

from harm_route.lang_data import load_reviews
from harm_route.text_stats import dominant_script

ROOT = Path(__file__).resolve().parents[1]
LANGS = yaml.safe_load((ROOT / "configs" / "languages.yaml").read_text())
REVIEWER_RE = re.compile(r"^[a-z0-9_-]{1,32}$")  # also keeps the name safe to use in a file path
LOCK = threading.Lock()  # several reviewers may save at once


def load_items(cfg, out_dir: Path, system: str, lang: str) -> list[dict]:
    eng = json.loads((ROOT / cfg["splits"].format(model=cfg["model"])).read_text())
    mt = json.loads((out_dir / "mt" / system / f"{lang}.json").read_text())
    qc = json.loads((out_dir / "mt" / system / f"{lang}_qc.json").read_text())
    items = [{"key": f"{s}|{i}", "chrf": round(qc[s][i]["chrf"], 1), "english": eng[s][i], "mt": mt[s][i],
              "back": qc[s][i]["back"]} for s in cfg["sets"] for i in range(len(mt[s]))]
    return sorted(items, key=lambda it: it["chrf"])  # likely problems first


def make_handler(items, out_dir: Path, system: str, lang: str, token: str | None):
    by_key = {it["key"]: it for it in items}
    script = LANGS[lang]["script"]
    rev_dir = out_dir / "review" / system

    def reviewer_rows(name: str) -> dict:
        """This reviewer's latest decision per key (only their own file)."""
        p = rev_dir / f"{lang}__{name}.jsonl"
        rows = {}
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    rows[r["key"]] = r
        return rows

    class H(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # quiet: request lines would be noise
            pass

        def _send(self, code: int, body, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _auth(self, q) -> bool:
            if token and q.get("t", [""])[0] != token:
                self._send(403, {"error": "missing or wrong token"})
                return False
            return True

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if not self._auth(q):
                return
            if u.path == "/":
                return self._send(200, PAGE.replace("__LANG__", lang).replace("__SYSTEM__", system), "text/html")
            if u.path == "/api/items":
                name = q.get("reviewer", [""])[0]
                if not REVIEWER_RE.match(name):
                    return self._send(400, {"error": "reviewer: lowercase letters, digits, _ or -"})
                return self._send(200, {"items": items, "done": reviewer_rows(name)})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            u = urlparse(self.path)
            if not self._auth(parse_qs(u.query)):
                return
            if u.path != "/api/decision":
                return self._send(404, {"error": "not found"})
            try:
                r = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            except (ValueError, json.JSONDecodeError):
                return self._send(400, {"error": "bad json"})
            name, key, dec = r.get("reviewer", ""), r.get("key", ""), r.get("decision", "")
            text, note = (r.get("text") or "").strip(), (r.get("note") or "").strip()
            if not REVIEWER_RE.match(name) or key not in by_key or dec not in ("ok", "fix", "drop"):
                return self._send(400, {"error": "bad reviewer / key / decision"})
            row = {"key": key, "decision": dec, "note": note}
            if dec == "fix":
                if not text or text == by_key[key]["mt"].strip():
                    return self._send(400, {"error": "fix needs an edited text (unchanged = use OK)"})
                sc, _ = dominant_script(text)
                if sc != script:
                    return self._send(400, {"error": f"edit is mostly {sc}, expected {script}"})
                row["text"] = text
            with LOCK:
                rev_dir.mkdir(parents=True, exist_ok=True)
                with (rev_dir / f"{lang}__{name}.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._send(200, {"ok": True})

    return H


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Translation review</title>
<style>
:root { --bg:#fafaf8; --fg:#1d1d1b; --muted:#6b6b66; --card:#fff; --line:#deded8; --accent:#2f6fde;
        --ok:#2e7d32; --fix:#b26a00; --drop:#c62828; }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --fg:#ececea; --muted:#9a9a94; --card:#1f1f1e;
        --line:#353533; --accent:#7aa7ff; --ok:#66bb6a; --fix:#ffb74d; --drop:#ef5350; } }
* { box-sizing:border-box } body { margin:0; background:var(--bg); color:var(--fg);
  font:16px/1.55 system-ui, -apple-system, "Kohinoor Devanagari", "Noto Sans Devanagari", sans-serif; }
main { max-width:860px; margin:0 auto; padding:16px }
.bar { display:flex; gap:12px; align-items:center; flex-wrap:wrap; color:var(--muted); font-size:14px }
.bar b { color:var(--fg) } progress { flex:1; min-width:120px }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin:12px 0 }
.lbl { font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin-bottom:4px }
.txt { white-space:pre-wrap; overflow-wrap:anywhere } .mt { font-size:22px; line-height:1.8 }
textarea { width:100%; min-height:110px; font:inherit; font-size:22px; line-height:1.8; padding:8px;
  background:var(--bg); color:var(--fg); border:1px solid var(--line); border-radius:8px }
input[type=text] { width:100%; font:inherit; padding:6px 8px; background:var(--bg); color:var(--fg);
  border:1px solid var(--line); border-radius:8px }
.btns { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px } button { font:inherit; padding:8px 14px;
  border-radius:8px; border:1px solid var(--line); background:var(--card); color:var(--fg); cursor:pointer }
button.ok { border-color:var(--ok); color:var(--ok) } button.fix { border-color:var(--fix); color:var(--fix) }
button.drop { border-color:var(--drop); color:var(--drop) } button:disabled { opacity:.4; cursor:default }
.status { min-height:1.4em; font-size:14px } .err { color:var(--drop) } .tag { font-weight:600 }
.hint { color:var(--muted); font-size:13px }
</style></head><body><main>
<div id="login" class="card">
  <div class="lbl">Reviewer name</div>
  <input id="who" type="text" placeholder="e.g. nirajan (lowercase, no spaces)">
  <div class="btns"><button id="start">Start reviewing __LANG__ (__SYSTEM__)</button></div>
  <p class="hint">Check: same request? nothing added or dropped? a harmful request stays exactly as harmful
  (not softened)? natural wording a native speaker would use?</p>
</div>
<div id="app" hidden>
  <div class="bar"><span><b id="pos"></b></span><progress id="prog"></progress><span id="cnt"></span></div>
  <div class="bar"><span id="key"></span><span id="chrf"></span><span id="prev" class="tag"></span></div>
  <div class="card"><div class="lbl">English (original)</div><div id="en" class="txt"></div></div>
  <div class="card"><div class="lbl">Machine translation (edit below if needed)</div>
    <textarea id="edit" spellcheck="false"></textarea>
    <div class="lbl" style="margin-top:10px">Back-translation to English (machine, for reference)</div>
    <div id="back" class="txt hint"></div></div>
  <div class="card"><div class="lbl">Note (optional)</div><input id="note" type="text"
    placeholder="e.g. partial, softened, unnatural word"></div>
  <div class="btns">
    <button id="bprev">&larr; Back</button>
    <button class="ok" id="bok">OK as is</button>
    <button class="fix" id="bfix" disabled>Save my edit</button>
    <button class="drop" id="bdrop">Drop</button>
    <button id="bnext">Skip &rarr;</button>
    <button id="bjump">Next unreviewed</button>
  </div>
  <div id="status" class="status"></div>
  <p class="hint">Shortcuts when not typing: o = OK, d = drop, &larr;/&rarr; = back/skip.
  Edits are saved only with "Save my edit". You can go back and change any decision (newest wins).</p>
</div>
</main><script>
const T = new URLSearchParams(location.search).get("t") || "";
const q = (p) => p + (p.includes("?") ? "&" : "?") + "t=" + encodeURIComponent(T);
let who = "", items = [], done = {}, i = 0;
const $ = (id) => document.getElementById(id);
try { $("who").value = localStorage.getItem("reviewer") || ""; } catch (e) {}

async function start() {
  who = $("who").value.trim().toLowerCase();
  const r = await fetch(q("/api/items?reviewer=" + encodeURIComponent(who)));
  const j = await r.json();
  if (!r.ok) { alert(j.error); return; }
  try { localStorage.setItem("reviewer", who); } catch (e) {}
  items = j.items; done = j.done; $("login").hidden = true; $("app").hidden = false;
  i = Math.max(0, items.findIndex(it => !done[it.key])); if (i < 0) i = 0; show();
}
function show() {
  const it = items[i], d = done[it.key], n = Object.keys(done).length;
  $("pos").textContent = `${i + 1} / ${items.length}`;
  $("prog").max = items.length; $("prog").value = n; $("cnt").textContent = `${n} reviewed`;
  $("key").textContent = it.key; $("chrf").textContent = `back-translation chrF ${it.chrf}`;
  $("prev").textContent = d ? `your decision: ${d.decision}` : "not reviewed yet";
  $("prev").style.color = d ? `var(--${d.decision})` : "var(--muted)";
  $("en").textContent = it.english;              // textContent, never innerHTML: prompts are untrusted text
  $("edit").value = d && d.decision === "fix" ? d.text : it.mt;
  $("back").textContent = it.back || "";
  $("note").value = d ? (d.note || "") : ""; $("status").textContent = ""; dirty();
}
function dirty() { $("bfix").disabled = $("edit").value.trim() === items[i].mt.trim(); }
async function save(decision) {
  const it = items[i], body = { reviewer: who, key: it.key, decision, note: $("note").value,
    text: decision === "fix" ? $("edit").value : undefined };
  const r = await fetch(q("/api/decision"), { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) { $("status").innerHTML = ""; const s = document.createElement("span");
    s.className = "err"; s.textContent = j.error; $("status").appendChild(s); return; }
  done[it.key] = { decision, note: body.note, text: body.text };
  if (i < items.length - 1) { i++; show(); } else { show(); $("status").textContent = "Last item saved."; }
}
$("start").onclick = start; $("who").onkeydown = (e) => { if (e.key === "Enter") start(); };
$("edit").oninput = dirty;
$("bok").onclick = () => save("ok"); $("bfix").onclick = () => save("fix"); $("bdrop").onclick = () => save("drop");
$("bprev").onclick = () => { if (i > 0) { i--; show(); } };
$("bnext").onclick = () => { if (i < items.length - 1) { i++; show(); } };
$("bjump").onclick = () => { const k = items.findIndex(it => !done[it.key]);
  if (k >= 0) { i = k; show(); } else $("status").textContent = "Everything is reviewed."; };
document.onkeydown = (e) => {
  if ($("app").hidden || ["TEXTAREA", "INPUT"].includes(document.activeElement.tagName)) return;
  if (e.key === "o") save("ok"); else if (e.key === "d") save("drop");
  else if (e.key === "ArrowLeft") $("bprev").click(); else if (e.key === "ArrowRight") $("bnext").click();
};
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--system", help="MT system to review (default: primary in the config)")
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = this computer only")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--token", help="required when --host is not 127.0.0.1; use 'auto' to generate one")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    system = args.system or cfg["primary"]
    if LANGS[args.lang]["source"] != "mt":
        raise SystemExit(f"{args.lang} is {LANGS[args.lang]['source']!r}; review its source language instead")
    token = secrets.token_urlsafe(16) if args.token == "auto" else args.token
    if args.host != "127.0.0.1" and not token:
        raise SystemExit("serving beyond this computer shows harmful prompts: pass --token (or --token auto)")
    out_dir = ROOT / cfg["out_dir"].format(model=cfg["model"])
    items = load_items(cfg, out_dir, system, args.lang)
    n_done = len(load_reviews(out_dir, system, args.lang))
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(items, out_dir, system, args.lang, token))
    link = f"http://{'127.0.0.1' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}/"
    print(f"{args.lang} ({system}): {len(items)} items, {n_done} keys reviewed by anyone so far")
    print(f"open: {link}{'?t=' + token if token else ''}\nCtrl-C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
