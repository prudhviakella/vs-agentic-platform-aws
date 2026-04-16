"""
app.py — VS AgentCore UI
=========================
FastAPI app serving a beautiful single-page clinical research UI.
Proxies SSE streams to the Platform API to avoid CORS.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8501
"""

import os
import base64
import logging

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, Response

log = logging.getLogger(__name__)

API_URL = os.environ.get("AGENT_API_URL", "http://localhost:8000")
API_KEY = os.environ.get("AGENT_API_KEY", "local-dev-key")
DOMAIN  = os.environ.get("AGENT_DOMAIN",  "pharma")
AGENT   = "clinical-trial"

HEADERS = {
    "X-API-Key":    API_KEY,
    "Content-Type": "application/json",
    "Accept":       "text/event-stream",
}

app = FastAPI()


@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    return await _proxy_sse(f"{API_URL}/api/v1/{AGENT}/chat", body)


@app.post("/resume")
async def resume(request: Request):
    body = await request.json()
    return await _proxy_sse(f"{API_URL}/api/v1/{AGENT}/resume", body)


async def _proxy_sse(url: str, payload: dict):
    async def generate():
        async with httpx.AsyncClient(timeout=180) as client:
            async with client.stream("POST", url, headers=HEADERS, json=payload) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        yield line + "\n"
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


# Minimal favicon to silence 404 in browser console
_FAVICON = base64.b64decode(
    "AAABAAEAEBAQAAEABAAoAQAAFgAAACgAAAAQAAAAIAAAAAEABAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAA////AAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
)

@app.get("/favicon.ico")
async def favicon():
    return Response(content=_FAVICON, media_type="image/x-icon")


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Clinical Trial Research Agent</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --font:'AppleGothic','Apple Gothic','Gill Sans MT','Gill Sans','Century Gothic','Trebuchet MS',ui-rounded,sans-serif;
  --bg:#07111e;--surface:#0d1928;--surface-2:#112030;
  --border:#1a2e44;--border-2:#1f3550;
  --accent:#00c2ff;--accent-dim:rgba(0,194,255,0.10);--accent-glow:rgba(0,194,255,0.05);
  --green:#00e5a0;--red:#ff4d6a;--amber:#ffb340;
  --text:#cddff0;--text-2:#6b8fae;--text-3:#2e4a63;
  --r:8px;--r-lg:14px;
}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:var(--font);-webkit-font-smoothing:antialiased}
.shell{display:grid;grid-template-rows:58px 1fr auto;height:100vh;max-width:860px;margin:0 auto}

/* header */
header{display:flex;align-items:center;justify-content:space-between;padding:0 28px;border-bottom:1px solid var(--border);background:var(--surface)}
.brand{display:flex;align-items:center;gap:11px}
.brand-icon{width:30px;height:30px;background:linear-gradient(135deg,#00c2ff,#005fff);border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:15px}
.brand-name{font-size:15px;font-weight:500;letter-spacing:.01em}
.header-right{display:flex;gap:10px;align-items:center}
.pill{font-size:11px;font-family:var(--font);color:var(--text-2);background:var(--surface-2);border:1px solid var(--border);padding:3px 10px;border-radius:20px;letter-spacing:.02em}
.pill.live{color:var(--green);border-color:rgba(0,229,160,.25);background:rgba(0,229,160,.06)}

/* messages */
.messages{overflow-y:auto;padding:28px 28px 8px;display:flex;flex-direction:column;gap:20px;scroll-behavior:smooth}
.messages::-webkit-scrollbar{width:3px}
.messages::-webkit-scrollbar-thumb{background:var(--border-2);border-radius:2px}

/* welcome */
.welcome{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:28px;padding:52px 20px;text-align:center}
.welcome-eyebrow{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);font-weight:500}
.welcome-title{font-size:36px;font-weight:500;line-height:1.15;letter-spacing:-.01em}
.welcome-title span{color:var(--accent);font-style:italic}
.welcome-sub{font-size:14px;color:var(--text-2);max-width:420px;line-height:1.7;font-weight:300}
.starters-label{font-size:11px;color:var(--text-3);letter-spacing:.06em;text-transform:uppercase}
.starters{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;max-width:620px}
.starter{background:var(--surface);border:1px solid var(--border);color:var(--text-2);font-family:var(--font);font-size:13px;font-weight:300;padding:9px 16px;border-radius:22px;cursor:pointer;transition:all .16s;line-height:1.4}
.starter:hover{border-color:var(--accent);color:var(--text);background:var(--accent-glow);transform:translateY(-1px)}

/* messages */
.msg{display:flex;gap:12px;animation:fadeUp .18s ease}
.msg.user{flex-direction:row-reverse}
.av{width:30px;height:30px;border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:500}
.av.agent{background:rgba(0,194,255,.12);border:1px solid rgba(0,194,255,.25);color:var(--accent)}
.av.user{background:var(--surface-2);border:1px solid var(--border);color:var(--text-2)}
.bubble{max-width:78%;padding:13px 18px;border-radius:var(--r-lg);font-size:14px;line-height:1.75;font-weight:300}
.msg.user .bubble{background:var(--surface-2);border:1px solid var(--border);color:var(--text);border-radius:var(--r-lg) 4px var(--r-lg) var(--r-lg)}
.msg.agent .bubble{background:var(--surface);border:1px solid var(--border);color:var(--text);border-radius:4px var(--r-lg) var(--r-lg) var(--r-lg)}

/* markdown */
.bubble h1,.bubble h2,.bubble h3{font-weight:500;margin:16px 0 8px;letter-spacing:-.01em}
.bubble h1{font-size:18px}.bubble h2{font-size:16px}.bubble h3{font-size:14px}
.bubble p{margin:8px 0}.bubble ul,.bubble ol{padding-left:20px;margin:8px 0}.bubble li{margin:5px 0}
.bubble strong{font-weight:500}.bubble em{color:var(--text-2)}
.bubble code{font-family:'Menlo','Courier New',monospace;font-size:12px;background:var(--surface-2);border:1px solid var(--border);padding:1px 6px;border-radius:4px;color:var(--accent)}
.bubble pre{background:var(--bg);border:1px solid var(--border);border-radius:var(--r);padding:12px;overflow-x:auto;margin:10px 0}
.bubble pre code{background:none;border:none;padding:0;font-size:12px}
.bubble blockquote{border-left:3px solid var(--accent);padding-left:14px;margin:8px 0;color:var(--text-2)}
.bubble a{color:var(--accent);text-decoration:none}
.bubble hr{border:none;border-top:1px solid var(--border);margin:14px 0}
.bubble table{border-collapse:collapse;width:100%;margin:10px 0;font-size:13px}
.bubble th,.bubble td{border:1px solid var(--border);padding:7px 12px;text-align:left}
.bubble th{background:var(--surface-2);font-weight:500}
.meta-footer{font-size:11px;color:var(--text-3);margin-top:12px;padding-top:10px;border-top:1px solid var(--border);display:flex;gap:14px;font-weight:300;letter-spacing:.01em}

/* tool step */
.tool-step{display:flex;align-items:center;gap:10px;padding:9px 14px;background:var(--surface);border:1px solid var(--border);border-radius:var(--r);font-size:12.5px;font-weight:300;color:var(--text-2);max-width:320px;margin-left:42px;animation:fadeUp .18s ease;letter-spacing:.01em}
.tool-step.done{color:var(--green);border-color:rgba(0,229,160,.2)}
.tool-step.thinking{border-style:dashed;border-color:var(--border-2)}
.spin{width:13px;height:13px;border:1.5px solid var(--border-2);border-top-color:var(--accent);border-radius:50%;animation:spin .65s linear infinite;flex-shrink:0}
.spin.pulse{border-color:transparent;border-top-color:var(--text-3);animation:spin 1.2s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.cur{display:inline-block;width:2px;height:13px;background:var(--accent);margin-left:1px;vertical-align:middle;animation:blink .9s step-end infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}

/* hitl */
.hitl{background:var(--surface);border:1px solid var(--border);border-top:2px solid var(--accent);border-radius:var(--r-lg);padding:20px;max-width:500px;margin-left:42px;animation:fadeUp .18s ease}
.hitl-tag{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--accent);margin-bottom:10px;display:flex;align-items:center;gap:6px;font-weight:500}
.hitl-tag::before{content:'';width:5px;height:5px;background:var(--accent);border-radius:50%}
.hitl-q{font-size:14px;font-weight:400;color:var(--text);margin-bottom:16px;line-height:1.55}
.hitl-opts{display:flex;flex-direction:column;gap:7px;margin-bottom:12px}
.hitl-opt{display:flex;align-items:center;gap:10px;padding:10px 14px;background:var(--surface-2);border:1px solid var(--border);border-radius:var(--r);cursor:pointer;font-size:13px;font-weight:300;color:var(--text-2);transition:all .14s;text-align:left;width:100%;font-family:var(--font)}
.hitl-opt:hover{border-color:var(--accent);color:var(--text);background:var(--accent-dim);transform:translateX(2px)}
.hitl-opt:hover .num{background:var(--accent);color:var(--bg)}
.hitl-opt.picked{border-color:var(--green);color:var(--green);background:rgba(0,229,160,.07);pointer-events:none}
.num{font-size:11px;width:20px;height:20px;background:var(--border);border-radius:4px;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all .14s;font-family:'Menlo',monospace}
.hitl-hint{font-size:12px;color:var(--text-3);font-style:italic;font-weight:300}

/* input */
.bar{padding:16px 28px;border-top:1px solid var(--border);background:var(--surface);display:flex;gap:10px;align-items:flex-end}
.inp-wrap{flex:1;background:var(--bg);border:1px solid var(--border);border-radius:var(--r-lg);display:flex;align-items:flex-end;padding:11px 16px;transition:border-color .14s}
.inp-wrap:focus-within{border-color:var(--accent)}
textarea{flex:1;background:none;border:none;outline:none;color:var(--text);font-family:var(--font);font-size:14px;font-weight:300;resize:none;line-height:1.5;max-height:120px;min-height:24px}
textarea::placeholder{color:var(--text-3)}
.send{width:36px;height:36px;background:var(--accent);border:none;border-radius:var(--r);cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all .14s;color:var(--bg)}
.send:hover{background:#33ccff;transform:scale(1.06)}
.send:disabled{background:var(--border);cursor:not-allowed;transform:none}
.send svg{width:15px;height:15px}
.new-btn{height:36px;background:var(--surface-2);border:1px solid var(--border);border-radius:var(--r);cursor:pointer;color:var(--text-2);font-family:var(--font);font-size:12px;font-weight:300;padding:0 14px;transition:all .14s;white-space:nowrap}
.new-btn:hover{border-color:var(--border-2);color:var(--text)}
.err{background:rgba(255,77,106,.07);border:1px solid rgba(255,77,106,.25);border-radius:var(--r);padding:10px 14px;font-size:13px;font-weight:300;color:var(--red);margin-left:42px;max-width:460px;animation:fadeUp .18s ease}
@keyframes fadeUp{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
</style>
</head>
<body>
<div class="shell">
  <header>
    <div class="brand">
      <div class="brand-icon">⚕</div>
      <span class="brand-name">Clinical Trial Research Agent</span>
    </div>
    <div class="header-right">
      <span class="pill" id="sid">session: —</span>
      <span class="pill live">● pharma</span>
    </div>
  </header>

  <div class="messages" id="msgs">
    <div class="welcome" id="welcome">
      <div class="welcome-eyebrow">Vidya Sankalp · AgentCore Platform</div>
      <div class="welcome-title">Clinical Trial<br><span>Intelligence</span></div>
      <p class="welcome-sub">Search 5,772 trial documents and a live biomedical knowledge graph. Powered by Pinecone, Neo4j, and AWS AgentCore.</p>
      <div class="starters-label">Try one of these</div>
      <div class="starters" id="starters"></div>
    </div>
  </div>

  <div class="bar">
    <div class="inp-wrap">
      <textarea id="inp" placeholder="Ask about a trial, drug, or clinical outcome…" rows="1"></textarea>
    </div>
    <button class="send" id="sbtn" onclick="send()">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <line x1="22" y1="2" x2="11" y2="13"/>
        <polygon points="22 2 15 22 11 13 2 9 22 2"/>
      </svg>
    </button>
    <button class="new-btn" onclick="newChat()">+ New</button>
  </div>
</div>

<script>
const STARTERS = [
  "What are the Phase 3 efficacy results for Pfizer BNT162b2?",
  "Tell me about the COVID vaccine trial",
  "Is mRNA-1273 safe for patients with heart failure?",
  "Which trials study remdesivir for COVID-19?",
  "What are the primary outcomes for the Moderna vaccine trial NCT04470427?",
  "Who sponsors the Hepatitis B TAF trial?",
];

// UUID fallback — crypto.randomUUID() only works on HTTPS (secure context)
// The ALB runs on HTTP so we need a Math.random() fallback
function uuid() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0;
    return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
  });
}

// inp declared first so send() can reference it without hoisting issues
const inp = document.getElementById('inp');

let threadId    = uuid();
let interrupted = false;
let streaming   = false;

document.getElementById('sid').textContent = 'session: ' + threadId.slice(0, 8);

// Starter chips
const startersEl = document.getElementById('starters');
STARTERS.forEach(q => {
  const b = document.createElement('button');
  b.className = 'starter';
  b.textContent = q;
  b.onclick = () => submit(q);
  startersEl.appendChild(b);
});

// Auto-resize textarea
inp.addEventListener('input', () => {
  inp.style.height = 'auto';
  inp.style.height = Math.min(inp.scrollHeight, 120) + 'px';
});
inp.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

// ── Send ──────────────────────────────────────────────────────────────────
function send() {
  const t = inp.value.trim();
  if (!t || streaming) return;
  inp.value = ''; inp.style.height = 'auto';
  submit(t);
}

async function submit(text) {
  hideWelcome();
  addUser(text);
  if (interrupted) await doResume(text);
  else await doChat(text);
}

async function doChat(msg) {
  await sse('/chat', { message: msg, thread_id: threadId, domain: 'pharma' });
}
async function doResume(ans) {
  interrupted = false;
  await sse('/resume', { thread_id: threadId, user_answer: ans, domain: 'pharma' });
}

// ── SSE stream ────────────────────────────────────────────────────────────
async function sse(url, payload) {
  streaming = true;
  setDisabled(true);

  // Show "Thinking…" immediately — before first event arrives (~1-2s gap)
  // This prevents the blank period after sending a message
  // toolEls is a stack — multiple tools can run concurrently (search + graph)
  let toolEls    = [addToolStep('Thinking…', true)];
  let agentEl    = null;
  let content    = '';
  let latency    = 0;
  let started    = false;
  let firstEvent = true;

  try {
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
      body: JSON.stringify(payload),
    });

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let   buf     = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (!raw || raw === '[DONE]') continue;
        let ev; try { ev = JSON.parse(raw); } catch { continue; }
        const t = ev.type || '';

        // Remove "Thinking…" on first real event
        if (firstEvent) {
          toolEls.forEach(el => el.remove());
          toolEls = [];
          firstEvent = false;
        }

        if (t === 'tool_start') {
          toolEls.push(addToolStep(toolLabel(ev.name), false));
          continue;
        }
        if (t === 'tool_end') {
          // Mark the most recent active (non-done) tool step as done
          const active = toolEls.filter(el => !el.classList.contains('done'));
          if (active.length > 0) markDone(active[active.length-1], toolLabel(ev.name));
          continue;
        }
        if (t === 'interrupt') {
          toolEls.filter(el => !el.classList.contains('done'))
            .forEach(el => markDone(el, 'Clarification needed'));
          toolEls = [];
          addHITL(ev.question || 'Please clarify:', ev.options || [], ev.allow_freetext !== false);
          interrupted = true; streaming = false; setDisabled(false); return;
        }
        if (t === 'error') {
          toolEls.forEach(el => el.remove()); toolEls = [];
          addErr(ev.message || 'Unknown error');
          break;
        }
        if (t === 'done') {
          latency = ev.latency_ms || 0;
          continue;
        }

        const token = ev.content || ev.result || ev.token || '';
        if (token) {
          if (!started) {
            // Mark all active tool steps done when answer starts streaming
            toolEls.filter(el => !el.classList.contains('done'))
              .forEach(el => markDone(el, 'Done'));
            toolEls = [];
            agentEl = addAgentBubble();
            started = true;
          }
          content += token;
          streamToken(agentEl, content);
        }
      }
    }

    if (started && agentEl) {
      const cleaned = clean(content);
      if (!cleaned) addErr('Response could not be displayed — try + New for a fresh session.');
      else finalize(agentEl, cleaned, latency);
    } else if (!started && !interrupted) addErr('No response received.');

  } catch (e) {
    addErr('Connection error: ' + e.message);
  } finally {
    // Clean up any lingering tool step indicators
    toolEls.forEach(el => el.remove());
    toolEls = [];
    streaming = false;
    setDisabled(false);
    scrollEnd();
  }
}

// ── DOM helpers ───────────────────────────────────────────────────────────
function hideWelcome() {
  const w = document.getElementById('welcome');
  if (w) w.remove();
}
function scrollEnd() {
  const m = document.getElementById('msgs');
  m.scrollTop = m.scrollHeight;
}
function addUser(text) {
  const m = document.getElementById('msgs');
  const d = document.createElement('div');
  d.className = 'msg user';
  d.innerHTML = `<div class="av user">U</div><div class="bubble">${esc(text)}</div>`;
  m.appendChild(d); scrollEnd();
}
function addAgentBubble() {
  const m = document.getElementById('msgs');
  const d = document.createElement('div');
  d.className = 'msg agent';
  d.innerHTML = `<div class="av agent">AI</div><div class="bubble"><span class="st"></span><span class="cur"></span></div>`;
  m.appendChild(d); scrollEnd();
  return d;
}
function streamToken(el, content) {
  const st = el.querySelector('.st');
  if (st) st.textContent = content;
  scrollEnd();
}
// Inline markdown renderer — no CDN dependency
// Handles: headers, bold, italic, code, lists, blockquote, hr, links, tables
function mdParse(t) {
  // Escape HTML in code blocks first
  t = t.replace(/```([\s\S]*?)```/g, (_, c) =>
    '<pre><code>' + c.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</code></pre>');
  t = t.replace(/`([^`]+)`/g, (_, c) =>
    '<code>' + c.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</code>');
  // Headers
  t = t.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  t = t.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  t = t.replace(/^# (.+)$/gm, '<h1>$1</h1>');
  // Bold / italic
  t = t.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  t = t.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  t = t.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Blockquote
  t = t.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');
  // HR
  t = t.replace(/^---+$/gm, '<hr>');
  // Unordered lists — tag with uli to avoid merging with ordered
  t = t.replace(/^[\*\-] (.+)$/gm, '<uli>$1</uli>');
  t = t.replace(/(<uli>.*<\/uli>\n?)+/g, s => '<ul>' + s.replace(/uli>/g,'li>') + '</ul>');
  // Ordered lists — tag with oli to avoid merging with unordered
  t = t.replace(/^\d+\. (.+)$/gm, '<oli>$1</oli>');
  t = t.replace(/(<oli>.*<\/oli>\n?)+/g, s => '<ol>' + s.replace(/oli>/g,'li>') + '</ol>');
  // Links
  t = t.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
  // Tables (simple)
  t = t.replace(/^\|(.+)\|$/gm, (_, row) => {
    if (row.includes('---')) return '';
    const cells = row.split('|').map(c => c.trim());
    return '<tr>' + cells.map(c => `<td>${c}</td>`).join('') + '</tr>';
  });
  t = t.replace(/(<tr>.*<\/tr>\n?)+/g, s => '<table>' + s + '</table>');
  // Paragraphs — wrap non-tag lines
  t = t.split('\n\n').map(block => {
    if (/^<(h[1-6]|ul|ol|li|pre|blockquote|hr|table|tr)/.test(block.trim())) return block;
    const trimmed = block.trim();
    if (!trimmed) return '';
    return '<p>' + trimmed.replace(/\n/g, ' ') + '</p>';
  }).join('\n');
  return t;
}

function finalize(el, content, latency) {
  const b = el.querySelector('.bubble');
  if (!b) return;
  const c = b.querySelector('.cur'); if (c) c.remove();
  b.innerHTML = mdParse(content);
  if (latency > 0) {
    const f = document.createElement('div');
    f.className = 'meta-footer';
    f.innerHTML = `<span>⏱ ${(latency/1000).toFixed(1)}s</span><span>Research purposes only · Not medical advice</span>`;
    b.appendChild(f);
  }
  scrollEnd();
}
function addToolStep(label, thinking = false) {
  const m = document.getElementById('msgs');
  const d = document.createElement('div');
  d.className = 'tool-step' + (thinking ? ' thinking' : '');
  d.innerHTML = `<div class="spin${thinking ? ' pulse' : ''}"></div><span>${esc(label)}</span>`;
  m.appendChild(d); scrollEnd();
  return d;
}
function markDone(el, label) {
  el.classList.add('done');
  el.classList.remove('thinking');
  const sp = el.querySelector('.spin');
  const tx = el.querySelector('span:last-child');
  if (sp) sp.outerHTML = '<span style="color:var(--green);font-size:13px;flex-shrink:0">✓</span>';
  if (tx && label) tx.textContent = label;
  scrollEnd();
}
function addHITL(question, options, ft) {
  const m = document.getElementById('msgs');
  const c = document.createElement('div');
  c.className = 'hitl';
  const opts = options.map((o, i) =>
    `<button class="hitl-opt" onclick="pick(this,'${escA(o)}')">`+
    `<span class="num">${i+1}</span><span>${esc(o)}</span></button>`
  ).join('');
  c.innerHTML =
    `<div class="hitl-tag">Clarification needed</div>`+
    `<div class="hitl-q">${esc(question)}</div>`+
    `<div class="hitl-opts">${opts}</div>`+
    (ft ? `<div class="hitl-hint">Or type your answer below</div>` : '');
  m.appendChild(c); scrollEnd();
}
function pick(btn, val) {
  btn.closest('.hitl-opts').querySelectorAll('.hitl-opt')
    .forEach(b => b.style.pointerEvents = 'none');
  btn.classList.add('picked');
  addUser(val);
  doResume(val);
}
function addErr(msg) {
  const m = document.getElementById('msgs');
  const d = document.createElement('div');
  d.className = 'err';
  d.textContent = '⚠ ' + msg;
  m.appendChild(d); scrollEnd();
}
function setDisabled(v) { document.getElementById('sbtn').disabled = v; }
function newChat() {
  threadId = uuid(); interrupted = false; streaming = false;
  document.getElementById('sid').textContent = 'session: ' + threadId.slice(0, 8);
  const m = document.getElementById('msgs');
  m.innerHTML = `
    <div class="welcome" id="welcome">
      <div class="welcome-eyebrow">Vidya Sankalp · AgentCore Platform</div>
      <div class="welcome-title">Clinical Trial<br><span>Intelligence</span></div>
      <p class="welcome-sub">Search 5,772 trial documents and a live biomedical knowledge graph.</p>
      <div class="starters-label">Try one of these</div>
      <div class="starters" id="starters"></div>
    </div>`;
  const s = document.getElementById('starters');
  STARTERS.forEach(q => {
    const b = document.createElement('button');
    b.className = 'starter'; b.textContent = q;
    b.onclick = () => submit(q);
    s.appendChild(b);
  });
  setDisabled(false);
  inp.value = ''; inp.style.height = 'auto';
}

// ── Utils ─────────────────────────────────────────────────────────────────
function toolLabel(n) {
  if (!n) return 'Processing…';
  if (n.includes('search'))     return 'Searching knowledge base…';
  if (n.includes('graph'))      return 'Querying knowledge graph…';
  if (n.includes('summariser')) return 'Synthesising results…';
  if (n.includes('ask_user'))   return 'Preparing clarification…';
  return n;
}
function clean(t) {
  // Strip EPISODIC tag
  t = t.replace(/\n?EPISODIC:\s*(YES|NO)[\d.\s]*/gi, '');
  // Strip duplicate disclaimer
  t = t.replace(/\n?This information is for research purposes only and does not constitute medical advice\.?\s*/gi, '');
  // Strip guardrail reason
  t = t.replace(/\n?\[Reason logged for review:.*?\]\s*/gis, '');

  // Strip SummarizationMiddleware summary leaking into response.
  // Summaries arrive in two formats:
  //   WITH dash:    "- User inquired about cancer trials..."
  //   WITHOUT dash: "User inquired about other trials in the database..."
  // Both formats need to be stripped.

  // Strip bullet-point summary lines
  t = t.replace(/^[-•]\s+(User (inquired|asked|mentioned|expressed)|AI (provided|gave|discussed)|The (user|AI|agent)|Overall,|Key points)[^\n]*/gim, '');

  // Detect if ENTIRE response is a summary leak (with or without bullet prefix).
  // Summarization HumanMessage starts with "Here is a summary of the conversation to date"
  // but the LLM echoes it in its own words starting with "User inquired..." or
  // "A tool was called..." or similar summary-style openings.
  const summaryStart = /^(User (inquired|asked|mentioned)|A tool was called|The (user|AI|agent) (inquired|asked|provided|discussed)|Here is a summary)/i;
  if (summaryStart.test(t.trim())) {
    // Whole response is a summary echo — strip the leading summary paragraph
    // and keep only content after the first double newline (the actual answer)
    const parts = t.split(/\n\n+/);
    // Find first part that doesn't look like a summary
    const realContent = parts.find(p =>
      p.trim().length > 0 &&
      !summaryStart.test(p.trim()) &&
      !/^[-•]\s+(User|AI|The user|The agent)/i.test(p.trim())
    );
    if (realContent) t = parts.slice(parts.indexOf(realContent)).join('\n\n');
    else return ''; // entire response was summary — show nothing
  }

  return t.trim();
}
function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function escA(s) { return String(s).replace(/'/g, "\\'"); }
</script>
</body>
</html>"""