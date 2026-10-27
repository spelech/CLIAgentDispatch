import asyncio
import base64
import json
import logging
import os
import urllib.parse
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from cli_agent_dispatch.config import settings

logger = logging.getLogger("CLIAgentDispatch.T3Pair")

router = APIRouter(tags=["T3 Code Pairing"])


async def create_t3_pairing_link(
    ttl: str = "15m",
    label: str = "SSO Device",
    base_url: str | None = None,
) -> dict[str, Any]:
    """Mint a T3 Code client pairing credential and magic `/pair#token=...` URL."""
    cli_path = settings.t3_cli_path
    if not cli_path or not os.path.exists(cli_path):
        raise RuntimeError(f"t3 binary not found at {cli_path}")

    ui_base = (base_url or settings.t3_ui_url or "https://ttt.wileyriley.com").rstrip("/")
    proc = await asyncio.create_subprocess_exec(
        cli_path,
        "auth",
        "pairing",
        "create",
        "--base-url",
        ui_base,
        "--ttl",
        ttl,
        "--label",
        label,
        "--json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Failed to create T3 pairing credential: {err}")

    data = json.loads(stdout.decode("utf-8", errors="replace"))
    credential = str(data.get("credential") or "")
    if not credential:
        raise RuntimeError("T3 pairing output did not include a credential.")

    pair_url = str(data.get("pairUrl") or f"{ui_base}/pair#token={credential}")
    return {
        "id": data.get("id"),
        "credential": credential,
        "label": data.get("label") or label,
        "expires_at": data.get("expiresAt"),
        "pair_url": pair_url,
        "ui_url": ui_base,
    }


async def send_t3_pairing_ntfy(
    pair_data: dict[str, Any],
    topic_override: str | None = None,
) -> bool:
    """Publish a push notification to ntfy with a 1-tap action button to pair T3 Code."""
    if os.environ.get("DISABLE_NTFY") == "1":
        return True

    parsed = urllib.parse.urlparse(settings.ntfy_url)
    topic = (topic_override or parsed.path.strip("/")) or "wileyriley-sso"
    root_url = f"{parsed.scheme}://{parsed.netloc}"

    pair_url = str(pair_data["pair_url"])
    credential = str(pair_data["credential"])
    label = str(pair_data.get("label") or "Mobile / Remote Device")
    expires_at = str(pair_data.get("expires_at") or "15m")
    ui_url = str(pair_data.get("ui_url") or settings.t3_ui_url).rstrip("/")
    copy_url = f"{ui_url}/sso-pair/copy?" + urllib.parse.urlencode(
        {"code": credential, "expires": expires_at}
    )

    # Note: Do NOT set "click" so tapping the notification body simply opens
    # the notification inside the ntfy app without launching the T3 site.
    payload = {
        "topic": topic,
        "title": f"T3 Code Pairing ({label})",
        "markdown": True,
        "message": (
            f"Pairing code for **{label}** (Server: `{ui_url}`):\n"
            f"```\n{credential}\n```\n"
            f"Expires: `{expires_at}`"
        ),
        "priority": 4,
        "tags": ["key", "thread"],
        "actions": [
            {
                "action": "view",
                "label": "📋 Copy Code",
                "url": copy_url,
                "clear": False,
            },
            {
                "action": "view",
                "label": "🚀 Pair This Browser",
                "url": pair_url,
                "clear": True,
            },
        ],
    }

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.ntfy_user and settings.ntfy_pass:
        raw_auth = f"{settings.ntfy_user}:{settings.ntfy_pass}".encode()
        headers["Authorization"] = f"Basic {base64.b64encode(raw_auth).decode('ascii')}"

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(root_url, headers=headers, json=payload)
            resp.raise_for_status()
        return True
    except Exception as e:
        logger.warning(
            f"Failed to publish T3 pairing notification to ntfy ({root_url}/{topic}): {e}"
        )
        return False


def _render_pair_card_html(pair_data: dict[str, Any], ntfy_sent: bool, user_label: str) -> str:
    pair_url = pair_data["pair_url"]
    credential = pair_data["credential"]
    expires_at = pair_data.get("expires_at") or ""
    ntfy_badge = (
        "📲 Sent to ntfy (wileyriley-sso)"
        if ntfy_sent
        else "ℹ️ Generated locally (use ?notify=1 to push to ntfy)"
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>T3 Code — SSO Device Pairing</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #090d16;
      color: #f3f4f6;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      margin: 0;
      padding: 1.25rem;
      box-sizing: border-box;
    }}
    .card {{
      max-width: 440px;
      width: 100%;
      background: #111827;
      border: 1px solid #1f2937;
      border-radius: 16px;
      padding: 2rem;
      text-align: center;
      box-shadow: 0 20px 40px rgba(0,0,0,0.5);
    }}
    h1 {{ font-size: 1.35rem; margin: 0 0 0.5rem; }}
    p.sub {{ color: #9ca3af; font-size: 0.92rem; margin: 0 0 1.5rem; }}
    .code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 1.5rem;
      letter-spacing: 0.14em;
      background: #0b0f19;
      border: 1px solid #374151;
      border-radius: 10px;
      padding: 0.85rem;
      margin-bottom: 1.25rem;
      color: #60a5fa;
      user-select: all;
    }}
    a.btn, button.btn {{
      display: block;
      width: 100%;
      box-sizing: border-box;
      border: none;
      cursor: pointer;
      font-size: 0.96rem;
      background: #2563eb;
      color: #ffffff;
      text-decoration: none;
      font-weight: 600;
      padding: 0.9rem 1.25rem;
      border-radius: 10px;
      margin-bottom: 0.75rem;
    }}
    a.btn.secondary, button.btn.secondary {{
      background: #1f2937;
      color: #e5e7eb;
    }}
    .meta {{ color: #6b7280; font-size: 0.8rem; margin-top: 1rem; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>🧵 T3 Code Device Pairing</h1>
    <p class="sub">Authenticated as <strong>{user_label}</strong></p>
    <div class="code" id="pair-code">{credential}</div>
    <button class="btn" type="button" onclick="copyText('{credential}', this, '📋 Copy Pairing Code')">📋 Copy Pairing Code</button>
    <button class="btn secondary" type="button" onclick="copyText('{pair_url}', this, '🔗 Copy Pairing Link')">🔗 Copy Pairing Link</button>
    <a class="btn secondary" href="{pair_url}">🚀 Pair This Browser Now</a>
    <a class="btn secondary" href="/sso-pair?mode=card&notify=1">📲 Push Link to ntfy (Phone / Tablet)</a>
    <div class="meta">Server URL: <code>https://ttt.wileyriley.com</code><br/>{ntfy_badge}<br/>Expires: {expires_at}</div>
  </div>
  <script>
    async function copyText(val, btn, originalLabel) {{
      try {{
        await navigator.clipboard.writeText(val);
        btn.textContent = "✅ Copied!";
        setTimeout(() => {{ btn.textContent = originalLabel; }}, 2000);
      }} catch (e) {{
        const el = document.createElement("textarea");
        el.value = val;
        document.body.appendChild(el);
        el.select();
        document.execCommand("copy");
        document.body.removeChild(el);
        btn.textContent = "✅ Copied!";
        setTimeout(() => {{ btn.textContent = originalLabel; }}, 2000);
      }}
    }}
  </script>
</body>
</html>"""


@router.get("/sso-pair", response_model=None)
@router.post("/sso-pair", response_model=None)
@router.get("/v1/t3/pair", response_model=None)
@router.post("/v1/t3/pair", response_model=None)
async def sso_pair_t3(
    request: Request,
    ttl: str = Query(default="15m", description="Pairing credential TTL (e.g. 15m, 1h, 24h)"),
    label: str | None = Query(default=None, description="Device label"),
    notify: bool = Query(default=False, description="Also push the magic pairing link to ntfy"),
    mode: str = Query(
        default="card",
        description="Response mode: 'card' (HTML page with Copy Code & Pair buttons), 'redirect' (302 to /pair#token=...), or 'json'",
    ),
    topic: str | None = Query(default=None, description="Optional ntfy topic override"),
) -> RedirectResponse | HTMLResponse | JSONResponse:
    """
    Generate a T3 Code pairing link after TinyAuth/PocketID SSO authentication
    (or internal API invocation) and either:
      - 302 redirect the browser straight into `https://ttt.wileyriley.com/pair#token=...`
      - render a mobile/desktop pairing card (`?mode=card`)
      - return JSON (`?mode=json`), optionally pushing the link to ntfy (`?notify=1`).
    """
    remote_user = (
        request.headers.get("Remote-Name")
        or request.headers.get("Remote-User")
        or request.headers.get("Remote-Email")
        or "SSO User"
    )
    effective_label = label or f"{remote_user} ({ttl})"

    try:
        pair_data = await create_t3_pairing_link(ttl=ttl, label=effective_label)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    ntfy_sent = False
    if notify:
        ntfy_sent = await send_t3_pairing_ntfy(pair_data, topic_override=topic)

    accept = (request.headers.get("Accept") or "").lower()
    if mode == "json" or (
        mode != "card" and "application/json" in accept and "text/html" not in accept
    ):
        return JSONResponse(
            {
                **pair_data,
                "ntfy_sent": ntfy_sent,
                "authenticated_user": remote_user,
            }
        )

    if mode == "card":
        return HTMLResponse(_render_pair_card_html(pair_data, ntfy_sent, remote_user))

    return RedirectResponse(url=pair_data["pair_url"], status_code=302)


@router.get("/sso-pair/copy", response_class=HTMLResponse)
@router.get("/v1/t3/pair/copy", response_class=HTMLResponse)
async def sso_pair_copy_only(
    code: str = Query(..., description="T3 pairing credential code to copy"),
    expires: str = Query(default="", description="Optional expiration timestamp"),
) -> HTMLResponse:
    """
    Lightweight clipboard-copy helper opened by the ntfy 'Copy Code' action button.
    Does NOT open the T3 Code web app or consume the single-use pairing token.
    """
    safe_code = "".join(ch for ch in code if ch.isalnum() or ch in ("-", "_"))[:32]
    safe_expires = "".join(ch for ch in expires if ch.isalnum() or ch in ("-", ":", ".", " "))[:40]
    ui_url = settings.t3_ui_url.rstrip("/")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Copy T3 Pairing Code</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #090d16;
      color: #f3f4f6;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      margin: 0;
      padding: 1.25rem;
      box-sizing: border-box;
    }}
    .card {{
      max-width: 420px;
      width: 100%;
      background: #111827;
      border: 1px solid #1f2937;
      border-radius: 16px;
      padding: 2rem;
      text-align: center;
      box-shadow: 0 20px 40px rgba(0,0,0,0.5);
    }}
    h1 {{ font-size: 1.25rem; margin: 0 0 0.5rem; }}
    p.sub {{ color: #9ca3af; font-size: 0.9rem; margin: 0 0 1.25rem; }}
    .code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 1.65rem;
      letter-spacing: 0.14em;
      background: #0b0f19;
      border: 1px solid #374151;
      border-radius: 10px;
      padding: 0.9rem;
      margin-bottom: 1.25rem;
      color: #60a5fa;
      user-select: all;
    }}
    button.btn {{
      display: block;
      width: 100%;
      border: none;
      cursor: pointer;
      font-size: 1rem;
      background: #2563eb;
      color: #ffffff;
      font-weight: 600;
      padding: 0.95rem 1.25rem;
      border-radius: 10px;
      margin-bottom: 0.75rem;
    }}
    .meta {{ color: #6b7280; font-size: 0.8rem; margin-top: 1rem; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>📋 T3 Mobile App Pairing Code</h1>
    <p class="sub" id="status-msg">Tap below to copy for the T3 mobile app (token is NOT consumed here)</p>
    <div class="code" id="code-box">{safe_code}</div>
    <button class="btn" id="copy-btn" type="button" onclick="copyCode()">📋 Copy Code: {safe_code}</button>
    <div class="meta">Server URL: <code>{ui_url}</code><br/>Expires: {safe_expires}</div>
  </div>
  <script>
    const CODE = "{safe_code}";
    async function copyCode() {{
      const btn = document.getElementById("copy-btn");
      const status = document.getElementById("status-msg");
      try {{
        await navigator.clipboard.writeText(CODE);
        btn.textContent = "✅ Copied to Clipboard!";
        status.textContent = "Paste this code into your T3 mobile/tablet app.";
      }} catch (e) {{
        const el = document.createElement("textarea");
        el.value = CODE;
        document.body.appendChild(el);
        el.select();
        document.execCommand("copy");
        document.body.removeChild(el);
        btn.textContent = "✅ Copied to Clipboard!";
        status.textContent = "Paste this code into your T3 mobile/tablet app.";
      }}
    }}
    window.addEventListener("DOMContentLoaded", () => {{ copyCode(); }});
  </script>
</body>
</html>"""
    return HTMLResponse(html)
