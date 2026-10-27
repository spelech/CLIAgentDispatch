# ARCHITECTURE.md: CLIAgentDispatch

## 🏛️ System Topology

```mermaid
flowchart TD
    subgraph Clients ["AI Clients & Operator Consoles"]
        OCL["OpenClaw Agent (:10.0.10.10)"]
        CLI["Terminal CLI (cli-agent-dispatch)"]
        MB["Monitorbot SRE Watcher"]
        BROWSER["Operator Web Browser"]
    end

    subgraph Hubs ["Routing & Gateway Hubs"]
        LITE["LiteLLM Gateway (:8448)"]
        MCG["Model Context Gateway (:8026)"]
        NTFY["ntfy Push Server"]
        SLACK["Slack Workspace"]
    end

    subgraph Service ["CLIAgentDispatch (:8032)"]
        API["FastAPI Gateway"]
        STORE["SessionStore (Thread Continuity)"]
        OAI["OpenAI Bridge (/v1/chat/completions)"]
        ANTH["Anthropic Bridge (/v1/messages)"]
        MCP["FastMCP Server (/mcp)"]
        ENG["DispatchEngine"]
        PAIR["SSO Pair Router (/sso-pair)"]
        TAP["Diagnostic RingBuffer"]
    end

    subgraph Engines ["Local Execution Engines"]
        T3_ENG["T3 Code Daemon & MCP Client"]
        AGY_CLI["Antigravity agy CLI Binary"]
        OC_SRV["OpenCode HTTP API (:4096)"]
        OC_CLI["OpenCode CLI Binary"]
    end

    OCL -->|POST /v1/messages| ANTH
    OCL -->|POST /v1/chat/completions| LITE
    LITE -->|POST /v1/chat/completions| OAI
    MCG -->|SSE /mcp| MCP
    MB -->|Python Library Import| ENG
    CLI -->|Command Execution| ENG
    BROWSER -->|HTTPS GET /sso-pair| PAIR

    OAI --> STORE
    ANTH --> STORE
    STORE --> ENG
    MCP --> ENG
    PAIR --> T3_ENG
    PAIR -.->|Push Alert| NTFY
    ENG --> TAP
    ENG -.->|Live Card Updates| SLACK

    ENG -->|1. JSON-RPC MCP| T3_ENG
    ENG -->|2. Subprocess Exec| AGY_CLI
    ENG -->|3. HTTP API| OC_SRV
    ENG -.->|Fallback CLI| OC_CLI
```

---

## 🔄 Sequence: Anthropic & OpenAI Streaming Delegation with Extended Reasoning

```mermaid
sequenceDiagram
    autonumber
    actor Client as OpenClaw / LiteLLM
    participant Gateway as CLIAgentDispatch (:8032)
    participant Store as SessionStore
    participant Engine as DispatchEngine
    participant Agy as Antigravity (agy CLI)

    Client->>Gateway: POST /v1/messages or /v1/chat/completions (stream=true)
    Gateway->>Store: get(client_session_id)
    alt Existing Session Found
        Store-->>Gateway: native_session_id ("conv-123")
        Note over Gateway: Extract delta user turn only
    else New Session
        Store-->>Gateway: None
        Note over Gateway: Format full prompt + system instructions
    end
    Gateway->>Engine: dispatch_stream(TaskRequest with effort=high)
    Engine->>Agy: agy --model <model> --effort high --output-format stream-json --print <prompt>
    loop Streaming NDJSON Output & Reasoning
        alt Thought Block
            Agy-->>Engine: {"type": "thought", "delta": "..."}
            Engine-->>Gateway: StreamChunk(thinking_delta)
            Gateway-->>Client: SSE thinking_delta chunk
        else Content Block
            Agy-->>Engine: {"type": "step_update", "text_delta": "..."}
            Engine-->>Gateway: StreamChunk(text_delta)
            Gateway-->>Client: SSE text_delta chunk
        end
    end
    Agy-->>Engine: {"type": "result", "conversation_id": "conv-123"}
    Engine->>Store: update_native_id(client_id, "conv-123")
    Gateway-->>Client: Final SSE stop event
```

---

## 🔄 Sequence: T3 Code Headless Execution & Mid-Thread Model Switching

```mermaid
sequenceDiagram
    autonumber
    actor Client as OpenClaw / LiteLLM
    participant Gateway as CLIAgentDispatch (:8032)
    participant T3Exec as T3Executor (MCP Client)
    participant T3Daemon as T3 Code Daemon
    participant Slack as Slack Operator Card

    Client->>Gateway: POST /v1/chat/completions (model="t3-code")
    Gateway->>T3Exec: dispatch_stream(prompt, session_id)
    T3Exec->>T3Daemon: JSON-RPC initialize + start_session
    T3Daemon-->>T3Exec: session_id & capabilities
    T3Exec->>Slack: Post interactive tool card with [Stop Turn] & [Hijack Thread]
    loop Turn Execution & Tool Calls
        T3Daemon-->>T3Exec: tool_call("edit_file", args)
        T3Exec->>T3Daemon: approve_tool_call(call_id)
        T3Exec->>Slack: Update activity status
        T3Daemon-->>T3Exec: stream_chunk(content)
        T3Exec-->>Gateway: text_delta
        Gateway-->>Client: SSE chunk
    end
    opt Operator Hijack
        Slack->>Gateway: POST /api/sessions/{id}/hijack
        Gateway->>T3Exec: cancel_turn()
    end
```

---

## 🔄 Sequence: SSO Web Pairing (`/sso-pair`) & ntfy Push Alert

```mermaid
sequenceDiagram
    autonumber
    actor Operator as Operator Browser
    participant Gateway as CLIAgentDispatch (:8032)
    participant T3 as t3 CLI / Daemon
    participant Ntfy as ntfy Push Server

    Operator->>Gateway: GET /sso-pair (Authenticated via TinyAuth)
    Gateway->>T3: t3 auth pairing create --json
    T3-->>Gateway: {code: "ABCD-1234", url: "https://ttt.wileyriley.com/pair#token=..."}
    Gateway->>Ntfy: POST /agents-push (Code ready, Action: Copy Code)
    Gateway-->>Operator: Render Pair Card with [Copy Pairing Code] & [Pair Now] buttons
    Operator->>Operator: Click "Copy Pairing Code" and paste into terminal
```

---

## 🔄 Sequence: FastMCP Tool Delegation

```mermaid
sequenceDiagram
    autonumber
    actor Caller as MCG Client / Agent
    participant MCG as Model Context Gateway (:8026)
    participant CAD as CLIAgentDispatch (:8032)
    participant T3 as T3 Code
    participant AGY as agy CLI
    participant OC as OpenCode (:4096)

    Caller->>MCG: execute_tool("cli-agent-dispatch/delegate_task")
    MCG->>CAD: POST /mcp/call_tool (delegate_task, executor="t3")
    alt Executor == t3
        CAD->>T3: JSON-RPC send_message(session, task)
        T3-->>CAD: result content & tool call logs
    else Executor == agy
        CAD->>AGY: exec(agy --model ... --print <prompt>)
        AGY-->>CAD: stdout/stderr
    else Executor == opencode
        CAD->>OC: POST /session/{id}/message
        OC-->>CAD: text output parts
    end
    CAD-->>MCG: tool execution result
    MCG-->>Caller: Formatted response & diagnostics
```

---

## 📊 Endpoints & Routing Contract

| Endpoint | Method | Protocol | Description |
| :--- | :--- | :--- | :--- |
| `/v1/chat/completions` | `POST` | OpenAI HTTP / SSE | OpenAI compatible chat completions with multi-turn continuity & `reasoning_delta` |
| `/v1/models` | `GET` | OpenAI HTTP | Dynamic model enumeration across all underlying engines |
| `/v1/messages` | `POST` | Anthropic HTTP / SSE | Anthropic Messages API with `thinking_delta` streaming blocks |
| `/v1/screenshot` | `GET` | Binary / JSON | Real-time screenshot probe for visual debugging |
| `/mcp` / `/sse` | `GET`/`POST`| FastMCP | Model Context Protocol gateway with 9 native SRE/delegation tools |
| `/sso-pair` | `GET` | HTML / JSON | Interactive SSO pairing portal with one-click token copy |
| `/health` | `GET` | HTTP JSON | Shallow health probe and service version metadata |
| `/api/diagnostics` | `GET` | HTTP JSON | Detailed engine health, ring-buffer logs, and active session status |
