# QSRR User Manual

Ask questions about your research papers using natural language.

---

## Setup (Mac/Linux)

### Requirements
- Python 3.11+
- Node.js 18+
- Ollama
- tmux
- ~25GB VRAM

### SSH Port Forwarding

```bash
ssh -L 3000:localhost:3000 magi
```

With the tunnel active, `http://localhost:3000` on your machine reaches the server.

### Installation

```bash
git clone https://github.com/hakeematyab/Queryable-Shared-Reference-Repository.git
cd Queryable-Shared-Reference-Repository/app
chmod +x setup.sh startup.sh shutdown.sh
./startup.sh
```

The startup script:
1. Runs `setup.sh` (installs dependencies, downloads required AI model)
2. Starts Ollama server
3. Starts backend on `http://localhost:8000`
4. Starts frontend on `http://localhost:3000`

All servers run in a tmux session named `qsrr`.

### Verify Servers are Running

```bash
tmux ls
```

If running, you'll see `qsrr` in the output. To view the session:

```bash
tmux attach -t qsrr
```

Press `Ctrl+B` then `D` to detach without stopping servers.

### First-Time Setup Notes
- You'll be prompted to log in to HuggingFace if not already logged in (required for model access)
- The AI model `qwen3:8b` will be downloaded (~5GB)

### Shutdown

```bash
./shutdown.sh
```

---

## Using the App

Open `http://localhost:3000` in your browser.

### Login

Enter any username. This saves your chat history under that name.

### Main Interface

| Area | Description |
|------|-------------|
| **Sidebar (left)** | "New Chat" button, list of previous chats, "Load More" for older chats |
| **Chat area (center)** | Conversation display |
| **Input bar (bottom)** | Text input, `+` button for adding papers, send button |

### Asking Questions

1. Type a question in the input box
2. Press Enter or click the send button
3. Wait for the response (loading messages appear while processing)

**Note**: You can only converse with one chat at a time. You'll have to wait until current response is generated before requesting a new response

### Adding Papers

1. Click the `+` button next to the input
2. Click "Add New Papers"
3. Select files (PDF, TXT, MD, DOC, DOCX) [**PDF Highly Recommended**]
4. Selected files appear as chips above the input
5. Press Enter to start indexing
6. Progress messages show indexing status

**Tip:**
- Only one person can index papers at a time so add multiple papers at once
- Indexing can take a while so be patient

### Chat History

- Click "New Chat" in sidebar to start fresh
- Click any previous chat to reload that conversation
- Click "Load More" at bottom of sidebar for older chats

---

## Response Elements

### Trust Badges

Every AI response shows a grounding indicator:

| Badge | Meaning |
|-------|---------|
| ✓ Green | Backed by sources — verify key claims |
| ⚠ Amber | Weak source support — check carefully |
| ✕ Red | No source backing — likely fabricated |
| ? Gray | Hallucination rating unavailable |

### Citations

Responses may include a "Sources" section:
- Click to expand the list
- Shows which papers/sections the information came from
- Copy button copies all citations

### Response Timer

| Display | Meaning |
|---------|---------|
| Start: X.Xs | Time until first token appeared |
| Clock + X.Xs | Total response time |

---

## Startup Options

```bash
./startup.sh            # Development mode (hot reload)
./startup.sh --prod     # Production mode (static build, multiple workers)
./startup.sh -w 8       # Set number of backend workers
```