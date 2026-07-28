/* ============================================================
 * Fin Customer Agent — 前端逻辑
 * 负责：会话管理、明暗主题、SSE 流式聊天、Markdown 渲染、
 * 节点进度条、合规元信息
 * ============================================================ */

(() => {
  "use strict";

  // ── DOM 引用 ──
  const $ = (sel) => document.querySelector(sel);

  const dom = {
    chatLog: $("#chatLog"),
    chatForm: $("#chatForm"),
    input: $("#messageInput"),
    sendButton: $("#sendButton"),
    latencyText: $("#latencyText"),
    connectionStatus: $("#connectionStatus"),
    sessionIdText: $("#sessionIdText"),
    composerStatus: $("#composerStatus"),
    newChatBtn: $("#newChatBtn"),
    sessionList: $("#sessionList"),
    sessionCount: $("#sessionCount"),
    sessionEmpty: $("#sessionEmpty"),
    themeToggle: $("#themeToggle"),
    themeLabel: $(".theme-label"),
    emptyState: $("#emptyState"),
  };

  // ── 状态 ──
  const state = {
    sessionId: localStorage.getItem("smartcs.sessionId") || "",
    sessions: loadSessions(),
    busy: false,
    theme: localStorage.getItem("smartcs.theme") || matchMediaTheme(),
    streaming: null,
  };

  function matchMediaTheme() {
    return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function loadSessions() {
    try {
      return JSON.parse(localStorage.getItem("smartcs.sessions") || "{}");
    } catch {
      return {};
    }
  }

  function saveSessions() {
    localStorage.setItem("smartcs.sessions", JSON.stringify(state.sessions));
  }

  // ── 主题 ──

  function applyTheme(theme) {
    state.theme = theme;
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("smartcs.theme", theme);
    if (dom.themeLabel) {
      dom.themeLabel.textContent = theme === "dark" ? "浅色模式" : "深色模式";
    }
  }

  dom.themeToggle?.addEventListener("click", () => {
    applyTheme(state.theme === "dark" ? "light" : "dark");
  });

  applyTheme(state.theme);

  // 跟随系统变更
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
    if (localStorage.getItem("smartcs.theme.pinned") === "1") return;
    applyTheme(e.matches ? "dark" : "light");
  });

  dom.themeToggle?.addEventListener("dblclick", () => {
    localStorage.setItem("smartcs.theme.pinned", "1");
  });

  // ── 会话管理 ──

  function bindSession(sessionId) {
    state.sessionId = sessionId || "";
    if (sessionId) localStorage.setItem("smartcs.sessionId", sessionId);
    dom.sessionIdText.textContent = sessionId
      ? sessionId.slice(0, 8)
      : "未创建";
    renderSessionList();
  }

  function trackSession(sessionId, title) {
    if (!sessionId) return;
    state.sessions[sessionId] = {
      title: title || state.sessions[sessionId]?.title || newConversationTitle(),
      time: Date.now(),
    };
    saveSessions();
    renderSessionList();
  }

  function newConversationTitle() {
    const d = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    return `${pad(d.getMonth() + 1)}/${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  }

  function renderSessionList() {
    const items = Object.entries(state.sessions)
      .sort((a, b) => b[1].time - a[1].time)
      .slice(0, 20);

    dom.sessionCount.textContent = String(items.length);
    dom.sessionList.innerHTML = "";

    for (const [id, meta] of items) {
      const li = document.createElement("li");
      li.className = "session-item" + (id === state.sessionId ? " active" : "");
      li.dataset.id = id;

      const title = document.createElement("span");
      title.className = "session-title";
      title.textContent = meta.title || "对话";

      const time = document.createElement("span");
      time.className = "session-time";
      time.textContent = formatRelativeTime(meta.time);

      const del = document.createElement("button");
      del.type = "button";
      del.className = "session-delete";
      del.setAttribute("aria-label", "删除会话");
      del.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M6 6l1 14a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-14"/></svg>';
      del.addEventListener("click", (e) => {
        e.stopPropagation();
        delete state.sessions[id];
        saveSessions();
        if (id === state.sessionId) startNewChat();
        renderSessionList();
      });

      li.append(title, time, del);
      li.addEventListener("click", () => switchToSession(id));
      dom.sessionList.append(li);
    }
  }

  function formatRelativeTime(ts) {
    const diff = Date.now() - ts;
    if (diff < 60_000) return "刚刚";
    if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}分钟前`;
    if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}小时前`;
    const d = new Date(ts);
    return `${d.getMonth() + 1}/${d.getDate()}`;
  }

  function switchToSession(id) {
    if (state.busy) return;
    if (id === state.sessionId) return;
    bindSession(id);
    clearChatLog();
    // 重新拉取历史（如果有相应接口则调用；这里简化为提示）
    pushNotice(`已切换到会话 ${id.slice(0, 8)}，发送新消息继续对话`);
  }

  function startNewChat() {
    bindSession("");
    clearChatLog();
    showEmptyState();
    dom.input.focus();
  }

  dom.newChatBtn?.addEventListener("click", startNewChat);

  function clearChatLog() {
    // 保留 emptyState 由 showEmptyState 决定是否显示
    [...dom.chatLog.querySelectorAll(".message")].forEach((n) => n.remove());
  }

  function showEmptyState() {
    if (!document.querySelector(".empty-state")) {
      dom.chatLog.append(dom.emptyState);
    }
  }

  function hideEmptyState() {
    dom.emptyState.remove();
  }

  // ── Markdown 渲染（轻量、安全） ──

  const ESCAPE_MAP = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  };

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ESCAPE_MAP[c]);
  }

  function renderMarkdown(text) {
    // 先转义所有 HTML
    let safe = escapeHtml(text);

    // 代码块 ```...```
    safe = safe.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code}</code></pre>`);
    // 行内代码 `...`
    safe = safe.replace(/`([^`\n]+)`/g, (_, code) => `<code>${code}</code>`);
    // 标题
    safe = safe.replace(/^### (.+)$/gm, "<h3>$1</h3>");
    safe = safe.replace(/^## (.+)$/gm, "<h2>$1</h2>");
    safe = safe.replace(/^# (.+)$/gm, "<h1>$1</h1>");
    // 引用
    safe = safe.replace(/^&gt; (.+)$/gm, "<blockquote>$1</blockquote>");
    // 链接 [text](url)
    safe = safe.replace(
      /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>',
    );
    // 粗体 + 斜体
    safe = safe.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    safe = safe.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
    // 列表
    safe = safe.replace(/(^|\n)((?:- .+(?:\n|$))+)/g, (m, p, block) => {
      const items = block
        .trim()
        .split("\n")
        .map((l) => l.replace(/^- /, ""))
        .map((l) => `<li>${l}</li>`)
        .join("");
      return `${p}<ul>${items}</ul>`;
    });
    safe = safe.replace(/(^|\n)((?:\d+\. .+(?:\n|$))+)/g, (m, p, block) => {
      const items = block
        .trim()
        .split("\n")
        .map((l) => l.replace(/^\d+\. /, ""))
        .map((l) => `<li>${l}</li>`)
        .join("");
      return `${p}<ol>${items}</ol>`;
    });
    // 换行
    safe = safe.replace(/\n/g, "<br>");
    return safe;
  }

  // ── 消息构造 ──

  const STEP_LABELS = {
    supervisor_route: "意图识别",
    knowledge_rag: "知识检索",
    ticket_handler: "工单处理",
    compliance_check: "合规审查",
    synthesize: "汇总生成",
  };

  function buildMessage(role, text) {
    const article = document.createElement("article");
    article.className = `message ${role}`;

    const row = document.createElement("div");
    row.className = "message-row";

    const mark = document.createElement("div");
    mark.className = "message-mark";
    mark.textContent = role === "user" ? "您" : "AI";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";

    const textEl = document.createElement("div");
    textEl.className = "message-text";
    textEl.textContent = text || "";

    bubble.append(textEl);
    row.append(mark, bubble);

    article.append(row);

    if (role === "assistant") {
      const meta = document.createElement("div");
      meta.className = "message-meta";
      article.append(meta);
    }

    dom.chatLog.append(article);
    dom.chatLog.scrollTop = dom.chatLog.scrollHeight;
    return article;
  }

  function updateAssistantText(article, text) {
    const textEl = article.querySelector(".message-text");
    textEl.textContent = text;
    dom.chatLog.scrollTop = dom.chatLog.scrollHeight;
  }

  function renderAssistantText(article, text) {
    const textEl = article.querySelector(".message-text");
    textEl.innerHTML = renderMarkdown(text);
    dom.chatLog.scrollTop = dom.chatLog.scrollHeight;
  }

  function addMetaTag(article, label, kind, title) {
    const meta = article.querySelector(".message-meta");
    if (!meta) return;
    const tag = document.createElement("span");
    tag.className = `message-tag ${kind}`;
    tag.textContent = label;
    if (title) tag.title = title;
    meta.append(tag);
  }

  function buildStepper(article) {
    const wrap = document.createElement("div");
    wrap.className = "stepper";
    const steps = ["supervisor_route", "knowledge_rag", "ticket_handler", "compliance_check", "synthesize"];
    for (const id of steps) {
      const step = document.createElement("span");
      step.className = "step";
      step.dataset.step = id;
      step.textContent = STEP_LABELS[id] || id;
      wrap.append(step);
    }
    const meta = article.querySelector(".message-meta");
    if (meta) meta.before(wrap);
    else article.append(wrap);
    return wrap;
  }

  function setStep(stepper, nodeId) {
    if (!stepper) return;
    for (const step of stepper.querySelectorAll(".step")) {
      step.classList.toggle("active", step.dataset.step === nodeId);
    }
  }

  function clearSteps(stepper) {
    stepper?.querySelectorAll(".step").forEach((s) => s.classList.remove("active"));
  }

  function pushNotice(text) {
    const article = buildMessage("assistant", text);
    renderAssistantText(article, text);
  }

  // ── 输入框自适应 ──

  function resizeInput() {
    dom.input.style.height = "auto";
    dom.input.style.height = `${Math.min(dom.input.scrollHeight, 180)}px`;
  }

  dom.input.addEventListener("input", resizeInput);

  // ── 状态切换 ──

  function setBusy(busy) {
    state.busy = busy;
    dom.chatForm.classList.toggle("busy", busy);
    dom.sendButton.disabled = busy;
    dom.input.disabled = busy;
    setComposerStatus(busy ? "处理中…" : "就绪", busy ? "busy" : "");
  }

  function setComposerStatus(text, kind = "") {
    dom.composerStatus.textContent = text;
    dom.composerStatus.className = `composer-status ${kind}`.trim();
  }

  function setLatency(ms) {
    dom.latencyText.textContent = ms ? `${Math.round(ms)} ms` : "--";
  }

  function setCompliance(level) {
    dom.connectionStatus.classList.remove("offline", "warn", "fail");
    if (level === "fail") {
      dom.connectionStatus.querySelector("span:last-child").textContent = "已拦截";
    } else {
      dom.connectionStatus.querySelector("span:last-child").textContent = "已连接";
    }
  }

  // ── 发送消息 ──

  async function sendMessage(message) {
    hideEmptyState();
    trackSession(state.sessionId, message.slice(0, 24));

    const userMsg = buildMessage("user", message);
    renderAssistantText(userMsg, message);

    const pending = buildMessage("assistant", "");
    const stepper = buildStepper(pending);

    const textEl = pending.querySelector(".message-text");
    textEl.classList.add("is-empty");

    setBusy(true);
    const startedAt = performance.now();
    let lastEvent = "";

    let response;
    try {
      response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: "web_user",
          session_id: state.sessionId || null,
          message,
        }),
      });
      if (!response.ok || !response.body) {
        const err = await safeReadError(response);
        throw new Error(err || `请求失败 (${response.status})`);
      }
    } catch (err) {
      textEl.classList.remove("is-empty");
      textEl.textContent = `请求失败：${err.message}`;
      setComposerStatus("出错", "error");
      setBusy(false);
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalText = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        let eventType = "";
        for (const raw of lines) {
          if (raw.startsWith("event: ")) {
            eventType = raw.slice(7).trim();
          } else if (raw.startsWith("data: ")) {
            const payload = raw.slice(6);
            if (!payload || eventType === "done") continue;
            let data;
            try {
              data = JSON.parse(payload);
            } catch {
              continue;
            }

            if (eventType === "node_update") {
              lastEvent = data.node || lastEvent;
              setStep(stepper, lastEvent);
              if (data.display && !finalText) {
                textEl.textContent = data.display;
                dom.chatLog.scrollTop = dom.chatLog.scrollHeight;
              }
            } else if (eventType === "final") {
              finalText = data.response || "";
              renderAssistantText(pending, finalText);
              textEl.classList.remove("is-empty");
              clearSteps(stepper);
              stepper.remove();
              bindSession(data.session_id);
              trackSession(data.session_id, message.slice(0, 24));

              const duration = performance.now() - startedAt;
              setLatency(duration);

              const intent = data.intent || "unknown";
              addMetaTag(pending, intentLabel(intent), "intent", "路由意图");
              const compliance = data.compliance_passed;
              if (compliance) {
                addMetaTag(pending, "已通过", "compliance-pass", "合规审查通过");
              } else {
                addMetaTag(pending, "已拦截", "compliance-fail", "合规审查未通过");
                setCompliance("fail");
              }
              addMetaTag(pending, `${Math.round(duration)} ms`, "", "总耗时");
            } else if (eventType === "error") {
              throw new Error(data.error || "流式错误");
            }
          }
        }
      }
    } catch (err) {
      textEl.classList.remove("is-empty");
      textEl.textContent = `流式传输异常：${err.message}`;
      setComposerStatus("出错", "error");
    } finally {
      setBusy(false);
      dom.input.focus();
    }
  }

  function intentLabel(intent) {
    return {
      knowledge_rag: "知识库",
      ticket_handler: "工单",
      compliance_checker: "合规",
      compliance_check: "合规",
    }[intent] || intent;
  }

  async function safeReadError(response) {
    try {
      const err = await response.json();
      return err.detail || err.error || err.message;
    } catch {
      return null;
    }
  }

  // ── 表单提交 ──

  dom.chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const message = dom.input.value.trim();
    if (!message || state.busy) return;
    dom.input.value = "";
    resizeInput();
    sendMessage(message);
  });

  dom.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      dom.chatForm.requestSubmit();
    }
  });

  // ── 快捷提问 ──

  document.querySelectorAll("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => {
      dom.input.value = button.dataset.prompt || "";
      resizeInput();
      dom.input.focus();
    });
  });

  // ── 启动 ──

  bindSession(state.sessionId);
  renderSessionList();
  resizeInput();
  setComposerStatus("就绪");
})();