const chatLog = document.querySelector("#chatLog");
const chatForm = document.querySelector("#chatForm");
const input = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");
const healthText = document.querySelector("#healthText");
const sessionText = document.querySelector("#sessionText");
const intentText = document.querySelector("#intentText");
const latencyText = document.querySelector("#latencyText");
const complianceBadge = document.querySelector("#complianceBadge");
const toolList = document.querySelector("#toolList");
const toolCountText = document.querySelector("#toolCountText");
const messageCountText = document.querySelector("#messageCountText");
const resultText = document.querySelector("#resultText");

let sessionId = localStorage.getItem("smartcs.sessionId") || "";
let messageCount = 1;

function setSession(id) {
  sessionId = id;
  localStorage.setItem("smartcs.sessionId", id);
  sessionText.textContent = id ? id.slice(0, 8) : "New thread";
}

function updateMessageCount() {
  messageCountText.textContent = `${messageCount} msg`;
}

function addMessage(role, text) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const mark = document.createElement("div");
  mark.className = "message-mark";
  mark.textContent = role === "user" ? "CL" : "AG";

  const body = document.createElement("div");
  body.className = "message-body";

  const meta = document.createElement("div");
  meta.className = "message-meta";
  meta.innerHTML =
    role === "user"
      ? "<span>Client</span><span>Submitted</span>"
      : "<span>Assistant</span><span>Processed</span>";

  const content = document.createElement("p");
  content.textContent = text;

  body.append(meta, content);
  article.append(mark, body);
  chatLog.append(article);
  chatLog.scrollTop = chatLog.scrollHeight;
  messageCount += 1;
  updateMessageCount();
  return article;
}

function setBusy(isBusy) {
  sendButton.disabled = isBusy;
  input.disabled = isBusy;
  sendButton.querySelector("span").textContent = isBusy ? "Sending" : "Send";
}

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
}

function setComplianceState(state, label) {
  complianceBadge.textContent = label;
  complianceBadge.classList.toggle("pass", state === "pass");
  complianceBadge.classList.toggle("fail", state === "fail");
  resultText.textContent = label;
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error();
    healthText.textContent = "Online";
  } catch {
    healthText.textContent = "Offline";
  }
}

async function loadTools() {
  try {
    const response = await fetch("/api/tools");
    const data = await response.json();
    const tools = data.tools || [];
    toolCountText.textContent = `${tools.length}`;
    toolList.innerHTML = "";

    tools.slice(0, 6).forEach((tool) => {
      const item = document.createElement("span");
      item.className = "tool-pill";
      item.textContent = tool.name;
      toolList.append(item);
    });
  } catch {
    toolCountText.textContent = "--";
    toolList.innerHTML = '<span class="tool-pill">Tools unavailable</span>';
  }
}

async function sendMessage(message) {
  addMessage("user", message);
  const pending = addMessage("assistant", "正在处理请求...");
  setBusy(true);
  setComplianceState("", "Running");

  const startedAt = performance.now();
  const contentEl = pending.querySelector("p");
  const metaEl = pending.querySelector(".message-meta");

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: "web_user",
        session_id: sessionId || null,
        message,
      }),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Request failed");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();

      let eventType = "";
      for (const line of lines) {
        if (line.startsWith("event: ")) {
          eventType = line.slice(7);
        } else if (line.startsWith("data: ")) {
          const raw = line.slice(6);
          if (!raw || eventType === "done") continue;

          try {
            const data = JSON.parse(raw);

            if (eventType === "node_update") {
              contentEl.textContent = data.display || "处理中...";
              if (data.intent) {
                intentText.textContent = data.intent;
              }
            } else if (eventType === "final") {
              const duration = performance.now() - startedAt;
              latencyText.textContent = `${Math.round(duration)} ms`;
              contentEl.textContent = data.response;
              metaEl.innerHTML = `
                <span>Assistant</span>
                <span>${data.intent || "unknown"}</span>
              `;
              setSession(data.session_id);
              intentText.textContent = data.intent || "unknown";
              setComplianceState(
                data.compliance_passed ? "pass" : "fail",
                data.compliance_passed ? "Passed" : "Escalated"
              );
            } else if (eventType === "error") {
              throw new Error(data.error || "Stream error");
            }
          } catch (parseErr) {
            if (parseErr.message && !parseErr.message.includes("JSON")) throw parseErr;
          }
        }
      }
    }
  } catch (error) {
    contentEl.textContent = `请求失败：${error.message}`;
    metaEl.innerHTML = "<span>Assistant</span><span>Failed</span>";
    setComplianceState("fail", "Failed");
  } finally {
    setBusy(false);
    input.focus();
  }
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  resizeInput();
  sendMessage(message);
});

input.addEventListener("input", resizeInput);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.prompt;
    resizeInput();
    input.focus();
  });
});

setSession(sessionId);
updateMessageCount();
checkHealth();
loadTools();
