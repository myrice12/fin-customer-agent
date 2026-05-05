const chatLog = document.querySelector("#chatLog");
const chatForm = document.querySelector("#chatForm");
const input = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");
const latencyText = document.querySelector("#latencyText");
const complianceBadge = document.querySelector("#complianceBadge");

let sessionId = localStorage.getItem("smartcs.sessionId") || "";

function setSession(id) {
  sessionId = id;
  localStorage.setItem("smartcs.sessionId", id);
}

function addMessage(role, text) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const mark = document.createElement("div");
  mark.className = "message-mark";
  mark.textContent = role === "user" ? "CL" : "AG";

  const body = document.createElement("div");
  body.className = "message-body";

  const content = document.createElement("p");
  content.textContent = text;

  body.append(content);
  article.append(mark, body);
  chatLog.append(article);
  chatLog.scrollTop = chatLog.scrollHeight;
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
  complianceBadge.className = `status-chip ${state}`;
}

async function sendMessage(message) {
  addMessage("user", message);
  const pending = addMessage("assistant", "正在处理请求...");
  setBusy(true);
  setComplianceState("", "Running");

  const startedAt = performance.now();
  const contentEl = pending.querySelector("p");

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
            } else if (eventType === "final") {
              const duration = performance.now() - startedAt;
              latencyText.textContent = `${Math.round(duration)} ms`;
              contentEl.textContent = data.response;
              setSession(data.session_id);
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
