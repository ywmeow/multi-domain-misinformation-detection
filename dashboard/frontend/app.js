const EXAMPLES = {
  news:
    "The city council approved a new public transport plan after a public consultation on Monday.",
  health:
    "A viral post claims that drinking lemon water cures all forms of cancer within a week.",
  science:
    "Researchers report that a new battery design retained most of its capacity after repeated charging in laboratory tests."
};

const MODEL_DETAILS = {
  bilstm: "First 512 tokens",
  graphsage: "Jruvika checkpoint"
};

function formatScore(probability) {
  const percent = probability * 100;
  if (percent > 0 && percent < 0.01) return "<0.01%";
  if (percent > 99.99 && percent < 100) return ">99.99%";
  return `${percent.toFixed(2)}%`;
}

function formatDuration(milliseconds) {
  if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`;
  return `${(milliseconds / 1000).toFixed(1)} s`;
}

function analysisStatus(analysis) {
  if (!analysis) return "Ready when you are";
  if (analysis.agreement === "agree") return "Both models agree";
  if (analysis.agreement === "disagree") return "The models disagree";
  return `${analysis.completed} of 2 models completed`;
}

function setAvailability(card, text) {
  const dot = document.createElement("span");
  dot.className = "status-dot";
  card.querySelector(".availability").replaceChildren(dot, text);
}

function resetCard(card, model) {
  delete card.dataset.label;
  card.querySelector("[data-prediction]").textContent = "Awaiting your text";
  card.querySelector("[data-message]").textContent = "";
  card.querySelector("[data-score]").textContent = "—";
  card.querySelector("[data-detail]").textContent = MODEL_DETAILS[model];
  card.querySelector("[data-time]").textContent = "— ms";
  card.querySelector(".score-fill").style.width = "0%";

  const meter = card.querySelector(".score-track");
  meter.removeAttribute("aria-valuenow");
  meter.setAttribute("aria-valuetext", "No prediction yet");
  setAvailability(card, "Available");
}

function renderPrediction(card, prediction) {
  if (prediction.status === "error") {
    card.dataset.label = "error";
    card.querySelector("[data-prediction]").textContent = "Prediction unavailable";
    card.querySelector("[data-message]").textContent = prediction.message;
    setAvailability(card, "Needs attention");
    return;
  }

  const percent = prediction.p_fake * 100;
  card.dataset.label = prediction.label;
  card.querySelector("[data-prediction]").textContent = `Predicted ${prediction.label}`;
  card.querySelector("[data-score]").textContent = formatScore(prediction.p_fake);
  card.querySelector("[data-time]").textContent = formatDuration(prediction.elapsed_ms);
  card.querySelector("[data-detail]").textContent = prediction.truncated
    ? "Truncated to 512 tokens"
    : `Fake threshold ≥ ${(prediction.threshold * 100).toFixed(0)}%`;
  card.querySelector(".score-fill").style.width = `${percent}%`;

  const meter = card.querySelector(".score-track");
  meter.setAttribute("aria-valuenow", String(percent));
  meter.setAttribute("aria-valuetext", `${percent.toFixed(4)} percent fake-class score`);
  setAvailability(card, "Completed");
}

function downloadResults(analysis) {
  if (!analysis) return;

  const result = {
    ...analysis,
    note: "Uncalibrated model outputs, not factual verification. RoBERTa is not connected."
  };
  const file = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(file);
  const link = document.createElement("a");
  link.href = url;
  link.download = "signal-analysis.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export default function mountDashboard(component) {
  const { parentElement, data, setTriggerValue } = component;
  const root = parentElement.querySelector(".signal-app");
  if (!root) return;

  // Discard old listeners on rerender so buttons use the latest Python data.
  root.__signalCleanup?.();
  const controller = new AbortController();
  const on = (element, event, handler) => {
    element.addEventListener(event, handler, { signal: controller.signal });
  };
  const find = selector => root.querySelector(selector);
  const elements = {
    form: find("#analysis-form"),
    input: find("#headline-input"),
    submit: find("#analyze-button"),
    submitLabel: find("#analyze-button-label"),
    clear: find("#clear-button"),
    error: find("#input-error"),
    count: find("#character-count"),
    status: find("#analysis-status"),
    summary: find("#analysis-summary"),
    submittedText: find("#submitted-text"),
    results: find(".results-grid"),
    export: find("#export-button")
  };
  const exampleButtons = root.querySelectorAll("[data-example]");

  // Drafts stay in this tab's memory; no browser storage is used.
  const state = root.__signalState ?? {
    draft: data.analysis?.text ?? "",
    pending: null
  };
  root.__signalState = state;
  if (state.pending?.id === data.last_request_id) state.pending = null;
  elements.input.value = state.draft;
  elements.input.maxLength = data.max_characters;

  function showError(message) {
    elements.error.textContent = message ?? "";
    elements.error.hidden = !message;
    elements.input.setAttribute("aria-invalid", String(Boolean(message)));
  }

  function updateInput() {
    state.draft = elements.input.value;
    const length = state.draft.length.toLocaleString();
    const limit = data.max_characters.toLocaleString();
    elements.count.textContent = `${length} / ${limit}`;

    const normalized = state.draft.normalize("NFC").trim().replace(/\s+/gu, " ");
    const changed = Boolean(data.analysis && normalized !== data.analysis.text);
    root.classList.toggle("input-changed", changed);
    if (!state.pending) {
      elements.status.textContent = changed
        ? "Text changed · analyze again to update"
        : analysisStatus(data.analysis);
    }
  }

  function updateBusyState() {
    const busy = Boolean(state.pending);
    const clearing = state.pending?.action === "reset";
    root.classList.toggle("is-busy", busy);
    elements.form.setAttribute("aria-busy", String(busy));
    elements.results.setAttribute("aria-busy", String(busy));
    elements.input.readOnly = busy;
    elements.submit.disabled = busy;
    elements.clear.disabled = busy;
    elements.export.disabled = busy || !data.analysis;
    elements.submitLabel.textContent = busy ? "Please wait…" : "Analyze text";
    exampleButtons.forEach(button => { button.disabled = busy; });

    if (busy) {
      elements.status.textContent = clearing
        ? "Clearing analysis…"
        : "Analyzing with BiLSTM and GraphSAGE…";
    }
  }

  function renderResults() {
    for (const model of Object.keys(MODEL_DETAILS)) {
      resetCard(find(`[data-model="${model}"]`), model);
    }
    elements.submittedText.hidden = !data.analysis;
    elements.submittedText.textContent = "";
    elements.summary.textContent = "2 models available";
    if (!data.analysis) return;

    for (const prediction of data.analysis.predictions) {
      if (prediction.model in MODEL_DETAILS) {
        renderPrediction(find(`[data-model="${prediction.model}"]`), prediction);
      }
    }
    elements.submittedText.textContent = `Analyzed text: “${data.analysis.text}”`;
    const duration = formatDuration(data.analysis.elapsed_ms);
    elements.summary.textContent = `${data.analysis.completed} models · ${duration} total`;
  }

  function sendRequest(action, text = "") {
    if (state.pending) return;
    state.pending = { id: crypto.randomUUID(), action };
    updateBusyState();
    setTriggerValue("request", { ...state.pending, text });
  }

  function submitAnalysis(event) {
    event.preventDefault();
    const text = elements.input.value;
    if (!text.trim()) {
      showError("Enter a headline, claim, or article to analyze.");
      elements.input.focus();
      return;
    }
    if (text.length > data.max_characters) {
      showError(`Keep the input under ${data.max_characters.toLocaleString()} characters.`);
      return;
    }
    showError(null);
    state.draft = text;
    sendRequest("analyze", text);
  }

  on(elements.form, "submit", submitAnalysis);
  on(elements.input, "input", () => {
    showError(null);
    updateInput();
  });
  on(elements.input, "keydown", event => {
    const shortcut = (event.ctrlKey || event.metaKey) && event.key === "Enter";
    if (shortcut && !state.pending) {
      event.preventDefault();
      elements.form.requestSubmit();
    }
  });
  on(elements.clear, "click", () => {
    elements.input.value = "";
    showError(null);
    updateInput();
    sendRequest("reset");
  });
  on(elements.export, "click", () => downloadResults(data.analysis));

  exampleButtons.forEach(button => {
    on(button, "click", () => {
      elements.input.value = EXAMPLES[button.dataset.example];
      showError(null);
      updateInput();
      elements.input.focus({ preventScroll: true });
    });
  });
  on(find("#workspace-link"), "click", event => {
    event.preventDefault();
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    find("#workspace").scrollIntoView({
      behavior: reducedMotion ? "instant" : "smooth",
      block: "start"
    });
  });

  renderResults();
  showError(data.error);
  updateInput();
  updateBusyState();

  const cleanup = () => controller.abort();
  root.__signalCleanup = cleanup;
  return () => {
    cleanup();
    if (root.__signalCleanup === cleanup) delete root.__signalCleanup;
  };
}
