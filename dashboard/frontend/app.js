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
  graphsage: "Jruvika checkpoint",
  transformer: "First 512 tokens"
};
const MODEL_COUNT = Object.keys(MODEL_DETAILS).length;

// Same word definition as the Python explainer (\w+), so weights line up with words.
const WORD = /[\p{L}\p{N}_]+/gu;

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
  if (analysis.agreement === "agree") return `All ${analysis.completed} models agree`;
  if (analysis.agreement === "disagree") {
    return `The models disagree · ${analysis.fake_votes} of ${analysis.completed} say fake`;
  }
  return `${analysis.completed} of ${MODEL_COUNT} models completed`;
}

function setAvailability(card, text) {
  const dot = document.createElement("span");
  dot.className = "status-dot";
  card.querySelector(".availability").replaceChildren(dot, text);
}

function resetCard(card, model) {
  delete card.dataset.label;
  card.querySelector("[data-explanation]").hidden = true;
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
  if (prediction.status === "unavailable") {
    card.dataset.label = "unavailable";
    card.querySelector("[data-prediction]").textContent = "Not connected";
    card.querySelector("[data-message]").textContent = prediction.message;
    setAvailability(card, "Not connected");
    return;
  }
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

function wordSpan(word, weight, maxAbs) {
  const span = document.createElement("span");
  span.className = weight > 0 ? "word word-fake" : "word word-real";
  // Strength scales with |weight| relative to this model's strongest word.
  const strength = Math.min(Math.abs(weight) / maxAbs, 1);
  span.style.setProperty("--strength", strength.toFixed(2));
  if (strength >= 0.6) span.classList.add("word-strong");
  span.title = `${weight > 0 ? "+" : ""}${weight.toFixed(4)} on the fake-class score`;
  span.textContent = word;
  return span;
}

function highlightText(container, text, weights) {
  // Built from text nodes and spans only; submitted text is never parsed as HTML.
  const values = Object.values(weights).map(Math.abs);
  const maxAbs = Math.max(...values, 1e-9);
  const pieces = [];
  let last = 0;
  for (const match of text.matchAll(WORD)) {
    pieces.push(document.createTextNode(text.slice(last, match.index)));
    const weight = weights[match[0].toLowerCase()];
    pieces.push(weight ? wordSpan(match[0], weight, maxAbs) : document.createTextNode(match[0]));
    last = match.index + match[0].length;
  }
  pieces.push(document.createTextNode(text.slice(last)));
  container.replaceChildren(...pieces);
}

function fillCues(list, cues) {
  const items = cues.map(([word, weight]) => {
    const item = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = word;
    const value = document.createElement("small");
    value.textContent = `${weight > 0 ? "+" : ""}${weight.toFixed(3)}`;
    item.append(label, value);
    return item;
  });
  if (!items.length) {
    const empty = document.createElement("li");
    empty.className = "cue-empty";
    empty.textContent = "No clear words";
    items.push(empty);
  }
  list.replaceChildren(...items);
}

function renderExplanation(card, explanation) {
  const area = card.querySelector("[data-explanation]");
  const message = card.querySelector("[data-explanation-message]");
  area.hidden = false;
  const ready = explanation.status === "ready";
  const weak = ready && explanation.weak_evidence;
  message.hidden = ready && !weak;
  message.textContent = !ready
    ? explanation.message
    : weak
      ? "Very confident: removing any single word barely changes this score, so no word stands out."
      : "";
  for (const selector of ["[data-highlight]", ".cue-lists", "[data-explanation-meta]"]) {
    card.querySelector(selector).hidden = !ready;
  }
  if (!ready) return;

  highlightText(card.querySelector("[data-highlight]"), explanation.text, explanation.weights);
  fillCues(card.querySelector("[data-cues-fake]"), explanation.top_fake);
  fillCues(card.querySelector("[data-cues-real]"), explanation.top_real);
  const shortened = explanation.shortened ? ` · first ${explanation.max_words} words` : "";
  card.querySelector("[data-explanation-meta]").textContent =
    `LIME · ${explanation.samples} samples${shortened} · ${formatDuration(explanation.elapsed_ms)}`;
}

function downloadResults(analysis, explanation) {
  if (!analysis) return;

  const result = {
    ...analysis,
    explanation: explanation?.text === analysis.text ? explanation : null,
    note: "Uncalibrated model outputs, not factual verification. LIME weights show which words a model relied on."
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
    explain: find("#explain-button"),
    explainLabel: find("#explain-button-label"),
    legend: find("#explanation-legend"),
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
  // An explanation is shown only while it belongs to the analyzed text.
  const explanation =
    data.analysis && data.explanation?.text === data.analysis.text ? data.explanation : null;
  const explainable = Boolean(data.analysis?.predictions.some(p => p.status === "ready"));

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
    elements.explain.disabled =
      Boolean(state.pending) || changed || !explainable || Boolean(explanation);
    if (!state.pending) {
      elements.status.textContent = changed
        ? "Text changed · analyze again to update"
        : analysisStatus(data.analysis);
    }
  }

  function updateBusyState() {
    const busy = Boolean(state.pending);
    const clearing = state.pending?.action === "reset";
    const explaining = state.pending?.action === "explain";
    root.classList.toggle("is-busy", busy);
    elements.form.setAttribute("aria-busy", String(busy));
    elements.results.setAttribute("aria-busy", String(busy));
    elements.input.readOnly = busy;
    elements.submit.disabled = busy;
    elements.clear.disabled = busy;
    elements.export.disabled = busy || !data.analysis;
    elements.explain.disabled = busy || !explainable || Boolean(explanation) ||
      root.classList.contains("input-changed");
    elements.explainLabel.textContent = explaining
      ? "Explaining…"
      : explanation ? "Explained" : "Explain predictions";
    elements.submitLabel.textContent = busy ? "Please wait…" : "Analyze text";
    exampleButtons.forEach(button => { button.disabled = busy; });

    if (busy) {
      elements.status.textContent = clearing
        ? "Clearing analysis…"
        : explaining
          ? "Explaining with LIME · this can take up to a minute…"
          : "Analyzing with BiLSTM, GraphSAGE and RoBERTa…";
    }
  }

  function renderResults() {
    for (const model of Object.keys(MODEL_DETAILS)) {
      resetCard(find(`[data-model="${model}"]`), model);
    }
    elements.submittedText.hidden = !data.analysis;
    elements.submittedText.textContent = "";
    elements.summary.textContent = `${MODEL_COUNT} models available`;
    elements.legend.hidden = !explanation;
    if (!data.analysis) return;

    for (const prediction of data.analysis.predictions) {
      if (prediction.model in MODEL_DETAILS) {
        renderPrediction(find(`[data-model="${prediction.model}"]`), prediction);
      }
    }
    if (explanation) {
      for (const item of explanation.explanations) {
        if (item.model in MODEL_DETAILS) {
          renderExplanation(find(`[data-model="${item.model}"]`), item);
        }
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
  on(elements.export, "click", () => downloadResults(data.analysis, explanation));
  on(elements.explain, "click", () => sendRequest("explain"));

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
