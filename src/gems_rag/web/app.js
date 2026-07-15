(function () {
  "use strict";

  const STORAGE_KEY = "gems-rag:selected-models";
  const app = {
    state: null,
    models: [],
    selected: readSelection()
  };

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  document.addEventListener("DOMContentLoaded", init);

  async function init() {
    $("#select-all-models").addEventListener("click", selectAllModels);
    $("#clear-models").addEventListener("click", clearModels);
    $("#token-form").addEventListener("submit", saveTokens);

    try {
      app.state = await api("/api/state");
      app.models = uniqueModels(
        app.state.catalogs.models.filter((entry) => (entry.metadata.roles || []).includes("answer"))
      );
      reconcileSelection();
      renderModels();
      renderTokens();
      setPageStatus("Ready");
    } catch (error) {
      setPageStatus("Could not load", true);
      showMessage(error.message, true);
    } finally {
      $("#app").dataset.loading = "false";
    }
  }

  function uniqueModels(entries) {
    const byId = new Map();
    entries.forEach((entry) => {
      const id = modelId(entry);
      const existing = byId.get(id);
      if (existing) {
        existing.roles = Array.from(new Set([...existing.roles, ...(entry.metadata.roles || [])]));
        existing.enabled = existing.enabled || entry.metadata.enabled;
        return;
      }
      byId.set(id, {
        id,
        provider: entry.provider,
        model: entry.model,
        roles: [...(entry.metadata.roles || [])],
        enabled: Boolean(entry.metadata.enabled)
      });
    });
    return Array.from(byId.values());
  }

  function renderModels() {
    const groups = groupBy(app.models, (model) => model.provider);
    $("#model-list").innerHTML = Object.entries(groups).map(([provider, models]) => `
      <fieldset class="provider-group">
        <legend>${escapeHtml(providerLabel(provider))}<span>${models.length}</span></legend>
        <div class="provider-models">
          ${models.map((model) => `
            <label class="model-option">
              <input
                type="checkbox"
                data-model="${escapeAttribute(model.id)}"
                data-roles="${escapeAttribute(model.roles.join(","))}"
                ${app.selected.has(model.id) ? "checked" : ""}
              >
              <span class="model-name">${escapeHtml(model.model)}</span>
            </label>
          `).join("")}
        </div>
      </fieldset>
    `).join("");

    $$("[data-model]").forEach((input) => input.addEventListener("change", () => {
      input.checked ? app.selected.add(input.dataset.model) : app.selected.delete(input.dataset.model);
      persistSelection();
      updateModelSummary();
    }));
    updateModelSummary();
  }

  function renderTokens() {
    const tokens = app.state.credentials.filter((row) => row.kind === "secret");
    const configured = tokens.filter((row) => row.configured).length;
    $("#token-summary").textContent = `${configured} of ${tokens.length} configured`;
    $("#token-list").innerHTML = tokens.map((row) => `
      <div class="token-row" data-token-row="${escapeAttribute(row.name)}">
        <label for="token-${escapeAttribute(row.name)}">
          <strong>${escapeHtml(row.label)}</strong>
          <span>${escapeHtml(row.name)}</span>
        </label>
        <input
          id="token-${escapeAttribute(row.name)}"
          data-token="${escapeAttribute(row.name)}"
          type="password"
          autocomplete="new-password"
          spellcheck="false"
          placeholder="${row.configured ? "Configured; paste to replace" : "Paste API token"}"
        >
        <span class="token-status ${row.configured ? "configured" : ""}">${row.configured ? "Configured" : "Not set"}</span>
        <button
          class="remove-token"
          type="button"
          data-clear-token="${escapeAttribute(row.name)}"
          data-configured="${row.configured ? "true" : "false"}"
          ${row.configured ? "" : "disabled"}
        >Remove</button>
      </div>
    `).join("");

    $$("[data-clear-token]").forEach((button) => button.addEventListener("click", () => clearToken(button)));
  }

  function selectAllModels() {
    app.selected = new Set(app.models.map((model) => model.id));
    syncModelChecks();
  }

  function clearModels() {
    app.selected.clear();
    syncModelChecks();
  }

  function syncModelChecks() {
    $$("[data-model]").forEach((input) => {
      input.checked = app.selected.has(input.dataset.model);
    });
    persistSelection();
    updateModelSummary();
  }

  function updateModelSummary() {
    $("#model-count").textContent = `${app.models.length} ${app.models.length === 1 ? "model" : "models"}`;
    $("#selected-count").textContent = `${app.selected.size} selected`;
  }

  async function saveTokens(event) {
    event.preventDefault();
    const values = $$("[data-token]")
      .map((input) => ({ name: input.dataset.token, value: input.value.trim() }))
      .filter((row) => row.value);

    if (!values.length) {
      showMessage("Enter at least one token to save.", true);
      $$("[data-token]")[0]?.focus();
      return;
    }

    setFormBusy(true);
    try {
      for (const row of values) {
        await api("/api/credentials", { method: "POST", body: row });
      }
      app.state = await api("/api/state");
      renderTokens();
      showMessage(`${values.length} ${values.length === 1 ? "token" : "tokens"} saved.`);
    } catch (error) {
      showMessage(error.message, true);
    } finally {
      setFormBusy(false);
    }
  }

  async function clearToken(button) {
    const name = button.dataset.clearToken;
    if (!window.confirm(`Remove ${name}?`)) return;
    setFormBusy(true);
    try {
      await api("/api/credentials/clear", { method: "POST", body: { name } });
      app.state = await api("/api/state");
      renderTokens();
      showMessage(`${name} removed.`);
    } catch (error) {
      showMessage(error.message, true);
    } finally {
      setFormBusy(false);
    }
  }

  function setFormBusy(busy) {
    $("#save-tokens").disabled = busy;
    $$("[data-clear-token]").forEach((button) => {
      button.disabled = busy || button.dataset.configured !== "true";
    });
    $("#token-form").setAttribute("aria-busy", String(busy));
  }

  function reconcileSelection() {
    const known = new Set(app.models.map((model) => model.id));
    app.selected = new Set(Array.from(app.selected).filter((id) => known.has(id)));
    persistSelection();
  }

  function readSelection() {
    try {
      const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      return new Set(Array.isArray(value) ? value.filter((item) => typeof item === "string") : []);
    } catch {
      return new Set();
    }
  }

  function persistSelection() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(Array.from(app.selected)));
  }

  function setPageStatus(message, error = false) {
    const status = $("#page-status");
    status.textContent = message;
    status.classList.toggle("error", error);
  }

  function showMessage(message, error = false) {
    const root = $("#message");
    root.textContent = message;
    root.classList.toggle("error", error);
  }

  async function api(path, options = {}) {
    const init = { method: options.method || "GET", headers: { Accept: "application/json" } };
    if (options.body !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(options.body);
    }
    const response = await fetch(path, init);
    const payload = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    return payload;
  }

  function groupBy(items, keyFn) {
    return items.reduce((groups, item) => {
      const key = keyFn(item);
      (groups[key] ||= []).push(item);
      return groups;
    }, {});
  }

  function modelId(entry) {
    return `${entry.provider}:${entry.model}`;
  }

  function providerLabel(value) {
    const labels = {
      openai: "OpenAI",
      anthropic: "Anthropic",
      xai: "xAI",
      qwen: "Qwen",
      local_openai: "Local models",
      litellm: "LiteLLM"
    };
    return labels[value] || String(value).replaceAll("_", " ");
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      "'": "&#39;",
      '"': "&quot;"
    })[character]);
  }

  function escapeAttribute(value) {
    return escapeHtml(value).replaceAll("`", "&#96;");
  }
})();
