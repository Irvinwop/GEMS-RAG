(function () {
  "use strict";

  const app = {
    state: null,
    selectedRetrievers: new Set(),
    selectedModels: new Set(),
    selectedContexts: new Set(["injected", "tool_native"]),
    currentConfigPath: null,
    exactPlan: null,
    activeJobId: null,
    pollTimer: null,
    initializedSelections: false
  };

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const icon = (name) => window.GemsIcons.icon(name);

  document.addEventListener("DOMContentLoaded", init);

  async function init() {
    window.GemsIcons.hydrate();
    bindStaticEvents();
    await refreshState();
  }

  function bindStaticEvents() {
    $$(".nav-item").forEach((button) => button.addEventListener("click", () => showView(button.dataset.view)));
    $("#refresh-state").addEventListener("click", refreshState);
    $("#refresh-manual").addEventListener("click", refreshState);
    $("#refresh-runs").addEventListener("click", refreshState);
    $("#retriever-filter").addEventListener("input", filterRetrievers);
    $("#select-manuscript").addEventListener("click", selectManuscriptRetrievers);
    $("#clear-retrievers").addEventListener("click", () => {
      app.selectedRetrievers.clear();
      syncRetrieverChecks();
      markDraft();
    });
    $("#select-small-models").addEventListener("click", selectSmallModels);
    $("#clear-models").addEventListener("click", () => {
      app.selectedModels.clear();
      syncModelChecks();
      markDraft();
    });
    ["experiment-name", "qa-limit", "top-k", "evidence-chars", "dry-run"].forEach((id) => {
      $("#" + id).addEventListener("input", markDraft);
      $("#" + id).addEventListener("change", markDraft);
    });
    $$('input[name="ingestion"]').forEach((input) => input.addEventListener("change", () => {
      updateIngestionNote();
      markDraft();
    }));
    $$('input[name="grader-mode"]').forEach((input) => input.addEventListener("change", () => {
      updateGraderMode();
      markDraft();
    }));
    $("#grader-select").addEventListener("change", markDraft);
    $("#materialize-config").addEventListener("click", () => materialize(true));
    $("#prepare-indexes").addEventListener("click", () => startJob("external_indexes"));
    $("#preflight-run").addEventListener("click", () => startJob("preflight"));
    $("#launch-run").addEventListener("click", () => startJob("run"));
    $("#rail-preflight").addEventListener("click", () => startJob("preflight"));
    $("#rail-indexes").addEventListener("click", () => startJob("external_indexes"));
    $("#rail-run").addEventListener("click", () => startJob("run"));
    $("#clear-console").addEventListener("click", () => {
      $("#job-output").textContent = "Ready.";
      $("#console-job-label").textContent = "No active job";
    });
    $("#import-grades").addEventListener("click", importGrades);
  }

  async function refreshState() {
    setBusy($("#refresh-state"), true);
    try {
      app.state = await api("/api/state");
      initializeSelections();
      renderState();
      document.querySelector("#app").dataset.loading = "false";
    } catch (error) {
      toast(error.message, "error");
    } finally {
      setBusy($("#refresh-state"), false);
    }
  }

  function initializeSelections() {
    if (app.initializedSelections || !app.state) return;
    const retrieverNames = new Set(app.state.catalogs.retrievers.map((entry) => entry.name));
    ["bm25", "qdrant_hash_vector_command", "gems_full", "canonical_rag_dpr"].forEach((name) => {
      if (retrieverNames.has(name)) app.selectedRetrievers.add(name);
    });
    const answerModels = app.state.catalogs.models.filter((entry) => entry.metadata.roles.includes("answer"));
    answerModels.forEach((entry) => {
      if (entry.metadata.enabled && ["tiny", "small"].includes(entry.metadata.size)) {
        app.selectedModels.add(modelId(entry));
      }
    });
    if (!app.selectedModels.size && answerModels.length) app.selectedModels.add(modelId(answerModels[0]));
    app.initializedSelections = true;
  }

  function renderState() {
    renderHeader();
    renderRetrievers();
    renderContexts();
    renderModels();
    renderGraders();
    renderManual();
    renderCredentials();
    renderRuns();
    renderJobs();
    updateIngestionNote();
    updateGraderMode();
    updatePlan();
    window.GemsIcons.hydrate();
  }

  function renderHeader() {
    const manual = app.state.manual;
    const manualChip = $("#manual-status");
    manualChip.className = `status-chip ${manual.status === "ready" ? "" : "error"}`.trim();
    manualChip.innerHTML = `<span class="status-dot"></span>${manual.status === "ready" ? "Manual verified" : "Manual incomplete"}`;
    $("#workspace-path").textContent = app.state.project.root;
    const secretRows = app.state.credentials.filter((row) => row.kind === "secret");
    const configured = secretRows.filter((row) => row.configured).length;
    $("#sidebar-summary").innerHTML = `
      <div><span>Methods</span><strong>${formatNumber(manual.ingestion.method_count)}</strong></div>
      <div><span>Retrievers</span><strong>${formatNumber(app.state.catalogs.retrievers.length)}</strong></div>
      <div><span>Keys set</span><strong>${configured}/${secretRows.length}</strong></div>
      <div><span>Runs</span><strong>${formatNumber(app.state.runs.length)}</strong></div>`;
  }

  function renderRetrievers() {
    const root = $("#retriever-groups");
    const groups = groupBy(app.state.catalogs.retrievers, (entry) => entry.metadata.family || entry.kind);
    const nativeFamilies = new Set(app.state.manual.ingestion.native_families || []);
    root.innerHTML = Object.entries(groups).map(([family, entries]) => {
      const selected = entries.filter((entry) => app.selectedRetrievers.has(entry.name)).length;
      const shouldOpen = selected > 0 || ["local", "gems_rag"].includes(family);
      return `<details class="catalog-group" data-family="${escapeAttribute(family)}" ${shouldOpen ? "open" : ""}>
        <summary>
          <span class="group-name">${escapeHtml(familyLabel(family))}</span>
          <span class="group-meta">${selected}/${entries.length}${icon("chevron-down")}</span>
        </summary>
        <div class="catalog-items">
          ${entries.map((entry) => {
            const meta = entry.metadata;
            const native = nativeFamilies.has(meta.family);
            return `<label class="catalog-row" data-search="${escapeAttribute([entry.name, meta.family, ...(meta.tags || []), ...(meta.modes || [])].join(" ").toLowerCase())}">
              <input type="checkbox" data-retriever="${escapeAttribute(entry.name)}" ${app.selectedRetrievers.has(entry.name) ? "checked" : ""}>
              <span class="catalog-copy">
                <span class="catalog-title">${escapeHtml(humanize(entry.name))}</span>
                <span class="catalog-meta">${escapeHtml((meta.modes || []).join(" · ") || entry.kind)}</span>
              </span>
              ${native ? '<span class="tag native">PDF</span>' : ""}
            </label>`;
          }).join("")}
        </div>
      </details>`;
    }).join("");
    $$('[data-retriever]', root).forEach((input) => input.addEventListener("change", () => {
      input.checked ? app.selectedRetrievers.add(input.dataset.retriever) : app.selectedRetrievers.delete(input.dataset.retriever);
      markDraft();
      renderRetrieverCounts();
    }));
    root.classList.remove("loading-lines");
    renderRetrieverCounts();
  }

  function renderRetrieverCounts() {
    $("#retriever-count").textContent = `${app.selectedRetrievers.size} selected`;
    $$(".catalog-group").forEach((group) => {
      const checks = $$('[data-retriever]', group);
      const selected = checks.filter((input) => input.checked).length;
      const meta = $(".group-meta", group);
      if (meta) meta.innerHTML = `${selected}/${checks.length}${icon("chevron-down")}`;
    });
    updatePlan();
  }

  function syncRetrieverChecks() {
    $$('[data-retriever]').forEach((input) => { input.checked = app.selectedRetrievers.has(input.dataset.retriever); });
    renderRetrieverCounts();
  }

  function filterRetrievers() {
    const query = $("#retriever-filter").value.trim().toLowerCase();
    $$(".catalog-group").forEach((group) => {
      let visible = 0;
      $$(".catalog-row", group).forEach((row) => {
        const match = !query || row.dataset.search.includes(query);
        row.classList.toggle("hidden-row", !match);
        if (match) visible += 1;
      });
      group.hidden = visible === 0;
      if (query && visible) group.open = true;
    });
  }

  function selectManuscriptRetrievers() {
    app.selectedRetrievers.clear();
    app.state.manual.ingestion.methods.forEach((method) => method.retrievers.forEach((retriever) => app.selectedRetrievers.add(retriever.name)));
    syncRetrieverChecks();
    markDraft();
  }

  function renderContexts() {
    const descriptions = {
      injected: "Automatic evidence payload",
      tool_explore: "Choose and open retrieved hits",
      tool_search: "Generate searches, then open hits",
      tool_native: "Provider function calls"
    };
    $("#context-options").innerHTML = app.state.context_modes.map((mode) => `<label class="mode-option">
      <input type="checkbox" data-context="${escapeAttribute(mode.name)}" ${app.selectedContexts.has(mode.name) ? "checked" : ""}>
      <span><strong>${escapeHtml(mode.label)}</strong><small>${escapeHtml(descriptions[mode.name] || "")}</small></span>
    </label>`).join("");
    $$('[data-context]').forEach((input) => input.addEventListener("change", () => {
      input.checked ? app.selectedContexts.add(input.dataset.context) : app.selectedContexts.delete(input.dataset.context);
      markDraft();
      updateContextCount();
    }));
    updateContextCount();
  }

  function updateContextCount() {
    $("#context-count").textContent = `${app.selectedContexts.size} selected`;
    updatePlan();
  }

  function renderModels() {
    const answerModels = app.state.catalogs.models.filter((entry) => entry.metadata.roles.includes("answer"));
    const groups = groupBy(answerModels, (entry) => entry.provider);
    $("#model-table").innerHTML = Object.entries(groups).map(([provider, entries]) => `<section class="model-provider">
      <h3>${escapeHtml(providerLabel(provider))}</h3>
      ${entries.map((entry) => `<label class="model-row">
        <input type="checkbox" data-model="${escapeAttribute(modelId(entry))}" ${app.selectedModels.has(modelId(entry)) ? "checked" : ""}>
        <span class="model-copy">
          <span class="model-name">${escapeHtml(entry.model)}</span>
          <span class="model-meta">${escapeHtml((entry.metadata.tags || []).slice(0, 3).join(" · "))}</span>
        </span>
        <span class="model-size">${escapeHtml(entry.metadata.size)}</span>
      </label>`).join("")}
    </section>`).join("");
    $$('[data-model]').forEach((input) => input.addEventListener("change", () => {
      input.checked ? app.selectedModels.add(input.dataset.model) : app.selectedModels.delete(input.dataset.model);
      markDraft();
      updateModelCount();
    }));
    $("#model-table").classList.remove("loading-lines");
    updateModelCount();
  }

  function updateModelCount() {
    $("#model-count").textContent = `${app.selectedModels.size} selected`;
    updatePlan();
  }

  function syncModelChecks() {
    $$('[data-model]').forEach((input) => { input.checked = app.selectedModels.has(input.dataset.model); });
    updateModelCount();
  }

  function selectSmallModels() {
    app.selectedModels.clear();
    app.state.catalogs.models.forEach((entry) => {
      if (entry.metadata.roles.includes("answer") && entry.metadata.enabled && ["tiny", "small"].includes(entry.metadata.size)) {
        app.selectedModels.add(modelId(entry));
      }
    });
    syncModelChecks();
    markDraft();
  }

  function renderGraders() {
    const graders = app.state.catalogs.models.filter((entry) => entry.metadata.roles.includes("grader"));
    $("#grader-select").innerHTML = graders.map((entry) => `<option value="${escapeAttribute(modelId(entry))}">${escapeHtml(providerLabel(entry.provider))} · ${escapeHtml(entry.model)}</option>`).join("");
  }

  function updateGraderMode() {
    const mode = selectedRadio("grader-mode");
    $("#grader-select-field").classList.toggle("hidden", mode !== "api");
    const note = $("#grader-note");
    if (mode === "gpt_pro") {
      note.dataset.icon = "archive";
      note.innerHTML = `${icon("archive")}Answers run without paid judge calls; export a grading ZIP from Runs.`;
    } else if (mode === "api") {
      note.dataset.icon = "key-round";
      note.innerHTML = `${icon("key-round")}One judge call is added for every run row.`;
    } else {
      note.dataset.icon = "check";
      note.innerHTML = `${icon("check")}Deterministic local scores; no judge model calls.`;
    }
    updatePlan();
  }

  function updateIngestionNote() {
    const mode = selectedRadio("ingestion");
    const note = $("#ingestion-note");
    note.innerHTML = mode === "native_pdf"
      ? `${icon("file-text")}PaperQA2 and RAG-Anything switch to raw-PDF parsing. MegaRAG and VisRAG remain native; other methods retain verified derivatives.`
      : `${icon("info")}One canonical corpus derived from the verified PDF for controlled comparisons.`;
  }

  function updatePlan() {
    if (!app.state) return;
    const qa = clampInteger($("#qa-limit").value, 1, 100000, 0);
    const retrievers = app.selectedRetrievers.size;
    const contexts = app.selectedContexts.size;
    const models = app.selectedModels.size;
    const conditions = retrievers * contexts * models;
    const rows = qa * conditions;
    let answerCalls = 0;
    const callsByContext = { injected: 1, tool_explore: 2, tool_search: 3, tool_native: 5 };
    app.selectedContexts.forEach((mode) => { answerCalls += qa * retrievers * models * (callsByContext[mode] || 1); });
    const graderMode = selectedRadio("grader-mode");
    const judgeCalls = graderMode === "api" ? rows : 0;
    const paidCalls = $("#dry-run").checked ? 0 : answerCalls + judgeCalls;
    setText("plan-rows", rows);
    setText("plan-qa", qa);
    setText("plan-conditions", conditions);
    setText("plan-answer-calls", answerCalls);
    setText("plan-judge-calls", judgeCalls);
    setText("plan-paid-calls", paidCalls);
    setText("plan-retrievers", retrievers);
    setText("plan-contexts", contexts);
    setText("plan-models", models);
    renderPlanWarnings();
    const valid = retrievers > 0 && contexts > 0 && models > 0;
    ["materialize-config", "prepare-indexes", "preflight-run", "launch-run", "rail-indexes", "rail-preflight", "rail-run"].forEach((id) => { $("#" + id).disabled = !valid; });
  }

  function renderPlanWarnings() {
    const warnings = [];
    if (!app.selectedRetrievers.size) warnings.push("No retrievers selected.");
    if (!app.selectedContexts.size) warnings.push("No context modes selected.");
    if (!app.selectedModels.size) warnings.push("No answer models selected.");
    const missing = requiredUnsetCredentials();
    if (missing.length && !$("#dry-run").checked) warnings.push(`${missing.map((row) => row.label).join(", ")} credentials are unset.`);
    if (selectedRadio("grader-mode") === "gpt_pro") warnings.push("Final judge scores arrive through a GPT Pro bundle import.");
    $("#plan-warnings").innerHTML = warnings.map((warning) => `<div class="plan-warning">${icon("info")}<span>${escapeHtml(warning)}</span></div>`).join("");
  }

  function requiredUnsetCredentials() {
    const providerToEnv = {
      openai: "OPENAI_API_KEY",
      anthropic: "ANTHROPIC_API_KEY",
      xai: "XAI_API_KEY",
      grok: "XAI_API_KEY",
      qwen: "DASHSCOPE_API_KEY"
    };
    const names = new Set();
    app.selectedModels.forEach((id) => {
      const provider = id.split(":", 1)[0];
      if (providerToEnv[provider]) names.add(providerToEnv[provider]);
    });
    if (selectedRadio("grader-mode") === "api") {
      const provider = $("#grader-select").value.split(":", 1)[0];
      if (providerToEnv[provider]) names.add(providerToEnv[provider]);
    }
    return app.state.credentials.filter((row) => names.has(row.name) && !row.configured);
  }

  function markDraft() {
    app.currentConfigPath = null;
    app.exactPlan = null;
    $("#plan-state").textContent = "Draft";
    $("#plan-state").className = "status-chip neutral";
    updatePlan();
  }

  function collectPayload() {
    return {
      name: $("#experiment-name").value,
      limit: clampInteger($("#qa-limit").value, 1, 100000, 12),
      top_k: clampInteger($("#top-k").value, 1, 100, 6),
      max_evidence_chars: clampInteger($("#evidence-chars").value, 100, 100000, 1600),
      ingestion_mode: selectedRadio("ingestion"),
      retrievers: Array.from(app.selectedRetrievers),
      context_modes: Array.from(app.selectedContexts),
      models: Array.from(app.selectedModels),
      grader_mode: selectedRadio("grader-mode"),
      grader: $("#grader-select").value,
      dry_run: $("#dry-run").checked
    };
  }

  async function materialize(showToast) {
    const buttons = [$("#materialize-config")];
    buttons.forEach((button) => setBusy(button, true));
    try {
      const result = await api("/api/configs", { method: "POST", body: collectPayload() });
      app.currentConfigPath = result.config_path;
      app.exactPlan = result.plan;
      applyExactPlan(result.plan);
      $("#plan-state").textContent = "Materialized";
      $("#plan-state").className = "status-chip";
      if (showToast) toast(`Config written: ${shortPath(result.config_path)}`);
      return result;
    } finally {
      buttons.forEach((button) => setBusy(button, false));
    }
  }

  function applyExactPlan(plan) {
    setText("plan-rows", plan.estimates.rows);
    setText("plan-qa", plan.dataset.qa_count);
    setText("plan-conditions", plan.dimensions.conditions);
    setText("plan-answer-calls", plan.estimates.answer_model_calls);
    setText("plan-judge-calls", plan.estimates.judge_model_calls);
    setText("plan-paid-calls", plan.estimates.paid_model_calls);
    setText("plan-retrievers", plan.dimensions.retrievers);
    setText("plan-contexts", plan.dimensions.context_modes);
    setText("plan-models", plan.dimensions.models);
  }

  async function startJob(action) {
    const actionButtons = action === "run"
      ? [$("#launch-run"), $("#rail-run")]
      : action === "external_indexes"
        ? [$("#prepare-indexes"), $("#rail-indexes")]
        : [$("#preflight-run"), $("#rail-preflight")];
    actionButtons.forEach((button) => setBusy(button, true));
    try {
      const config = app.currentConfigPath ? { config_path: app.currentConfigPath } : await materialize(false);
      const job = await api("/api/jobs", {
        method: "POST",
        body: {
          action,
          config_path: config.config_path,
          run_mode: "resume",
          external_checks: false,
          ingestion_mode: selectedRadio("ingestion")
        }
      });
      app.activeJobId = job.id;
      $("#job-output").textContent = "Queued.\n";
      $("#console-job-label").textContent = `${action} · ${job.id}`;
      toast(`${humanize(action)} job queued.`);
      schedulePoll(100);
    } catch (error) {
      toast(error.message, "error");
    } finally {
      actionButtons.forEach((button) => setBusy(button, false));
    }
  }

  function schedulePoll(delay = 1200) {
    clearTimeout(app.pollTimer);
    app.pollTimer = setTimeout(pollActiveJob, delay);
  }

  async function pollActiveJob() {
    if (!app.activeJobId) return;
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(app.activeJobId)}`);
      renderActiveJob(job);
      if (["queued", "running"].includes(job.status)) {
        schedulePoll();
      } else {
        app.activeJobId = null;
        toast(`${humanize(job.action)} ${job.status}.`, job.status === "complete" ? "" : "error");
        await refreshState();
      }
    } catch (error) {
      toast(error.message, "error");
      schedulePoll(2500);
    }
  }

  function renderJobs() {
    const jobs = app.state.jobs || [];
    const active = jobs.find((job) => ["queued", "running"].includes(job.status));
    if (active) {
      app.activeJobId = active.id;
      renderActiveJob(active);
      schedulePoll();
    } else {
      const chip = $("#active-job-status");
      chip.className = "status-chip neutral";
      chip.innerHTML = '<span class="status-dot"></span>Idle';
    }
  }

  function renderActiveJob(job) {
    const chip = $("#active-job-status");
    chip.className = `status-chip ${job.status === "failed" ? "error" : job.status === "complete" ? "" : "warning"}`.trim();
    chip.innerHTML = `<span class="status-dot"></span>${escapeHtml(humanize(job.action))} · ${escapeHtml(job.status)}`;
    $("#console-job-label").textContent = `${humanize(job.action)} · ${job.id}`;
    const output = $("#job-output");
    output.textContent = (job.logs || []).join("\n") || `${humanize(job.status)}.\n`;
    output.scrollTop = output.scrollHeight;
  }

  function renderManual() {
    const report = app.state.manual;
    const manual = report.manual;
    const artifacts = report.artifacts;
    setText("manual-page-total", manual.pages || 0);
    $("#manual-document-title").textContent = manual.title;
    $("#manual-author").textContent = manual.author;
    $("#manual-hash").textContent = `sha256:${manual.sha256}`;
    const badge = $("#manual-ready-badge");
    badge.textContent = report.status === "ready" ? "Verified source" : "Audit failed";
    badge.className = `status-chip ${report.status === "ready" ? "" : "error"}`.trim();
    const metrics = [
      [manual.pages, "PDF pages"],
      [artifacts.raw_chunks, "Raw chunks"],
      [artifacts.canonical_chunks, "Canonical chunks"],
      [artifacts.page_images, "Page renders"],
      [artifacts.figure_records, "Figure records"],
      [formatBytes(manual.bytes), "PDF size"]
    ];
    $("#manual-metrics").innerHTML = metrics.map(([value, label]) => `<div class="audit-metric"><strong>${escapeHtml(formatNumber(value))}</strong><span>${escapeHtml(label)}</span></div>`).join("");
    $("#manual-checks").innerHTML = report.checks.map((check) => `<div class="check-item">
      <span class="check-icon">${icon(check.ok ? "check" : "x")}</span>
      <span><strong>${escapeHtml(humanize(check.name))}</strong><small title="${escapeAttribute(check.detail)}">${escapeHtml(check.detail)}</small></span>
    </div>`).join("");
    $("#native-method-count").textContent = `${report.ingestion.native_method_count} methods with native document paths`;
    $("#manual-matrix-body").innerHTML = report.ingestion.methods.map((method) => {
      const native = method.retrievers.filter((retriever) => retriever.native);
      return `<tr>
        <td><span class="method-name">${escapeHtml(method.label)}</span><span class="method-id">${escapeHtml(method.method_id)}</span></td>
        <td>${escapeHtml(method.retrievers.map((retriever) => retriever.name).join(", ") || "Catalog reference")}</td>
        <td><span class="tag">Verified</span> PDF derivative</td>
        <td>${native.length ? `<span class="native-path">${escapeHtml(native.map((retriever) => retriever.native.label).join("; "))}</span>` : '<span class="muted">Shared corpus only</span>'}</td>
      </tr>`;
    }).join("");
  }

  function renderCredentials() {
    const rows = app.state.credentials;
    const configured = rows.filter((row) => row.configured).length;
    setText("credential-configured-count", configured);
    setText("credential-unset-count", rows.length - configured);
    $("#credential-list").innerHTML = rows.map((row) => `<div class="credential-row" data-credential-row="${escapeAttribute(row.name)}">
      <div class="credential-name"><strong>${escapeHtml(row.label)}</strong><span>${escapeHtml(row.name)}</span></div>
      <span class="status-chip ${row.configured ? "" : "neutral"}"><span class="status-dot"></span>${row.configured ? escapeHtml(humanize(row.source)) : "Unset"}</span>
      <label class="credential-input">
        <span class="sr-only">${escapeHtml(row.label)} value</span>
        <input type="${row.kind === "secret" ? "password" : "url"}" autocomplete="off" placeholder="${row.configured ? "Replace configured value" : row.kind === "secret" ? "Paste API key" : "https://..."}">
      </label>
      <div class="credential-actions">
        <button class="icon-button save-credential" type="button" title="Save ${escapeAttribute(row.label)}" aria-label="Save ${escapeAttribute(row.label)}" data-icon="save"></button>
        <button class="icon-button danger clear-credential" type="button" title="Clear ${escapeAttribute(row.label)}" aria-label="Clear ${escapeAttribute(row.label)}" data-icon="trash-2" ${row.configured ? "" : "disabled"}></button>
      </div>
    </div>`).join("");
    $("#credential-list").classList.remove("loading-lines");
    $$(".save-credential").forEach((button) => button.addEventListener("click", () => saveCredential(button.closest("[data-credential-row]"))));
    $$(".clear-credential").forEach((button) => button.addEventListener("click", () => clearCredential(button.closest("[data-credential-row]"))));
    window.GemsIcons.hydrate($("#credential-list"));
  }

  async function saveCredential(row) {
    const input = $("input", row);
    const button = $(".save-credential", row);
    if (!input.value.trim()) {
      toast("Enter a value before saving.", "error");
      input.focus();
      return;
    }
    setBusy(button, true);
    try {
      await api("/api/credentials", { method: "POST", body: { name: row.dataset.credentialRow, value: input.value } });
      input.value = "";
      toast(`${row.dataset.credentialRow} configured.`);
      await refreshState();
    } catch (error) {
      toast(error.message, "error");
    } finally {
      setBusy(button, false);
    }
  }

  async function clearCredential(row) {
    const button = $(".clear-credential", row);
    setBusy(button, true);
    try {
      await api("/api/credentials/clear", { method: "POST", body: { name: row.dataset.credentialRow } });
      toast(`${row.dataset.credentialRow} cleared.`);
      await refreshState();
    } catch (error) {
      toast(error.message, "error");
    } finally {
      setBusy(button, false);
    }
  }

  function renderRuns() {
    const runs = app.state.runs;
    $("#run-table-body").innerHTML = runs.length ? runs.map((run) => `<tr>
      <td><span class="run-name">${escapeHtml(run.name)}</span><span class="run-path">${escapeHtml(shortPath(run.path))}</span></td>
      <td>${formatNumber(run.rows)}</td>
      <td>${escapeHtml(formatDate(run.modified_at))}</td>
      <td>${escapeHtml(formatBytes(run.bytes))}</td>
      <td>${run.has_gpt_pro_grades ? '<span class="tag native">GPT Pro</span>' : '<span class="muted">Unimported</span>'}</td>
      <td class="align-right"><div class="run-actions">
        <button class="button secondary bundle-action" type="button" data-mode="archive" data-runs="${escapeAttribute(run.path)}" data-icon="archive">Archive</button>
        <button class="button primary bundle-action" type="button" data-mode="gpt_pro" data-runs="${escapeAttribute(run.path)}" data-icon="download">GPT Pro ZIP</button>
      </div></td>
    </tr>`).join("") : '<tr><td colspan="6" class="muted">No run outputs found.</td></tr>';
    $("#import-run").innerHTML = runs.map((run) => `<option value="${escapeAttribute(run.path)}">${escapeHtml(run.name)}</option>`).join("");
    $$(".bundle-action").forEach((button) => button.addEventListener("click", () => createBundle(button)));
    window.GemsIcons.hydrate($("#view-runs"));
  }

  async function createBundle(button) {
    setBusy(button, true);
    try {
      const result = await api("/api/bundles", { method: "POST", body: { runs: button.dataset.runs, mode: button.dataset.mode } });
      showBundleResult(result);
      toast(`${humanize(button.dataset.mode)} ZIP created.`);
    } catch (error) {
      toast(error.message, "error");
    } finally {
      setBusy(button, false);
    }
  }

  function showBundleResult(result) {
    const root = $("#bundle-result");
    root.classList.remove("hidden");
    root.innerHTML = `<span>${escapeHtml(shortPath(result.output))} · ${escapeHtml(formatBytes(result.bytes))} · ${formatNumber(result.grading_tasks)} tasks</span>
      <a href="/api/download?path=${encodeURIComponent(result.output)}">${icon("download")}Download ZIP</a>`;
  }

  async function importGrades() {
    const runs = $("#import-run").value;
    const file = $("#import-grade-file").files[0];
    if (!runs || !file) {
      toast("Select a run and completed grades file.", "error");
      return;
    }
    const button = $("#import-grades");
    setBusy(button, true);
    try {
      if (file.size > 20 * 1024 * 1024) throw new Error("Grades file must be 20 MB or smaller.");
      const result = await api("/api/import-grades", {
        method: "POST",
        body: { runs, grades_filename: file.name, grades_base64: await fileToBase64(file) }
      });
      toast(`${formatNumber(result.grades_imported)} grades imported${result.grades_missing ? `; ${result.grades_missing} missing` : ""}.`, result.ok ? "" : "error");
      await refreshState();
    } catch (error) {
      toast(error.message, "error");
    } finally {
      setBusy(button, false);
    }
  }

  function showView(name) {
    $$(".nav-item").forEach((button) => button.classList.toggle("active", button.dataset.view === name));
    $$(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${name}`));
    window.scrollTo({ top: 0, behavior: "smooth" });
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

  function setBusy(button, busy) {
    if (!button) return;
    button.disabled = busy;
    button.setAttribute("aria-busy", String(busy));
  }

  function toast(message, type = "") {
    const element = document.createElement("div");
    element.className = `toast ${type}`.trim();
    element.textContent = message;
    $("#toast-region").appendChild(element);
    setTimeout(() => element.remove(), 4800);
  }

  function selectedRadio(name) {
    return $(`input[name="${name}"]:checked`)?.value || "";
  }

  function setText(id, value) {
    $("#" + id).textContent = formatNumber(value);
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

  function familyLabel(value) {
    const labels = {
      local: "Local baselines",
      local_vector_db: "Vector database",
      gems_rag: "GEMS-RAG ablations",
      self_rag_policy: "Self-RAG",
      crag_policy: "CRAG",
      paperqa2: "PaperQA2"
    };
    return labels[value] || humanize(value);
  }

  function providerLabel(value) {
    const labels = { openai: "OpenAI", anthropic: "Anthropic", xai: "xAI", qwen: "Qwen", local_openai: "Local", litellm: "LiteLLM" };
    return labels[value] || humanize(value);
  }

  function humanize(value) {
    return String(value || "").replaceAll("_", " ").replaceAll("-", " ").replace(/\b\w/g, (character) => character.toUpperCase());
  }

  function clampInteger(value, minimum, maximum, fallback) {
    const number = Number.parseInt(value, 10);
    return Number.isFinite(number) ? Math.min(maximum, Math.max(minimum, number)) : fallback;
  }

  function formatNumber(value) {
    if (typeof value === "string") return value;
    return Number(value || 0).toLocaleString("en-US");
  }

  function formatBytes(value) {
    const bytes = Number(value || 0);
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function formatDate(value) {
    const date = new Date(value);
    return Number.isNaN(date.valueOf()) ? "Unknown" : date.toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
  }

  function shortPath(value) {
    const root = app.state?.project?.root;
    return root && String(value).startsWith(root) ? String(value).slice(root.length + 1) : String(value);
  }

  function fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error("Could not read grades file."));
      reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
      reader.readAsDataURL(file);
    });
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[character]);
  }

  function escapeAttribute(value) {
    return escapeHtml(value).replaceAll("`", "&#96;");
  }
})();
