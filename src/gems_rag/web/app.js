(function () {
  "use strict";

  const STORAGE_KEY = "gems-rag:ablation-setup-v2";
  const LEGACY_MODEL_KEY = "gems-rag:selected-models";
  const DEFAULTS = {
    name: "mutcd-ablation",
    outputDir: "runs",
    zipName: "mutcd-ablation-gpt-pro.zip",
    limit: 12,
    topK: 6,
    evidenceChars: 1600,
    dataset: "mutcd150",
    ingestionMode: "shared_corpus",
    dryRun: false,
    retrievers: ["bm25"],
    contexts: ["injected"],
    models: [],
    configPath: null,
    artifacts: null,
    activeJobId: null,
    dirty: true
  };

  const app = {
    state: null,
    models: [],
    retrievers: [],
    setup: readSetup(),
    selectedModels: new Set(),
    selectedRetrievers: new Set(),
    selectedContexts: new Set(),
    configPath: null,
    artifacts: null,
    activeJobId: null,
    dirty: true,
    exactRows: null,
    runStatus: null,
    activeJob: null,
    pollTimer: null,
    ragAudit: new Map(),
    ragAuditJobId: null
  };

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  document.addEventListener("DOMContentLoaded", init);

  async function init() {
    bindStaticEvents();
    hydrateFields();
    try {
      app.state = await api("/api/state");
      app.models = uniqueModels(
        app.state.catalogs.models.filter((entry) => (entry.metadata.roles || []).includes("answer"))
      );
      app.retrievers = app.state.catalogs.retrievers;
      restoreSelections();
      renderDataset();
      renderRetrievers();
      renderContexts();
      renderModels();
      renderTokens();
      updateSummary();
      restoreRagAudit();
      restoreActiveJob();
      setPageStatus("Ready");
      if (app.configPath && !app.dirty) await refreshRunStatus(true);
      if (app.activeJobId) schedulePoll(100);
    } catch (error) {
      setPageStatus("Could not load", true);
      showMessage(error.message, true);
    } finally {
      $("#app").dataset.loading = "false";
    }
  }

  function bindStaticEvents() {
    ["output-dir", "zip-name", "qa-limit", "top-k", "evidence-chars", "dry-run"].forEach((id) => {
      $("#" + id).addEventListener("input", markDraft);
      $("#" + id).addEventListener("change", markDraft);
    });
    $("#experiment-name").addEventListener("input", updateExperimentName);
    $$('input[name="ingestion"]').forEach((input) => input.addEventListener("change", markDraft));
    $("#select-all-rags").addEventListener("click", selectAllRetrievers);
    $("#clear-rags").addEventListener("click", clearRetrievers);
    $("#test-rags").addEventListener("click", () => startJob("rag_audit"));
    $("#select-all-models").addEventListener("click", selectAllModels);
    $("#clear-models").addEventListener("click", clearModels);
    $("#token-form").addEventListener("submit", saveTokens);
    $("#prepare-rags").addEventListener("click", () => startJob("external_indexes"));
    $("#start-run").addEventListener("click", () => startJob("run"));
    $("#stop-run").addEventListener("click", stopJob);
    $("#export-zip").addEventListener("click", exportZip);
  }

  function hydrateFields() {
    const setup = app.setup;
    $("#experiment-name").value = setup.name;
    $("#experiment-name").dataset.previousName = setup.name;
    $("#output-dir").value = setup.outputDir;
    $("#zip-name").value = setup.zipName;
    $("#qa-limit").value = setup.limit;
    $("#top-k").value = setup.topK;
    $("#evidence-chars").value = setup.evidenceChars;
    $("#dry-run").checked = setup.dryRun;
    const ingestion = $(`input[name="ingestion"][value="${cssEscape(setup.ingestionMode)}"]`);
    (ingestion || $('input[name="ingestion"][value="shared_corpus"]')).checked = true;
    app.configPath = setup.configPath;
    app.artifacts = setup.artifacts;
    app.activeJobId = setup.activeJobId;
    app.dirty = Boolean(setup.dirty);
  }

  function restoreSelections() {
    const knownModels = new Set(app.models.map((model) => model.id));
    const knownRetrievers = new Set(app.retrievers.map((retriever) => retriever.name));
    const knownContexts = new Set(app.state.context_modes.map((context) => context.name));
    app.selectedModels = new Set(app.setup.models.filter((id) => knownModels.has(id)));
    app.selectedRetrievers = new Set(app.setup.retrievers.filter((name) => knownRetrievers.has(name)));
    app.selectedContexts = new Set(app.setup.contexts.filter((name) => knownContexts.has(name)));
    if (!app.selectedRetrievers.size && knownRetrievers.has("bm25")) app.selectedRetrievers.add("bm25");
    if (!app.selectedContexts.size && knownContexts.has("injected")) app.selectedContexts.add("injected");
    reconcileSelectedRetrievers();
    persistSetup();
  }

  function restoreActiveJob() {
    const jobs = app.state.jobs || [];
    let job = app.activeJobId ? jobs.find((entry) => entry.id === app.activeJobId) : null;
    if (!job && app.configPath) {
      job = jobs.find((entry) => entry.config_path === app.configPath && ["queued", "running"].includes(entry.status));
    }
    if (job) {
      app.activeJobId = job.id;
      renderJob(job);
      persistSetup();
    }
  }

  function restoreRagAudit() {
    const job = (app.state.jobs || []).find((entry) => entry.action === "rag_audit" && entry.report);
    if (job) applyRagAudit(job.report, job.id);
  }

  function renderRetrievers() {
    const segments = [
      { key: "query_driven", title: "Interactive RAGs", matches: (entry) => entry.interaction === "query_driven" },
      { key: "fixed_question", title: "Fixed-question plans", matches: (entry) => entry.interaction === "fixed_question" },
      { key: "controls", title: "Controls and upper bounds", matches: (entry) => !["query_driven", "fixed_question"].includes(entry.interaction) }
    ];
    $("#rag-list").innerHTML = segments.map(renderRetrieverSegment).join("");
    $$('[data-retriever]').forEach((input) => input.addEventListener("change", () => {
      input.checked ? app.selectedRetrievers.add(input.dataset.retriever) : app.selectedRetrievers.delete(input.dataset.retriever);
      markDraft();
    }));
    updateSelectionCounts();
  }

  function renderContexts() {
    $("#context-list").innerHTML = app.state.context_modes.map((context) => `
      <label class="check-option context-option">
        <input type="checkbox" data-context="${escapeAttribute(context.name)}" ${app.selectedContexts.has(context.name) ? "checked" : ""}>
        <span>${escapeHtml(context.label)}</span>
      </label>
    `).join("");
    $$('[data-context]').forEach((input) => input.addEventListener("change", () => {
      input.checked ? app.selectedContexts.add(input.dataset.context) : app.selectedContexts.delete(input.dataset.context);
      const removed = reconcileSelectedRetrievers();
      renderRetrievers();
      markDraft();
      if (removed.length) {
        showMessage(`${removed.length} incompatible ${removed.length === 1 ? "RAG was" : "RAGs were"} deselected.`);
      }
    }));
    updateSelectionCounts();
  }

  function renderModels() {
    const groups = groupBy(app.models, (model) => model.provider);
    $("#model-list").innerHTML = Object.entries(groups).map(([provider, models]) => `
      <fieldset class="choice-group">
        <legend>${escapeHtml(providerLabel(provider))}<span>${models.length}</span></legend>
        <div class="choice-grid">
          ${models.map((model) => `
            <label class="check-option">
              <input type="checkbox" data-model="${escapeAttribute(model.id)}" ${app.selectedModels.has(model.id) ? "checked" : ""}>
              <span class="choice-name">${escapeHtml(model.model)}</span>
            </label>
          `).join("")}
        </div>
      </fieldset>
    `).join("");
    $$('[data-model]').forEach((input) => input.addEventListener("change", () => {
      input.checked ? app.selectedModels.add(input.dataset.model) : app.selectedModels.delete(input.dataset.model);
      markDraft();
    }));
    updateSelectionCounts();
  }

  function renderTokens() {
    const tokens = app.state.credentials.filter((row) => row.kind === "secret");
    const configured = tokens.filter((row) => row.configured).length;
    $("#token-summary").textContent = `${configured} of ${tokens.length} configured`;
    $("#token-list").innerHTML = tokens.map((row) => `
      <div class="token-row" data-token-row="${escapeAttribute(row.name)}">
        <label for="token-${escapeAttribute(row.name)}">
          <strong>${escapeHtml(tokenLabel(row))}</strong>
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
    $$('[data-clear-token]').forEach((button) => button.addEventListener("click", () => clearToken(button)));
  }

  function renderDataset() {
    const known = new Set(app.state.datasets.map((dataset) => dataset.id));
    const selected = known.has(app.setup.dataset) ? app.setup.dataset : app.state.default_dataset;
    $("#dataset-list").innerHTML = app.state.datasets.map((dataset) => {
      const recordType = dataset.includes_gold_answers ? "Q/A pairs" : "questions, no gold";
      return `
        <label>
          <input type="radio" name="dataset" value="${escapeAttribute(dataset.id)}"
            ${dataset.id === selected ? "checked" : ""} ${dataset.available ? "" : "disabled"}>
          ${escapeHtml(dataset.label)} (${formatNumber(dataset.qa_count)} ${recordType})
        </label>
      `;
    }).join("");
    $$('input[name="dataset"]').forEach((input) => input.addEventListener("change", () => {
      const removed = reconcileSelectedRetrievers();
      renderRetrievers();
      updateDatasetSource();
      markDraft();
      if (removed.length) {
        showMessage(`${removed.join(", ")} requires gold references and was deselected.`);
      }
    }));
    updateDatasetSource();
  }

  function updateDatasetSource() {
    const dataset = selectedDatasetInfo();
    if (!dataset) return;
    const recordType = dataset.includes_gold_answers ? "Q/A pairs" : "questions (no gold answers)";
    $("#qa-source").textContent = `${formatNumber(dataset.qa_count)} ${recordType} | ${dataset.qa_path}`;
    $("#qa-source").title = dataset.qa_sha256 ? `SHA-256 ${dataset.qa_sha256}` : "Question source unavailable";
  }

  function renderRetrieverSegment(segment) {
    const retrievers = app.retrievers.filter(segment.matches);
    const groups = groupBy(retrievers, (entry) => entry.metadata.family || entry.kind);
    return `
      <section class="rag-segment" data-rag-segment="${escapeAttribute(segment.key)}" data-count="${retrievers.length}">
        <div class="rag-segment-header">
          <h3>${escapeHtml(segment.title)}</h3>
          <span>${retrievers.length}</span>
        </div>
        ${Object.entries(groups).map(([family, entries]) => `
          <fieldset class="choice-group">
            <legend>${escapeHtml(familyLabel(family))}<span>${entries.length}</span></legend>
            <div class="choice-grid">
              ${entries.map(renderRetrieverOption).join("")}
            </div>
          </fieldset>
        `).join("")}
      </section>
    `;
  }

  function renderRetrieverOption(retriever) {
    const compatible = isRetrieverCompatible(retriever);
    const audit = app.ragAudit.get(retriever.name);
    const auditStatus = audit?.status || "untested";
    const title = compatible ? (audit?.problems || []).join("; ") : retrieverIncompatibility(retriever);
    return `
      <label class="check-option rag-option ${compatible ? "" : "incompatible"}" ${title ? `title="${escapeAttribute(title)}"` : ""}>
        <input
          type="checkbox"
          data-retriever="${escapeAttribute(retriever.name)}"
          ${app.selectedRetrievers.has(retriever.name) ? "checked" : ""}
          ${compatible ? "" : "disabled"}
        >
        <span class="choice-copy">
          <span class="choice-name">${escapeHtml(retriever.name)}</span>
          <span class="choice-meta">${escapeHtml(contextModeLabel(retriever.context_modes))}</span>
        </span>
        <span class="audit-status ${escapeAttribute(auditStatus.replaceAll("_", "-"))}">${escapeHtml(auditStatusLabel(auditStatus))}</span>
      </label>
    `;
  }

  function applyRagAudit(report, jobId) {
    app.ragAudit = new Map((report.retrievers || []).map((row) => [row.name, row]));
    app.ragAuditJobId = jobId;
    $("#rag-audit-summary").textContent = auditSummaryText(report.summary);
    renderRetrievers();
  }

  function auditSummaryText(summary) {
    if (!summary) return "Not tested";
    const parts = [`${summary.ready || 0} ready`];
    if (summary.blocked_by_credentials) parts.push(`${summary.blocked_by_credentials} need API keys`);
    if (summary.blocked) parts.push(`${summary.blocked} blocked`);
    if (summary.not_checked) parts.push(`${summary.not_checked} not checked`);
    if (summary.failed) parts.push(`${summary.failed} failed`);
    return parts.join(", ");
  }

  function auditStatusLabel(status) {
    const labels = {
      untested: "not tested",
      ready: "ready",
      blocked: "blocked",
      blocked_by_credentials: "API key",
      not_checked: "not checked",
      failed: "failed"
    };
    return labels[status] || humanize(status);
  }

  function contextModeLabel(modes) {
    const supported = Array.isArray(modes) ? modes : [];
    if (supported.length === 4) return "all 4 modes";
    if (supported.length === 2 && supported.includes("injected") && supported.includes("tool_explore")) {
      return "inject + explore";
    }
    if (supported.length === 1 && supported[0] === "injected") return "inject only";
    const labels = { injected: "inject", tool_explore: "explore", tool_search: "search", tool_native: "native" };
    return supported.map((mode) => labels[mode] || mode).join(" + ");
  }

  function unsupportedContextModes(retriever) {
    const supported = new Set(retriever.context_modes || []);
    return Array.from(app.selectedContexts).filter((mode) => !supported.has(mode));
  }

  function isRetrieverCompatible(retriever) {
    return retrieverIncompatibility(retriever) === "";
  }

  function retrieverIncompatibility(retriever) {
    const unsupported = unsupportedContextModes(retriever);
    if (unsupported.length) return `Does not support: ${unsupported.join(", ")}`;
    const dataset = selectedDatasetInfo();
    if (retriever.interaction === "gold_reference" && !dataset?.includes_gold_references) {
      return `${dataset?.label || "Selected dataset"} has no gold references`;
    }
    return "";
  }

  function reconcileSelectedRetrievers() {
    const byName = new Map(app.retrievers.map((retriever) => [retriever.name, retriever]));
    const removed = [];
    app.selectedRetrievers.forEach((name) => {
      const retriever = byName.get(name);
      if (retriever && !isRetrieverCompatible(retriever)) {
        app.selectedRetrievers.delete(name);
        removed.push(name);
      }
    });
    return removed;
  }

  function tokenLabel(row) {
    const labels = {
      OPENAI_API_KEY: "OpenAI / GraphRAG",
      ANTHROPIC_API_KEY: "Anthropic",
      XAI_API_KEY: "xAI / Grok",
      DASHSCOPE_API_KEY: "Qwen",
      LOCAL_OPENAI_API_KEY: "Local OpenAI-compatible"
    };
    return labels[row.name] || row.label;
  }

  function selectAllRetrievers() {
    app.selectedRetrievers = new Set(
      app.retrievers.filter(isRetrieverCompatible).map((retriever) => retriever.name)
    );
    syncChecks("data-retriever", app.selectedRetrievers);
    markDraft();
  }

  function clearRetrievers() {
    app.selectedRetrievers.clear();
    syncChecks("data-retriever", app.selectedRetrievers);
    markDraft();
  }

  function selectAllModels() {
    app.selectedModels = new Set(app.models.map((model) => model.id));
    syncChecks("data-model", app.selectedModels);
    markDraft();
  }

  function clearModels() {
    app.selectedModels.clear();
    syncChecks("data-model", app.selectedModels);
    markDraft();
  }

  function syncChecks(attribute, selected) {
    $$(`[${attribute}]`).forEach((input) => {
      input.checked = selected.has(input.getAttribute(attribute));
    });
  }

  function updateExperimentName() {
    const input = $("#experiment-name");
    const zipInput = $("#zip-name");
    const previousDefault = `${slug(input.dataset.previousName)}-gpt-pro.zip`;
    if (!zipInput.value.trim() || zipInput.value.trim() === previousDefault) {
      zipInput.value = `${slug(input.value)}-gpt-pro.zip`;
    }
    input.dataset.previousName = input.value;
    markDraft();
  }

  function markDraft() {
    app.dirty = true;
    app.exactRows = null;
    app.runStatus = null;
    app.artifacts = null;
    persistSetup();
    updateSummary();
  }

  function updateSelectionCounts() {
    if (!app.state) return;
    $("#rag-count").textContent = `${app.retrievers.length} ${app.retrievers.length === 1 ? "RAG" : "RAGs"}`;
    $("#selected-rag-count").textContent = `${app.selectedRetrievers.size} selected`;
    $("#context-count").textContent = `${app.selectedContexts.size} selected`;
    $("#model-count").textContent = `${app.models.length} ${app.models.length === 1 ? "model" : "models"}`;
    $("#selected-model-count").textContent = `${app.selectedModels.size} selected`;
  }

  function updateSummary() {
    updateSelectionCounts();
    const expected = app.runStatus?.expected_rows ?? app.exactRows ?? estimatedRows();
    const completed = app.runStatus?.completed_rows || 0;
    const progress = $("#run-progress");
    progress.max = Math.max(expected, 1);
    progress.value = Math.min(completed, Math.max(expected, 1));
    $("#progress-count").textContent = `${formatNumber(completed)} / ${formatNumber(expected)} rows`;

    const active = app.activeJob && ["queued", "running"].includes(app.activeJob.status);
    const valid = selectedDatasetInfo()?.available && app.selectedRetrievers.size > 0 && app.selectedContexts.size > 0 && app.selectedModels.size > 0;
    $("#test-rags").disabled = active || app.selectedRetrievers.size === 0;
    $("#prepare-rags").disabled = active || !valid;
    $("#start-run").disabled = active || !valid;
    $("#stop-run").disabled = !active;
    $("#export-zip").disabled = active || !app.runStatus || app.runStatus.rows_on_disk < 1 || app.runStatus.invalid_rows > 0;

    let runState = "Ready to configure.";
    if (!valid) runState = "Select at least one RAG, context mode, and model.";
    if (app.dirty && valid) runState = `Draft matrix: ${formatNumber(expected)} rows.`;
    if (app.runStatus?.resumable) runState = `${formatNumber(completed)} rows saved. Run / resume continues from the next row.`;
    if (app.runStatus?.complete) runState = `Complete: ${formatNumber(completed)} rows saved.`;
    if (app.runStatus?.invalid_rows) runState = `${app.runStatus.invalid_rows} incomplete row fragment found; resume will repair it.`;
    if (active && app.activeJob.action === "rag_audit") runState = "Testing selected RAGs in their compatible modes.";
    if (active && app.activeJob.action !== "rag_audit") runState = `${humanize(app.activeJob.action)} ${app.activeJob.status}: ${formatNumber(completed)} rows saved.`;
    if (app.activeJob?.status === "failed" && !active && app.activeJob.action === "run") runState = `Stopped or failed: ${formatNumber(completed)} rows remain resumable.`;
    $("#run-state").textContent = runState;

    const artifacts = app.runStatus || app.artifacts;
    $("#runs-path").textContent = artifacts?.runs_path || previewRunsPath();
    $("#zip-path").textContent = artifacts?.zip_path || previewZipPath();
    const download = $("#download-zip");
    const zipExists = Boolean(app.runStatus?.zip_exists);
    download.hidden = !zipExists;
    download.href = zipExists ? `/api/download?path=${encodeURIComponent(app.runStatus.zip_path)}` : "#";
    persistSetup();
  }

  function estimatedRows() {
    const datasetCount = selectedDatasetInfo()?.qa_count || DEFAULTS.limit;
    const questionCount = Math.min(numberValue("qa-limit", DEFAULTS.limit), datasetCount);
    return questionCount * app.selectedRetrievers.size * app.selectedContexts.size * app.selectedModels.size;
  }

  function previewRunsPath() {
    const output = $("#output-dir").value.trim() || DEFAULTS.outputDir;
    return `${trimTrailingSlash(output)}/${slug($("#experiment-name").value)}/runs.jsonl`;
  }

  function previewZipPath() {
    const runPath = previewRunsPath();
    return `${runPath.slice(0, runPath.lastIndexOf("/") + 1)}${$("#zip-name").value.trim() || DEFAULTS.zipName}`;
  }

  function collectPayload({ audit = false } = {}) {
    validateSetup({ audit });
    const baseName = $("#experiment-name").value.trim();
    return {
      name: audit ? `${slug(baseName)}-rag-audit` : baseName,
      output_dir: $("#output-dir").value.trim(),
      zip_name: audit ? `${slug(baseName)}-rag-audit.zip` : $("#zip-name").value.trim(),
      limit: numberValue("qa-limit", DEFAULTS.limit),
      top_k: numberValue("top-k", DEFAULTS.topK),
      max_evidence_chars: numberValue("evidence-chars", DEFAULTS.evidenceChars),
      dataset: selectedDataset(),
      ingestion_mode: selectedIngestion(),
      retrievers: Array.from(app.selectedRetrievers),
      context_modes: audit ? ["injected"] : Array.from(app.selectedContexts),
      models: audit ? [app.models[0].id] : Array.from(app.selectedModels),
      grader_mode: "gpt_pro",
      dry_run: audit || $("#dry-run").checked
    };
  }

  function validateSetup({ audit = false } = {}) {
    if (!$("#experiment-name").value.trim()) throw new Error("Experiment name is required.");
    if (!$("#output-dir").value.trim()) throw new Error("Output folder is required.");
    if (!$("#zip-name").value.trim()) throw new Error("ZIP filename is required.");
    if (!app.selectedRetrievers.size) throw new Error("Select at least one RAG.");
    if (!audit && !app.selectedContexts.size) throw new Error("Select at least one context mode.");
    if (!audit && !app.selectedModels.size) throw new Error("Select at least one model.");
    if (audit && !app.models.length) throw new Error("No catalog model is available for audit materialization.");
    for (const id of ["qa-limit", "top-k", "evidence-chars"]) {
      if (!$("#" + id).checkValidity()) {
        $("#" + id).reportValidity();
        throw new Error("Run settings contain an invalid number.");
      }
    }
  }

  async function materialize() {
    const result = await api("/api/configs", { method: "POST", body: collectPayload() });
    app.configPath = result.config_path;
    app.artifacts = result.artifacts;
    app.exactRows = result.plan.estimates.rows;
    app.dirty = false;
    app.runStatus = null;
    persistSetup();
    updateSummary();
    return result;
  }

  async function materializeAudit() {
    return api("/api/configs", { method: "POST", body: collectPayload({ audit: true }) });
  }

  async function startJob(action) {
    setActionBusy(true);
    showMessage("");
    try {
      const materialized = action === "rag_audit"
        ? await materializeAudit()
        : app.dirty || !app.configPath ? await materialize() : { config_path: app.configPath };
      const job = await api("/api/jobs", {
        method: "POST",
        body: {
          action,
          config_path: materialized.config_path,
          run_mode: "resume",
          ingestion_mode: selectedIngestion(),
          dry_run: $("#dry-run").checked,
          zip_name: $("#zip-name").value.trim(),
          external_checks: true,
          timeout_s: 30
        }
      });
      app.activeJobId = job.id;
      app.activeJob = job;
      $("#job-log").textContent = "Queued.\n";
      persistSetup();
      renderJob(job);
      schedulePoll(100);
    } catch (error) {
      showMessage(error.message, true);
    } finally {
      setActionBusy(false);
    }
  }

  function schedulePoll(delay = 1100) {
    window.clearTimeout(app.pollTimer);
    app.pollTimer = window.setTimeout(pollActiveJob, delay);
  }

  async function pollActiveJob() {
    if (!app.activeJobId) return;
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(app.activeJobId)}`);
      app.activeJob = job;
      renderJob(job);
      if (app.configPath) await refreshRunStatus(true);
      if (["queued", "running"].includes(job.status)) {
        schedulePoll();
      } else {
        app.activeJobId = null;
        persistSetup();
        await refreshRunStatus(true);
        const completeMessages = {
          run: "Run complete; ZIP created.",
          external_indexes: "Selected RAG indexes prepared.",
          rag_audit: `RAG test complete: ${auditSummaryText(job.report?.summary)}.`
        };
        const failedMessage = job.action === "run"
          ? "Job stopped. Saved rows can be resumed."
          : `${humanize(job.action)} failed. Open the job log for details.`;
        showMessage(job.status === "complete" ? completeMessages[job.action] : failedMessage, job.status !== "complete");
      }
    } catch (error) {
      if (error.status === 404) {
        app.activeJobId = null;
        app.activeJob = null;
        persistSetup();
        await refreshRunStatus(true);
        showMessage("The server restarted. Saved rows are still available to resume.");
      } else {
        showMessage(error.message, true);
        schedulePoll(2500);
      }
    }
  }

  function renderJob(job) {
    app.activeJob = job;
    if (job.action === "rag_audit" && job.report && app.ragAuditJobId !== job.id) {
      applyRagAudit(job.report, job.id);
    }
    const output = $("#job-log");
    output.textContent = (job.logs || []).join("\n") || `${humanize(job.status)}.\n`;
    output.scrollTop = output.scrollHeight;
    if (job.status === "failed") $("#job-log-wrap").open = true;
    setPageStatus(["queued", "running"].includes(job.status) ? `${humanize(job.action)} ${job.status}` : "Ready", job.status === "failed");
    updateSummary();
  }

  async function refreshRunStatus(silent = false) {
    if (!app.configPath) return;
    const query = new URLSearchParams({
      config_path: app.configPath,
      zip_name: $("#zip-name").value.trim()
    });
    try {
      app.runStatus = await api(`/api/run-status?${query}`);
      app.artifacts = {
        run_dir: app.runStatus.run_dir,
        runs_path: app.runStatus.runs_path,
        zip_path: app.runStatus.zip_path
      };
      app.exactRows = app.runStatus.expected_rows;
      updateSummary();
    } catch (error) {
      if (!silent) showMessage(error.message, true);
      if (error.status === 400) {
        app.configPath = null;
        app.dirty = true;
        persistSetup();
      }
    }
  }

  async function stopJob() {
    if (!app.activeJobId) return;
    $("#stop-run").disabled = true;
    try {
      await api(`/api/jobs/${encodeURIComponent(app.activeJobId)}/cancel`, { method: "POST", body: {} });
      showMessage("Stopping after the current process write.");
      schedulePoll(100);
    } catch (error) {
      showMessage(error.message, true);
    }
  }

  async function exportZip() {
    if (!app.runStatus?.runs_path) return;
    $("#export-zip").disabled = true;
    try {
      await api("/api/bundles", {
        method: "POST",
        body: {
          runs: app.runStatus.runs_path,
          mode: "gpt_pro",
          zip_name: $("#zip-name").value.trim()
        }
      });
      await refreshRunStatus(true);
      showMessage("ZIP created.");
    } catch (error) {
      showMessage(error.message, true);
    } finally {
      updateSummary();
    }
  }

  async function saveTokens(event) {
    event.preventDefault();
    const values = $$('[data-token]')
      .map((input) => ({ name: input.dataset.token, value: input.value.trim() }))
      .filter((row) => row.value);
    if (!values.length) {
      showMessage("Enter at least one token to save.", true);
      $$('[data-token]')[0]?.focus();
      return;
    }
    setTokenBusy(true);
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
      setTokenBusy(false);
    }
  }

  async function clearToken(button) {
    const name = button.dataset.clearToken;
    if (!window.confirm(`Remove ${name}?`)) return;
    setTokenBusy(true);
    try {
      await api("/api/credentials/clear", { method: "POST", body: { name } });
      app.state = await api("/api/state");
      renderTokens();
      showMessage(`${name} removed.`);
    } catch (error) {
      showMessage(error.message, true);
    } finally {
      setTokenBusy(false);
    }
  }

  function setTokenBusy(busy) {
    $("#save-tokens").disabled = busy;
    $$('[data-clear-token]').forEach((button) => {
      button.disabled = busy || button.dataset.configured !== "true";
    });
    $("#token-form").setAttribute("aria-busy", String(busy));
  }

  function setActionBusy(busy) {
    if (busy) {
      $("#test-rags").disabled = true;
      $("#prepare-rags").disabled = true;
      $("#start-run").disabled = true;
    } else {
      updateSummary();
    }
  }

  function persistSetup() {
    const hasFields = Boolean($("#experiment-name"));
    const setup = {
      name: hasFields ? $("#experiment-name").value : app.setup.name,
      outputDir: hasFields ? $("#output-dir").value : app.setup.outputDir,
      zipName: hasFields ? $("#zip-name").value : app.setup.zipName,
      limit: hasFields ? numberValue("qa-limit", DEFAULTS.limit) : app.setup.limit,
      topK: hasFields ? numberValue("top-k", DEFAULTS.topK) : app.setup.topK,
      evidenceChars: hasFields ? numberValue("evidence-chars", DEFAULTS.evidenceChars) : app.setup.evidenceChars,
      dataset: hasFields ? selectedDataset() : app.setup.dataset,
      ingestionMode: hasFields ? selectedIngestion() : app.setup.ingestionMode,
      dryRun: hasFields ? $("#dry-run").checked : app.setup.dryRun,
      retrievers: Array.from(app.selectedRetrievers),
      contexts: Array.from(app.selectedContexts),
      models: Array.from(app.selectedModels),
      configPath: app.configPath,
      artifacts: app.artifacts,
      activeJobId: app.activeJobId,
      dirty: app.dirty
    };
    app.setup = setup;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(setup));
  }

  function readSetup() {
    try {
      const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
      if (stored && typeof stored === "object") return { ...DEFAULTS, ...stored };
      const legacyModels = JSON.parse(localStorage.getItem(LEGACY_MODEL_KEY) || "[]");
      return { ...DEFAULTS, models: Array.isArray(legacyModels) ? legacyModels : [] };
    } catch {
      return { ...DEFAULTS };
    }
  }

  function uniqueModels(entries) {
    const byId = new Map();
    entries.forEach((entry) => {
      const id = modelId(entry);
      if (!byId.has(id)) {
        byId.set(id, { id, provider: entry.provider, model: entry.model });
      }
    });
    return Array.from(byId.values());
  }

  function numberValue(id, fallback) {
    const value = Number.parseInt($("#" + id).value, 10);
    return Number.isFinite(value) ? value : fallback;
  }

  function selectedIngestion() {
    return $('input[name="ingestion"]:checked')?.value || DEFAULTS.ingestionMode;
  }

  function selectedDataset() {
    return $('input[name="dataset"]:checked')?.value || app.setup.dataset || app.state?.default_dataset || DEFAULTS.dataset;
  }

  function selectedDatasetInfo() {
    return app.state?.datasets?.find((dataset) => dataset.id === selectedDataset()) || app.state?.dataset || null;
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
    if (!response.ok) {
      const error = new Error(payload.error || `HTTP ${response.status}`);
      error.status = response.status;
      throw error;
    }
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
      xai: "xAI / Grok",
      qwen: "Qwen",
      local_openai: "Local models"
    };
    return labels[value] || humanize(value);
  }

  function familyLabel(value) {
    const labels = {
      local: "Local baselines",
      local_vector_db: "Vector database",
      canonical_rag: "Canonical RAG",
      gems_rag: "GEMS-RAG",
      gfm_rag: "GFM-RAG",
      raganything: "RAG-Anything",
      paperqa2: "PaperQA2"
    };
    return labels[value] || humanize(value);
  }

  function humanize(value) {
    const text = String(value || "").replaceAll("_", " ");
    return text ? text[0].toUpperCase() + text.slice(1) : "";
  }

  function slug(value) {
    return String(value || DEFAULTS.name).toLowerCase().replace(/[^a-z0-9-]+/g, "-").replace(/^-+|-+$/g, "") || DEFAULTS.name;
  }

  function trimTrailingSlash(value) {
    return String(value).replace(/[\\/]+$/, "");
  }

  function formatNumber(value) {
    return new Intl.NumberFormat("en-US").format(Number(value) || 0);
  }

  function cssEscape(value) {
    return window.CSS?.escape ? window.CSS.escape(value) : String(value).replace(/["\\]/g, "\\$&");
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
