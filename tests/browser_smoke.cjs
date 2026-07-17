const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

const baseUrl = process.argv[2] || "http://127.0.0.1:8765/";
const outputDir = process.argv[3] || "data/working/gui/screenshots";
const runOutputDir = "data/working/gui/browser-smoke-runs";
const runName = "browser-smoke-resume";
const zipName = "browser-smoke-output.zip";
fs.mkdirSync(outputDir, { recursive: true });
fs.rmSync(runOutputDir, { recursive: true, force: true });

const desktopScreenshot = path.join(outputDir, "ablation-setup-desktop.png");
const mobileScreenshot = path.join(outputDir, "ablation-setup-mobile.png");

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 980 }, deviceScaleFactor: 1 });
  const errors = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));

  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await page.locator('#app[data-loading="false"]').waitFor();
  await page.evaluate(() => window.localStorage.clear());
  await page.reload({ waitUntil: "networkidle" });
  await page.locator('#app[data-loading="false"]').waitFor();

  const catalog = await page.evaluate(async () => {
    const response = await fetch("/api/state");
    const state = await response.json();
    const answerModels = state.catalogs.models.filter((entry) => entry.metadata.roles.includes("answer"));
    return {
      modelIds: Array.from(new Set(answerModels.map((entry) => `${entry.provider}:${entry.model}`))).sort(),
      providerCounts: answerModels.reduce((counts, entry) => {
        counts[entry.provider] = (counts[entry.provider] || 0) + 1;
        return counts;
      }, {}),
      retrieverIds: state.catalogs.retrievers.map((entry) => entry.name).sort(),
      interactionCounts: state.catalogs.retrievers.reduce((counts, entry) => {
        counts[entry.interaction] = (counts[entry.interaction] || 0) + 1;
        return counts;
      }, {}),
      contexts: state.context_modes.map((entry) => entry.name).sort(),
      dataset: state.dataset,
      datasets: state.datasets,
      defaultDataset: state.default_dataset,
      ragBackendPresets: state.rag_backend_presets
    };
  });

  const modelCheckboxes = page.locator('#model-list input[type="checkbox"][data-model]');
  const renderedModelIds = (await modelCheckboxes.evaluateAll((elements) => elements.map((element) => element.dataset.model))).sort();
  assert.equal(catalog.modelIds.length, 87);
  assert.deepEqual(catalog.providerCounts, {
    openai: 21,
    anthropic: 11,
    xai: 5,
    qwen: 47,
    local_openai: 3
  });
  assert.equal(renderedModelIds.length, catalog.modelIds.length);
  assert.equal(new Set(renderedModelIds).size, catalog.modelIds.length);
  assert.deepEqual(renderedModelIds, catalog.modelIds);

  const ragCheckboxes = page.locator('#rag-list input[type="checkbox"][data-retriever]');
  const renderedRagIds = (await ragCheckboxes.evaluateAll((elements) => elements.map((element) => element.dataset.retriever))).sort();
  assert.equal(renderedRagIds.length, 42);
  assert.deepEqual(renderedRagIds, catalog.retrieverIds);
  assert.deepEqual(catalog.interactionCounts, {
    query_driven: 39,
    fixed_question: 1,
    gold_reference: 1,
    no_retrieval: 1
  });
  assert.equal(await page.locator('[data-rag-segment="query_driven"]').getAttribute("data-count"), "39");
  assert.equal(await page.locator('[data-rag-segment="fixed_question"]').getAttribute("data-count"), "1");
  assert.equal(await page.locator('[data-rag-segment="controls"]').getAttribute("data-count"), "2");
  assert.equal(await page.locator('#context-list input[type="checkbox"][data-context]').count(), 4);
  assert.deepEqual(
    (await page.locator('[data-context]').evaluateAll((elements) => elements.map((element) => element.dataset.context))).sort(),
    catalog.contexts
  );

  const tokenInputs = page.locator("#token-list .token-row input");
  assert.equal(await page.locator("#token-list .token-row").count(), 5);
  assert.equal(await tokenInputs.count(), 5);
  assert.deepEqual(await tokenInputs.evaluateAll((inputs) => inputs.map((input) => input.value)), ["", "", "", "", ""]);
  assert.equal(await page.locator('[data-token="GRAPHRAG_API_KEY"]').count(), 0);
  assert.equal((await page.locator('[data-token="OPENAI_API_KEY"]').locator("xpath=preceding-sibling::label").innerText()).includes("GraphRAG"), false);

  assert.deepEqual(catalog.ragBackendPresets.map((entry) => entry.provider), ["openai", "local_openai"]);
  assert.equal(await page.locator('input[name="rag-backend-provider"][value="openai"]').isChecked(), true);
  assert.equal(await page.locator("#rag-base-url").inputValue(), "");
  assert.equal(await page.locator("#rag-chat-model").inputValue(), "gpt-4o-mini");
  assert.equal(await page.locator("#rag-embedding-model").inputValue(), "text-embedding-3-small");
  assert.equal(await page.locator("#rag-embedding-dim").inputValue(), "1536");
  assert.equal(await page.locator("#rag-reasoning-effort").inputValue(), "");

  assert.equal(await page.locator('#output-dir').inputValue(), "runs");
  assert.equal(await page.locator('[data-retriever="bm25"]').isChecked(), true);
  assert.equal(await page.locator('[data-context="injected"]').isChecked(), true);
  assert.equal(await page.locator('[data-context="tool_native"]').isChecked(), false);
  assert.equal(await checkedModelCount(page), 0);
  assert.equal(await page.locator('input[name="dataset"]').count(), 2);
  assert.equal(catalog.defaultDataset, "mutcd150");
  assert.equal(catalog.dataset.qa_count, 150);
  assert.equal(catalog.dataset.includes_gold_answers, false);
  assert.equal(await page.locator('input[name="dataset"][value="mutcd150"]').isChecked(), true);
  assert.match(await page.locator("#qa-source").innerText(), /^150 questions \(no gold answers\) \| /);

  await page.locator("#test-rags").click();
  await page.waitForFunction(() => document.querySelector("#rag-audit-summary")?.textContent === "1 ready", null, { timeout: 60000 });
  assert.equal(await page.locator('[data-retriever="bm25"]').locator("xpath=ancestor::label").locator(".audit-status").innerText(), "ready");

  await page.locator('input[name="rag-backend-provider"][value="local_openai"]').check();
  assert.equal(await page.locator("#rag-base-url").inputValue(), "http://localhost:8000/v1");
  assert.equal(await page.locator("#rag-chat-model").inputValue(), "qwen3:8b");
  assert.equal(await page.locator("#rag-embedding-model").inputValue(), "nomic-embed-text");
  assert.equal(await page.locator("#rag-embedding-dim").inputValue(), "768");
  assert.equal(await page.locator("#rag-reasoning-effort").inputValue(), "none");
  assert.equal(await page.locator("#rag-audit-summary").innerText(), "Not tested");

  assert.equal(await page.locator('[data-retriever="oracle_gold_refs"]').isDisabled(), true);
  await page.locator('input[name="dataset"][value="curated49"]').check();
  assert.match(await page.locator("#qa-source").innerText(), /^49 Q\/A pairs \| /);
  await page.locator('[data-retriever="oracle_gold_refs"]').check();
  await page.locator('[data-context="tool_native"]').check();
  assert.equal(await page.locator('[data-retriever="oracle_gold_refs"]').isChecked(), false);
  assert.equal(await page.locator('[data-retriever="oracle_gold_refs"]').isDisabled(), true);
  assert.match(await page.locator("#message").innerText(), /incompatible RAG was deselected/);
  await page.locator('input[name="dataset"][value="mutcd150"]').check();

  const firstModel = modelCheckboxes.first();
  const modelId = await firstModel.getAttribute("data-model");
  await firstModel.check();
  await page.locator("#experiment-name").fill(runName);
  await page.locator("#output-dir").fill(runOutputDir);
  await page.locator("#zip-name").fill(zipName);
  await page.locator("#qa-limit").fill("1");
  await page.locator("#dry-run").check();
  assert.equal(await selectedModelCount(page), 1);
  assert.match(await page.locator("#progress-count").innerText(), /0 \/ 2 rows/);

  await page.locator("#start-run").click();
  await page.waitForFunction(() => document.querySelector("#run-state")?.textContent.startsWith("Complete:"), null, { timeout: 60000 });
  const status = await page.evaluate(async ({ runName, zipName }) => {
    const setup = JSON.parse(localStorage.getItem("gems-rag:ablation-setup-v2"));
    const query = new URLSearchParams({ config_path: setup.configPath, zip_name: zipName });
    const response = await fetch(`/api/run-status?${query}`);
    return { setup, body: await response.json(), runName };
  }, { runName, zipName });
  assert.equal(status.body.complete, true);
  assert.equal(status.body.completed_rows, 2);
  assert.equal(status.body.expected_rows, 2);
  assert.equal(status.body.invalid_rows, 0);
  assert.equal(status.body.zip_exists, true);
  assert.ok(status.body.runs_path.endsWith(`${runOutputDir}/${runName}/runs.jsonl`));
  assert.ok(status.body.zip_path.endsWith(`${runOutputDir}/${runName}/${zipName}`));
  assert.equal(status.setup.ragBackend.provider, "local_openai");
  assert.equal(status.setup.ragBackend.chat_model, "qwen3:8b");
  assert.equal(await page.locator("#download-zip").isVisible(), true);

  await page.reload({ waitUntil: "networkidle" });
  await page.locator('#app[data-loading="false"]').waitFor();
  assert.equal(await page.locator(`[data-model="${modelId}"]`).isChecked(), true);
  assert.equal(await page.locator("#output-dir").inputValue(), runOutputDir);
  assert.equal(await page.locator("#zip-name").inputValue(), zipName);
  assert.equal(await page.locator('input[name="dataset"][value="mutcd150"]').isChecked(), true);
  assert.equal(await page.locator('input[name="rag-backend-provider"][value="local_openai"]').isChecked(), true);
  assert.equal(await page.locator("#rag-chat-model").inputValue(), "qwen3:8b");
  assert.equal(await page.locator("#rag-audit-summary").innerText(), "Not tested");
  await page.waitForFunction(() => document.querySelector("#run-state")?.textContent.startsWith("Complete:"));
  assert.match(await page.locator("#progress-count").innerText(), /2 \/ 2 rows/);

  for (const selector of [".sidebar", ".primary-nav", ".nav-item", "[data-view]", "#view-manual", "#view-runs"]) {
    assert.equal(await page.locator(selector).count(), 0, `legacy UI remains: ${selector}`);
  }

  await assertViewport(page, "desktop ablation setup");
  await page.screenshot({ path: desktopScreenshot, fullPage: true });

  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(250);
  await assertViewport(page, "mobile ablation setup");
  await page.screenshot({ path: mobileScreenshot, fullPage: true });

  assert.deepEqual(errors, []);
  await browser.close();
  process.stdout.write(JSON.stringify({ ok: true, screenshots: [desktopScreenshot, mobileScreenshot], run: status.body }) + "\n");
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

async function checkedModelCount(page) {
  return page.locator('#model-list input[type="checkbox"][data-model]:checked').count();
}

async function selectedModelCount(page) {
  const text = await page.locator("#selected-model-count").innerText();
  const match = text.match(/\d+/);
  assert.ok(match, `selected count does not contain a number: ${JSON.stringify(text)}`);
  return Number(match[0]);
}

async function assertViewport(page, label) {
  const overflow = await page.evaluate(() => ({
    documentWidth: document.documentElement.scrollWidth,
    viewportWidth: document.documentElement.clientWidth,
    wideElements: Array.from(document.querySelectorAll("body *"))
      .map((element) => ({ element, rect: element.getBoundingClientRect() }))
      .filter(({ rect }) => rect.width > 0 && (rect.right > document.documentElement.clientWidth + 1 || rect.width > document.documentElement.clientWidth + 1))
      .slice(0, 12)
      .map(({ element, rect }) => ({ tag: element.tagName, id: element.id, className: String(element.className).slice(0, 80), left: Math.round(rect.left), right: Math.round(rect.right), width: Math.round(rect.width) })),
    offenders: Array.from(document.querySelectorAll("button, input, select"))
      .filter((element) => element.clientWidth > 0 && element.scrollWidth > element.clientWidth + 2)
      .slice(0, 10)
      .map((element) => ({ tag: element.tagName, id: element.id, text: element.textContent.trim().slice(0, 60), client: element.clientWidth, scroll: element.scrollWidth }))
  }));
  assert.ok(overflow.documentWidth <= overflow.viewportWidth + 1, `${label} has document overflow: ${JSON.stringify(overflow)}`);
  assert.deepEqual(overflow.wideElements, [], `${label} has elements outside the viewport`);
  assert.deepEqual(overflow.offenders, [], `${label} has clipped controls`);
}
