const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

const baseUrl = process.argv[2] || "http://127.0.0.1:8765/";
const outputDir = process.argv[3] || "data/working/gui/screenshots";
fs.mkdirSync(outputDir, { recursive: true });

const desktopScreenshot = path.join(outputDir, "model-picker-desktop.png");
const mobileScreenshot = path.join(outputDir, "model-picker-mobile.png");

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
      ids: Array.from(new Set(answerModels.map((entry) => `${entry.provider}:${entry.model}`))).sort(),
      providerCounts: answerModels.reduce((counts, entry) => {
        counts[entry.provider] = (counts[entry.provider] || 0) + 1;
        return counts;
      }, {})
    };
  });
  const modelCheckboxes = page.locator('#model-list input[type="checkbox"][data-model]');
  const renderedModelIds = (await modelCheckboxes.evaluateAll((elements) => elements.map((element) => element.dataset.model))).sort();
  assert.equal(catalog.ids.length, 87);
  assert.deepEqual(catalog.providerCounts, {
    openai: 21,
    anthropic: 11,
    xai: 5,
    qwen: 47,
    local_openai: 3
  });
  assert.equal(renderedModelIds.length, catalog.ids.length);
  assert.equal(new Set(renderedModelIds).size, catalog.ids.length);
  assert.deepEqual(renderedModelIds, catalog.ids);

  const tokenInputs = page.locator("#token-list .token-row input");
  assert.equal(await page.locator("#token-list .token-row").count(), 6);
  assert.equal(await tokenInputs.count(), 6);
  assert.deepEqual(await tokenInputs.evaluateAll((inputs) => inputs.map((input) => input.value)), ["", "", "", "", "", ""]);

  for (const selector of [
    ".sidebar",
    ".primary-nav",
    ".nav-item",
    "[data-view]",
    "[data-retriever]",
    "#retriever-groups",
    "#view-manual",
    "#manual-status",
    "#view-runs",
    "#run-table-body",
    "#plan-rows",
    "#job-output"
  ]) {
    assert.equal(await page.locator(selector).count(), 0, `legacy UI remains: ${selector}`);
  }

  assert.equal(await selectedCount(page), await checkedModelCount(page));
  const firstModel = modelCheckboxes.first();
  const modelId = await firstModel.getAttribute("data-model");
  const originalState = await firstModel.isChecked();
  await firstModel.click();
  const persistedState = !originalState;
  assert.equal(await selectedCount(page), await checkedModelCount(page));
  assert.ok(await page.evaluate(() => window.localStorage.length > 0));

  await page.reload({ waitUntil: "networkidle" });
  await page.locator('#app[data-loading="false"]').waitFor();
  assert.equal(await page.locator(`[data-model="${modelId}"]`).isChecked(), persistedState);
  assert.equal(await selectedCount(page), await checkedModelCount(page));

  await assertViewport(page, "desktop model picker");
  await page.screenshot({ path: desktopScreenshot, fullPage: true });

  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(250);
  await assertViewport(page, "mobile model picker");
  await page.screenshot({ path: mobileScreenshot, fullPage: true });

  assert.deepEqual(errors, []);
  await browser.close();
  process.stdout.write(JSON.stringify({ ok: true, screenshots: [desktopScreenshot, mobileScreenshot] }) + "\n");
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

async function checkedModelCount(page) {
  return page.locator('#model-list input[type="checkbox"][data-model]:checked').count();
}

async function selectedCount(page) {
  const text = await page.locator("#selected-count").innerText();
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
