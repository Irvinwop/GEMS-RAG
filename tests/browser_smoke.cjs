const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

const baseUrl = process.argv[2] || "http://127.0.0.1:8765/";
const outputDir = process.argv[3] || "data/working/gui/screenshots";
fs.mkdirSync(outputDir, { recursive: true });

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
  assert.match(await page.locator("#manual-status").innerText(), /Manual verified/);
  assert.ok((await page.locator('[data-retriever]').count()) >= 40);
  assert.ok((await page.locator('[data-model]').count()) >= 10);
  assert.equal(await page.locator("#plan-rows").innerText(), "480");
  await assertViewport(page, "desktop experiment");
  await page.screenshot({ path: path.join(outputDir, "experiment-desktop.png"), fullPage: true });

  await page.locator('label:has(input[name="ingestion"][value="native_pdf"]) span').click();
  assert.match(await page.locator("#ingestion-note").innerText(), /PaperQA2/);
  await page.locator("#materialize-config").click();
  await page.locator("#plan-state", { hasText: "Materialized" }).waitFor();
  await page.locator(".toast").evaluateAll((elements) => elements.forEach((element) => element.remove()));

  await page.locator('[data-view="manual"]').click();
  await page.locator("#view-manual.active").waitFor();
  await page.waitForTimeout(250);
  const manualImage = page.locator(".manual-preview img");
  await manualImage.waitFor();
  assert.ok(await manualImage.evaluate((image) => image.complete && image.naturalWidth > 1000));
  assert.equal(await page.locator("#manual-matrix-body tr").count(), 19);
  await assertViewport(page, "desktop manual");
  await page.screenshot({ path: path.join(outputDir, "manual-desktop.png"), fullPage: true });

  await page.locator('[data-view="credentials"]').click();
  await page.locator("#view-credentials.active").waitFor();
  await page.waitForTimeout(250);
  assert.equal(await page.locator(".credential-row").count(), 8);
  assert.equal(await page.locator('.credential-row input').first().getAttribute("value"), null);
  await assertViewport(page, "desktop credentials");
  await page.screenshot({ path: path.join(outputDir, "credentials-desktop.png"), fullPage: true });

  await page.locator('[data-view="runs"]').click();
  await page.locator("#view-runs.active").waitFor();
  await page.waitForTimeout(250);
  assert.ok((await page.locator("#run-table-body tr").count()) >= 10);
  await assertViewport(page, "desktop runs");
  await page.screenshot({ path: path.join(outputDir, "runs-desktop.png"), fullPage: true });

  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('[data-view="experiment"]').click();
  await page.locator("#view-experiment.active").waitFor();
  await page.waitForTimeout(250);
  await assertViewport(page, "mobile experiment");
  await page.screenshot({ path: path.join(outputDir, "experiment-mobile.png"), fullPage: true });

  await page.locator('[data-view="credentials"]').click();
  await page.locator("#view-credentials.active").waitFor();
  await page.waitForTimeout(250);
  assert.equal(await page.locator(".nav-item").last().isVisible(), true);
  await assertViewport(page, "mobile credentials");
  await page.screenshot({ path: path.join(outputDir, "credentials-mobile.png"), fullPage: true });

  assert.deepEqual(errors, []);
  await browser.close();
  process.stdout.write(JSON.stringify({ ok: true, screenshots: fs.readdirSync(outputDir).sort() }) + "\n");
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

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
  assert.deepEqual(overflow.offenders, [], `${label} has clipped controls`);
}
