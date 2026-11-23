// Every value from a document is inserted with textContent or as an input value, never as HTML.

const $ = (id) => document.getElementById(id);

function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "data") Object.assign(el.dataset, v);
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k in el && k !== "list") el[k] = v;
    else el.setAttribute(k, v);
  }
  el.append(...children.flat().filter((c) => c !== null && c !== undefined));
  return el;
}

const FIELD_GROUPS = [
  ["Vendor", [["vendor.name", "Name", "text"], ["vendor.address", "Address", "text"], ["vendor.tax_id", "Tax ID"]]],
  ["Invoice", [["invoice.number", "Number"], ["invoice.issue_date", "Issue date"], ["invoice.due_date", "Due date"], ["invoice.currency", "Currency"], ["invoice.po_number", "PO number"]]],
  ["Amounts", [["amounts.subtotal", "Subtotal"], ["amounts.discount", "Discount"], ["amounts.shipping", "Shipping"], ["amounts.tax_amount", "Tax"], ["amounts.tax_rate", "Tax rate %"], ["amounts.total", "Total"]]],
];
const OPTIONAL_WHEN_EMPTY = new Set(["amounts.discount", "amounts.shipping"]);
const LINE_FIELDS = [["description", "Description"], ["quantity", "Qty"], ["unit_price", "Unit price"], ["line_total", "Line total"]];
const RULES = {
  required_fields: "Required fields", line_math: "Quantity × price", lines_to_subtotal: "Lines add up",
  subtotal_tax_total: "Subtotal + tax = total", dates: "Dates", tax_id: "Tax ID format",
  currency: "Currency code", instruction_text: "Text addressed to the AI", schema: "Schema",
};
const STEPS = [
  ["ingest", "Ingest"], ["extract", "Extract"], ["cross_check", "Cross-check"],
  ["validate", "Validate"], ["repair", "Repair"], ["human_review", "Human review"],
];
const NODE_STEP = { extract_text: "extract", extract_image: "extract" };
const ICON = { pass: "✓", warn: "!", fail: "✕" };
const WORD = { pass: "Pass", warn: "Not checked", fail: "Fail" };

let config = { confidence_threshold: 0.85, max_upload_mb: 5, max_pages: 5, uploads_enabled: true };
let samples = [];
let current = null; // { runId, fields, confidence, checks, disagreements, result, source }
let previewUrl = null;

async function api(path, options) {
  const r = await fetch(path, options);
  let body = null;
  try { body = await r.json(); } catch { /* not JSON */ }
  if (!r.ok) {
    const detail = typeof body?.detail === "string" ? body.detail : `The server answered ${r.status}.`;
    throw Object.assign(new Error(detail), { status: r.status });
  }
  return body;
}

// --- Start ---------------------------------------------------------------------------------------

async function init() {
  try {
    [config, samples] = await Promise.all([api("/v1/config"), api("/v1/samples")]);
  } catch {
    showStartError("The server is not answering yet. Reload in a few seconds.");
    return;
  }
  const mode = $("mode");
  mode.hidden = false;
  mode.textContent = config.mode === "live" ? config.models.reader : "Offline mode, no AI model";
  mode.title = config.mode === "live"
    ? `Reader ${config.models.reader}. Second opinion ${config.models.second_text} (text), ${config.models.second_vision} (images).`
    : "No API key is set. Rule-based readers stand in for the models and can only read digital PDFs.";
  $("samples").replaceChildren(...samples.map((s) =>
    h("button", { type: "button", class: "sample", onclick: () => startSample(s) },
      h("img", { src: s.preview_url, alt: "", loading: "lazy" }),
      h("span", {}, h("b", {}, s.title), s.blurb))));
  if (config.uploads_enabled) {
    $("upload-note").textContent = `PDF, PNG or JPG in English, up to ${config.max_upload_mb} MB and ${config.max_pages} pages. ${config.runs_per_ip_per_hour} runs an hour.`;
    $("privacy").textContent = config.mode === "live"
      ? "This demo keeps no copy of your file and runs on free model tiers, whose providers may use what they receive. Please don't upload invoices with real personal details. Extracted fields appear in a public trace."
      : "Your file stays in this server's memory and is dropped when the run ends.";
  } else {
    $("file").disabled = true;
    $("upload-note").textContent = "Uploads are switched off right now. The samples still work.";
  }
}

function showStartError(message) {
  const el = $("start-error");
  el.textContent = message;
  el.hidden = !message;
}

function setBusy(busy) {
  document.querySelectorAll(".sample").forEach((b) => (b.disabled = busy));
  $("file").disabled = busy || !config.uploads_enabled;
}

async function startSample(sample) {
  const body = new FormData();
  body.append("sample_id", sample.id);
  await start(body, sample.title, h("img", { src: sample.preview_url, alt: `${sample.title}: the document being read` }));
}

async function startFile(file) {
  if (!file) return;
  if (file.size > config.max_upload_mb * 1024 * 1024) {
    showStartError(`That file is larger than ${config.max_upload_mb} MB.`);
    return;
  }
  const body = new FormData();
  body.append("file", file);
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  let preview;
  if (file.type.startsWith("image/")) {
    previewUrl = URL.createObjectURL(file);
    preview = h("img", { src: previewUrl, alt: "The document being read" });
  } else {
    previewUrl = null;
    preview = h("figcaption", {}, file.name);
  }
  await start(body, file.name, preview);
}

async function start(body, name, preview) {
  showStartError("");
  setBusy(true);
  let created;
  try {
    created = await api("/v1/extract", { method: "POST", body });
  } catch (e) {
    showStartError(e instanceof TypeError ? "Could not reach the server. Check your connection and try again." : e.message);
    setBusy(false);
    $("start-error").scrollIntoView({ block: "nearest" });
    return;
  }
  current = { runId: created.run_id, fields: null, confidence: {}, checks: [], disagreements: [], result: null };
  $("run-h").textContent = name;
  $("preview").replaceChildren(preview);
  for (const id of ["fields", "lines", "checks", "metrics", "calls", "trace"]) $(id).replaceChildren();
  $("review").hidden = true;
  $("json-box").hidden = true;
  buildRail();
  setVerdict("running", created.cached ? "Replaying a recorded run" : "Reading the document", []);
  $("run").hidden = false;
  $("run-h").focus({ preventScroll: true });
  $("run").scrollIntoView({ block: "start" });
  renderFields(false);
  listen(current);
}

// --- Live progress -------------------------------------------------------------------------------

function listen(run) {
  const source = new EventSource(`/v1/runs/${run.runId}/events`);
  const on = (type, fn) => source.addEventListener(type, (e) => {
    if (current === run) fn(JSON.parse(e.data));
    else source.close();
  });

  on("node_start", (e) => step(e.node, "running", e.attempt > 1 ? `attempt ${e.attempt}` : "running"));
  on("node_end", (e) => step(e.node, e.summary?.error ? "failed" : "done", describe(e)));
  on("node_paused", (e) => step(e.node, "paused", "waiting for you"));
  on("partial", (e) => {
    if (e.fields) run.fields = e.fields;
    if (e.confidence) run.confidence = e.confidence;
    if (e.checks) { run.checks = e.checks; renderChecks(e.checks); }
    renderFields(false);
  });
  on("result", (e) => { source.close(); finish(run, e.result); });
  source.onerror = () => {
    if (source.readyState === EventSource.CLOSED && current === run && !run.result) poll(run);
  };
}

async function poll(run) {
  for (let i = 0; i < 60 && current === run; i++) {
    try {
      const result = await api(`/v1/runs/${run.runId}`);
      if (result.status !== "running") return finish(run, result);
    } catch (e) {
      if (e.status === 404) break;
    }
    await new Promise((r) => setTimeout(r, 2000));
  }
  if (current === run) {
    setBusy(false);
    setVerdict("failed", "Lost contact with the run", ["The server may have restarted. Please try again."]);
  }
}

function buildRail() {
  $("rail").replaceChildren(...STEPS.map(([id, label]) =>
    h("li", { class: "step", id: `step-${id}`, data: { state: "idle" } }, h("b", {}, label), h("small", {}, ""))));
}

function step(node, state, text) {
  const el = $(`step-${NODE_STEP[node] ?? node}`);
  if (!el) return;
  el.dataset.state = state;
  if (node === "extract_text") el.querySelector("b").textContent = "Extract from text";
  if (node === "extract_image") el.querySelector("b").textContent = "Extract from image";
  el.querySelector("small").textContent = text;
}

const ms = (n) => (n >= 1000 ? `${(n / 1000).toFixed(1)} s` : `${n} ms`);

function describe(e) {
  const s = e.summary ?? {};
  const bits = [ms(e.latency_ms)];
  if (s.file_kind) bits.push({ pdf_text: "PDF with text", pdf_scan: "scanned PDF", image: "image" }[s.file_kind]);
  if (s.disagreements) bits.push(s.disagreements.length ? `${s.disagreements.length} differ` : "readings agree");
  if (s.unavailable) bits.push("no second opinion");
  if (s.failed) bits.push(s.failed.length ? `${s.failed.length} failed` : "all pass");
  if (s.changed) bits.push(s.changed.length ? `${s.changed.length} changed` : "same values");
  if (e.node === "repair") bits.push(`try ${e.attempt}`);
  if (s.error) bits.push("failed");
  return bits.join(", ");
}

// --- Result --------------------------------------------------------------------------------------

function finish(run, r) {
  run.result = r;
  run.fields = r.fields;
  run.confidence = r.confidence;
  run.disagreements = r.disagreements;
  run.checks = r.checks;
  setBusy(false);
  const reviewing = r.status === "awaiting_review" && !r.cached;
  for (const [id] of STEPS) {
    const el = $(`step-${id}`);
    if (el.dataset.state === "idle") el.querySelector("small").textContent = "not needed";
    if (el.dataset.state === "running") el.dataset.state = r.status === "failed" ? "failed" : "done";
  }
  if (r.reviewed_by_human) step("human_review", r.decision === "approved" ? "done" : "failed", r.decision === "approved" ? "approved" : "rejected");
  renderVerdict(r);
  renderFields(reviewing);
  renderChecks(r.checks);
  renderReview(reviewing);
  renderMetrics(r);
  if (r.fields) {
    $("json-box").hidden = false;
    $("json").textContent = JSON.stringify(r.fields, null, 2);
  }
}

function setVerdict(kind, title, lines, footnote) {
  const box = $("verdict");
  box.dataset.kind = kind;
  box.replaceChildren(
    h("strong", {}, title),
    ...(lines.length > 1 ? [h("ul", {}, lines.map((x) => h("li", {}, x)))] : lines.map((x) => h("p", {}, x))),
    ...(footnote ? [h("p", { class: "note" }, footnote)] : []));
}

function readable(reason) {
  const [rule, ...rest] = reason.split(": ");
  return RULES[rule] && rest.length ? `${RULES[rule]}: ${rest.join(": ")}` : reason;
}

function renderVerdict(r) {
  const foot = r.cached ? "Recorded earlier from a live run, because today's free quota is used up." : null;
  if (r.decision === "approved") {
    const lines = [];
    if (r.reviewed_by_human) {
      if (r.corrections.length) lines.push(...r.corrections.map((c) => `${label(c.path)}: ${c.before ?? "empty"} → ${c.after ?? "empty"}`));
      if (r.overrides.length) lines.push(`Approved although these rules still fail: ${r.overrides.map((x) => RULES[x] ?? x).join(", ")}`);
      if (!lines.length) lines.push("You approved it without changes.");
    } else lines.push("Both readings agree on every required field, and no rule failed.");
    setVerdict("approved", r.reviewed_by_human ? "Approved by you" : "Approved", lines, foot);
  } else if (r.decision === "review") {
    setVerdict("review", "Needs a person", r.review_reasons.map(readable), foot);
  } else {
    setVerdict("failed", r.reviewed_by_human ? "Rejected" : "Could not read this document", [r.error ?? "Something went wrong."], foot);
  }
}

function label(path) {
  for (const [, rows] of FIELD_GROUPS) for (const [p, l] of rows) if (p === path) return l;
  const [, i, leaf] = path.split(".");
  return i === undefined ? path : `Line ${+i + 1} ${LINE_FIELDS.find(([f]) => f === leaf)?.[1].toLowerCase() ?? leaf}`;
}

function meter(value) {
  if (value === undefined) return h("span", {});
  const pct = Math.floor(value * 100);
  const level = value >= config.confidence_threshold ? "high" : value >= 0.5 ? "mid" : "low";
  const bar = h("i");
  bar.style.width = `${pct}%`;
  return h("div", { class: "meter", data: { level }, role: "img", "aria-label": `Confidence ${pct} percent` },
    h("span", {}, bar), h("b", {}, `${pct}%`));
}

const read = (fields, path) => path.split(".").reduce((o, k) => (o === null || o === undefined ? o : o[k]), fields);
const implicated = (path) => current.checks.some((c) => c.status === "fail" && c.fields.includes(path) && read(current.fields, path) !== null);
const differs = (path) => current.disagreements.includes(path) || (path.startsWith("line_items.") && current.disagreements.includes("line_items"));

function editor(path, value, name) {
  const original = value ?? "";
  return h("input", {
    class: "edit", value: original, name: path, "aria-label": name, maxLength: 500,
    data: { original },
    oninput: (e) => e.target.classList.toggle("changed", e.target.value !== original),
  });
}

function renderFields(editing) {
  const f = current.fields;
  if (!f && !editing) {
    $("fields").replaceChildren(h("p", { class: "note group" }, current.result ? "No fields were extracted." : "Fields appear here as they are read."));
    $("lines").replaceChildren();
    return;
  }
  $("fields").replaceChildren(...FIELD_GROUPS.map(([title, rows]) =>
    h("div", { class: "group" }, h("h3", {}, title),
      rows.filter(([p]) => editing || !OPTIONAL_WHEN_EMPTY.has(p) || read(f, p) !== null).map(([path, name, kind]) => {
        const value = read(f, path);
        const note = differs(path) ? h("em", {}, "the two readings differ") : implicated(path) ? h("em", {}, "part of a failed check") : null;
        const shown = value === null || value === undefined ? h("span", { class: "none" }, "not on the document") : value;
        return h("div", { class: "row" },
          h("span", { class: "k" }, name),
          h("div", { class: `v ${kind ?? ""}` }, editing ? editor(path, value, name) : shown, note),
          meter(current.confidence[path]));
      }))));
  const items = f?.line_items ?? [];
  $("lines").replaceChildren(h("div", { class: "group" }, h("h3", {}, "Line items"),
    items.length === 0 ? h("p", { class: "note" }, "None found.") :
    h("div", { class: "scroll" }, h("table", { class: "lines" },
      h("thead", {}, h("tr", {}, LINE_FIELDS.map(([, n], i) => h("th", { class: i ? "num" : "", scope: "col" }, n)), h("th", { scope: "col" }, "Confidence"))),
      h("tbody", {}, items.map((line, i) => {
        const paths = LINE_FIELDS.map(([leaf]) => `line_items.${i}.${leaf}`);
        const scores = paths.map((p) => current.confidence[p]).filter((x) => x !== undefined);
        const off = paths.filter(differs);
        return h("tr", {},
          LINE_FIELDS.map(([leaf, name], j) => h("td", { class: j ? "num" : "" },
            editing ? editor(paths[j], line[leaf], `Line ${i + 1} ${name}`) : (line[leaf] ?? "missing"),
            j === 0 && off.length ? h("em", {}, "the two readings differ") : null)),
          h("td", {}, meter(scores.length ? Math.min(...scores) : undefined)));
      }))))));
}

function renderChecks(checks) {
  $("checks").replaceChildren(...checks.map((c) => h("li", {},
    h("span", { class: "badge", data: { s: c.status, icon: ICON[c.status] } }, WORD[c.status]),
    h("div", {}, h("b", {}, RULES[c.rule] ?? c.rule), h("span", {}, c.detail)))));
}

function renderReview(show) {
  const box = $("review");
  box.hidden = !show;
  if (!show) return;
  const error = h("p", { class: "error", role: "alert", hidden: true });
  const submit = async (action) => {
    const buttons = box.querySelectorAll("button");
    buttons.forEach((b) => (b.disabled = true));
    error.hidden = true;
    const corrections = {};
    for (const input of $("fields-form").querySelectorAll("input.edit")) {
      if (input.value !== input.dataset.original) corrections[input.name] = input.value.trim() === "" ? null : input.value.trim();
    }
    try {
      const result = await api(`/v1/runs/${current.runId}/review`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ action, corrections: action === "approve" ? corrections : {} }),
      });
      finish(current, result);
      $("verdict").scrollIntoView({ block: "center" });
    } catch (e) {
      error.textContent = e.message;
      error.hidden = false;
      buttons.forEach((b) => (b.disabled = false));
    }
  };
  $("fields-form").onsubmit = (e) => { e.preventDefault(); submit("approve"); };
  box.replaceChildren(
    h("p", {}, h("b", {}, "The run is paused for you. "), "Edit any field above, then approve. The rules run again on what you approve."),
    error,
    h("div", { class: "buttons" },
      h("button", { type: "submit", class: "primary" }, "Approve"),
      h("button", { type: "button", class: "quiet danger", onclick: () => submit("reject") }, "Reject document")));
}

function renderMetrics(r) {
  const m = r.metrics;
  if (!m) return;
  const tile = (k, v) => h("div", {}, h("dt", {}, k), h("dd", {}, v));
  const offline = r.mode === "offline";
  $("metrics").replaceChildren(
    tile("Latency", ms(m.latency_ms)),
    tile("Cost at list price", offline ? "No model" : `$${m.cost_usd.toFixed(5)}`),
    tile("Tokens in", m.tokens_in.toLocaleString("en")),
    tile("Tokens out", m.tokens_out.toLocaleString("en")));
  $("calls").replaceChildren(m.llm_calls.length ? h("details", {}, h("summary", {}, `${m.llm_calls.length} reader call${m.llm_calls.length > 1 ? "s" : ""}`),
    h("div", { class: "scroll" }, h("table", {},
      h("thead", {}, h("tr", {}, ["Node", "Model", "In", "Out", "Time"].map((t, i) => h("th", { class: i > 1 ? "num" : "", scope: "col" }, t)))),
      h("tbody", {}, m.llm_calls.map((c) => h("tr", {}, h("td", {}, c.node), h("td", {}, c.model),
        h("td", { class: "num" }, c.tokens_in), h("td", { class: "num" }, c.tokens_out), h("td", { class: "num" }, ms(c.latency_ms)))))))) : "");
  $("trace").replaceChildren(r.trace_url ? h("a", { href: r.trace_url, target: "_blank", rel: "noopener noreferrer" }, "View trace in Langfuse") : "");
}

// --- Evidence ------------------------------------------------------------------------------------

const METRICS = [
  ["field_f1", "Field-level F1", (v) => v.toFixed(3)],
  ["doc_exact", "Documents fully correct", pct],
  ["line_item_f1", "Line-item F1", (v) => v.toFixed(3)],
  ["catch_rate", "Planted errors caught", pct],
  ["false_flag_rate", "Clean documents sent to review", pct],
  ["ece", "Calibration error", (v) => v.toFixed(3)],
  ["injection_changed", "Injections that changed the output", String],
  ["latency_p50_ms", "Latency, median", ms],
  ["latency_p95_ms", "Latency, p95", ms],
  ["cost_per_doc_usd", "Cost per document", (v) => `$${v.toFixed(5)}`],
];
function pct(v) { return `${(v * 100).toFixed(1)}%`; }

async function loadEval() {
  const box = $("eval");
  let report;
  try { report = await api("/v1/eval/latest"); } catch {
    box.replaceChildren(h("p", { class: "note" }, "No evaluation has been published yet. Numbers appear here only after the eval harness has run."));
    return;
  }
  const agent = report.systems.agent?.overall ?? {};
  const base = report.systems.baseline?.overall ?? {};
  const show = (o, k, f) => (o[k] === null || o[k] === undefined ? "n/a" : f(o[k]));
  box.replaceChildren(
    h("p", { class: "note" }, `${report.n_docs} documents, ${report.set} set, ${new Date(report.created_at).toLocaleDateString("en", { dateStyle: "medium" })}, commit ${report.git_sha.slice(0, 7)}.` +
      (report.mode === "offline" ? " Measured in offline mode with rule-based readers, so this says nothing about the AI models." : "")),
    h("div", { class: "scroll" }, h("table", {},
      h("thead", {}, h("tr", {}, ["Metric", "Single prompt", "Agent", "Target"].map((t, i) => h("th", { class: i ? "num" : "", scope: "col" }, t)))),
      h("tbody", {}, METRICS.map(([k, name, f]) => h("tr", {}, h("td", {}, name),
        h("td", { class: "num" }, show(base, k, f)), h("td", { class: "num" }, show(agent, k, f)),
        h("td", { class: "num" }, report.targets?.[k] ?? "")))))));
}

// --- Wiring --------------------------------------------------------------------------------------

$("file").addEventListener("change", (e) => { startFile(e.target.files[0]); e.target.value = ""; });
const drop = $("drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  if (config.uploads_enabled && !$("file").disabled) startFile(e.dataTransfer.files[0]);
});
$("again").addEventListener("click", () => $("start").scrollIntoView({ block: "start" }));
$("copy").addEventListener("click", async (e) => {
  const button = e.currentTarget;
  try { await navigator.clipboard.writeText($("json").textContent); button.textContent = "Copied"; }
  catch { button.textContent = "Copy failed"; }
  setTimeout(() => (button.textContent = "Copy JSON"), 1500);
});

init();
loadEval();
