// Behaviour check for agrisense/gps/frontend/index.html (run by tests/test_core.py; needs only Node.js).
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const assert = require("assert");

const html = fs.readFileSync(path.join(__dirname, "..", "agrisense", "gps", "frontend", "index.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function load(geolocation, alreadyTried) {
  const sent = [];
  const handlers = {};
  const btn = { disabled: false, textContent: "", on: {}, addEventListener(type, fn) { this.on[type] = fn; } };
  const parent = { postMessage: (message) => sent.push(message) };
  if (alreadyTried) parent.__agrisenseGpsTried = true;
  const win = { parent, addEventListener: (type, fn) => { handlers[type] = fn; } };
  const doc = { getElementById: () => btn, documentElement: { dir: "" }, body: { style: {} } };
  vm.runInNewContext(script, { window: win, document: doc, navigator: geolocation ? { geolocation } : {} });
  return {
    sent, btn, parent, doc,
    render: (args, theme) => handlers.message({ data: { type: "streamlit:render", args, theme } }),
    values: () => sent.filter((m) => m.type === "streamlit:setComponentValue").map((m) => m.value),
  };
}
const answer = (reply) => {            // a geolocation object that answers at once
  const geo = { calls: 0, getCurrentPosition(ok, fail) { geo.calls += 1; reply.coords ? ok(reply) : fail(reply); } };
  return geo;
};
const FIX = { coords: { latitude: 30.5, longitude: 31.2, accuracy: 20 } };
const ARGS = { texts: { button: "BTN", wait: "WAIT" }, auto: true, rtl: true };

// announces itself to Streamlit
let c = load(answer(FIX));
assert.strictEqual(c.sent[0].type, "streamlit:componentReady");
assert.strictEqual(c.sent[0].apiVersion, 1);
assert.ok(c.sent.every((m) => m.isStreamlitMessage === true));
assert.ok(c.sent.some((m) => m.type === "streamlit:setFrameHeight"));

// asks once on the first render, sends coordinates, applies texts / direction / theme
let geo = answer(FIX);
c = load(geo);
c.render(ARGS, { textColor: "#111" });
assert.strictEqual(geo.calls, 1);
assert.deepStrictEqual(Object.keys(c.values()[0]).sort(), ["accuracy", "lat", "lon", "ts"]);
assert.strictEqual(c.values()[0].lat, 30.5);
assert.strictEqual(c.values()[0].lon, 31.2);
assert.strictEqual(c.values()[0].accuracy, 20);
assert.strictEqual(c.btn.textContent, "BTN");
assert.strictEqual(c.btn.disabled, false);
assert.strictEqual(c.doc.documentElement.dir, "rtl");
assert.strictEqual(c.doc.body.style.color, "#111");
assert.strictEqual(c.parent.__agrisenseGpsTried, true);
c.render(ARGS);
assert.strictEqual(geo.calls, 1, "must not ask again on the next render");

// the button asks again; every answer has its own ts
c.btn.on.click();
assert.strictEqual(geo.calls, 2);
assert.strictEqual(c.values().length, 2);
assert.notStrictEqual(c.values()[0].ts, c.values()[1].ts);

// no automatic ask when the page already asked, or when auto is off
geo = answer(FIX); c = load(geo, true); c.render(ARGS);
assert.strictEqual(geo.calls, 0);
c.btn.on.click();
assert.strictEqual(geo.calls, 1);
geo = answer(FIX); c = load(geo); c.render({ ...ARGS, auto: false });
assert.strictEqual(geo.calls, 0);

// button is disabled with the waiting text until the browser answers
const pending = { getCurrentPosition(ok) { pending.ok = ok; } };
c = load(pending, true); c.render(ARGS); c.btn.on.click();
assert.strictEqual(c.btn.disabled, true);
assert.strictEqual(c.btn.textContent, "WAIT");
pending.ok(FIX);
assert.strictEqual(c.btn.disabled, false);
assert.strictEqual(c.btn.textContent, "BTN");

// errors: denied / failed / unsupported
c = load(answer({ code: 1 })); c.render(ARGS);
assert.strictEqual(c.values()[0].error, "denied");
for (const code of [2, 3]) { c = load(answer({ code })); c.render(ARGS); assert.strictEqual(c.values()[0].error, "failed"); }
c = load(null); c.render(ARGS);
assert.strictEqual(c.values()[0].error, "unsupported");

// messages that are not render events are ignored
c = load(answer(FIX), true); c.render(ARGS);
assert.doesNotThrow(() => c.render(ARGS));
console.log("gps component OK");
