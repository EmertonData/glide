import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { simulate, getCorrelationBounds } from "../ppi.js";

const fixtures = JSON.parse(readFileSync(new URL("./fixtures.json", import.meta.url)));

test("simulate matches glide-py on a large synthetic dataset", () => {
  for (const fixture of fixtures.simulate) {
    const output = simulate(fixture.params);
    const ppiError = Math.abs(output.ppi.halfWidth - fixture.ppi_half_width) / fixture.ppi_half_width;
    assert.ok(ppiError < 0.02);
    const humanError = Math.abs(output.humanOnly.halfWidth - fixture.human_half_width) / fixture.human_half_width;
    assert.ok(humanError < 0.02);
    const essError = Math.abs(output.effectiveSampleSize - fixture.effective_sample_size) / fixture.effective_sample_size;
    assert.ok(essError < 0.05);
  }
});

test("simulate with zero correlation recovers the human-only estimate", () => {
  const output = simulate({ totalSize: 3300, humanSize: 265, trueMean: 0.77, proxyMean: 0.76, correlation: 0 });
  assert.ok(Math.abs(output.ppi.halfWidth - output.humanOnly.halfWidth) < 1e-12);
  assert.equal(output.effectiveSampleSize, 265);
});

test("simulate raises when humanSize is out of range", () => {
  assert.throws(
    () => simulate({ totalSize: 100, humanSize: 100, trueMean: 0.7, proxyMean: 0.6, correlation: 0.5 }),
    /'humanSize' must satisfy 0 < humanSize < totalSize/,
  );
});

test("simulate raises when trueMean is out of range", () => {
  assert.throws(
    () => simulate({ totalSize: 100, humanSize: 10, trueMean: 1, proxyMean: 0.6, correlation: 0.5 }),
    /'trueMean' must be in \(0, 1\)/,
  );
});

test("simulate raises when correlation is out of range", () => {
  assert.throws(
    () => simulate({ totalSize: 100, humanSize: 10, trueMean: 0.7, proxyMean: 0.6, correlation: -0.5 }),
    /'correlation' must be in \[0, 1\]/,
  );
});

test("simulate clamps infeasible correlation", () => {
  const output = simulate({ totalSize: 100, humanSize: 10, trueMean: 0.6, proxyMean: 0.8, correlation: 1 });
  assert.ok(Number.isFinite(output.ppi.halfWidth));
  assert.ok(!Number.isNaN(output.ppi.halfWidth));
});

test("getCorrelationBounds for equal accuracies", () => {
  const bounds = getCorrelationBounds(0.7, 0.7);
  assert.ok(Math.abs(bounds.minCorrelation - -0.42857142857142866) < 1e-9);
  assert.equal(bounds.maxCorrelation, 1);
});

test("getCorrelationBounds for unequal accuracies", () => {
  const bounds = getCorrelationBounds(0.6, 0.8);
  assert.ok(Math.abs(bounds.minCorrelation - -0.4082482904638634) < 1e-9);
  assert.ok(Math.abs(bounds.maxCorrelation - 0.6123724356957946) < 1e-9);
});
