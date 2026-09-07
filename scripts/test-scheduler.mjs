/**
 * `npm run test:scheduler` — the scheduler's behaviour, asserted.
 *
 * These are the claims the paper makes, so they are worth failing loudly. The
 * suite needs only the standard library, so it runs before torch is installed.
 */

import { ENGINE_DIR, findSystemPython, paint, venvPython, venvReady } from "./lib/env.mjs";
import { spawnSync } from "node:child_process";
import { writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const SUITE = String.raw`
import sys, time, json
sys.path.insert(0, ".")
from coordinator.scheduler import (
    MeshPolicy, admit, aggregation_weights, deadline_for, efficiency,
    plan_round, should_abort_round, straggler_action, update_reliability,
    update_throughput,
)

failures = []

def expect(name, condition, detail=""):
    if condition:
        print("  ok   " + name)
    else:
        failures.append(name)
        print("  FAIL " + name + ("  " + detail if detail else ""))

now = time.time()

def node(nid, gflops, mem=8192, thr=0.0, rel=0.85, lat=5.0, active=True, training=True):
    return {
        "node_id": nid, "capability": {"gflops": gflops}, "gpu_memory_mb": mem,
        "allocated_memory_mb": 0, "last_seen": now, "active": active,
        "supports_training": training, "reliability": rel, "latency_ms": lat,
        "throughput_sps": thr,
    }

# 1. A heterogeneous mesh finishes together rather than waiting on the slowest.
pool = [node("fast", 4200), node("mid", 1800), node("slow", 700)]
plan = plan_round(pool, 1200)
times = [a.predicted_seconds for a in plan.assignments]
expect("shards equalise predicted finish time", max(times) - min(times) <= 0.5 * max(times) * 0.1,
       json.dumps(times))
expect("every sample is assigned exactly once",
       sum(a.samples for a in plan.assignments) == 1200)
expect("the fastest node carries the largest shard",
       max(plan.assignments, key=lambda a: a.samples).node_id == "fast")
expect("predicted speedup beats a single GPU", plan.predicted_speedup > 1.0,
       str(plan.predicted_speedup))

# 2. Adding a weak machine must not slow the mesh down.
plan_without = plan_round([node("fast", 4200), node("mid", 1800)], 1200)
expect("adding a weak node does not raise makespan",
       plan.predicted_makespan_seconds <= plan_without.predicted_makespan_seconds + 1e-6,
       "%.2f vs %.2f" % (plan.predicted_makespan_seconds, plan_without.predicted_makespan_seconds))

# 3. Admission control keeps unusable hardware out, with a reason.
gate = admit([node("tiny", 4000, mem=512), node("cpu-only", 0, training=False), node("good", 3000)])
expect("a device below the memory floor is rejected", gate["tiny"].tier == "rejected")
expect("a device with no accelerator is rejected", gate["cpu-only"].tier == "rejected")
expect("every decision carries a human reason", all(d.reason for d in gate.values()))
expect("a capable device is admitted", gate["good"].tier == "full")

# 4. A single node still forms a valid mesh.
solo = plan_round([node("only", 2500)], 300)
expect("a single node receives the whole dataset",
       len(solo.assignments) == 1 and solo.assignments[0].samples == 300)

# 5. Straggler policy escalates in the right order.
d = deadline_for(60.0)
expect("a worker inside its deadline is left alone", straggler_action(30, d, True) == "wait")
expect("a soft miss with an idle peer speculates", straggler_action(d.soft_seconds + 1, d, True) == "speculate")
expect("a soft miss with no idle peer waits", straggler_action(d.soft_seconds + 1, d, False) == "wait")
expect("a hard miss drops the shard", straggler_action(d.hard_seconds + 1, d, True) == "drop")
cold = deadline_for(60.0, cold=True)
expect("a first round gets a much longer deadline",
       cold.soft_seconds >= MeshPolicy().cold_start_grace_seconds and cold.soft_seconds > d.soft_seconds,
       "%.0f vs %.0f" % (cold.soft_seconds, d.soft_seconds))
expect("a warm round is unaffected by the cold grace", d.soft_seconds == deadline_for(60.0).soft_seconds)
expect("a first round is not declared late at its own predicted time",
       straggler_action(70, cold, True) == "wait")

expect("losing most of the data aborts the round", should_abort_round(700, 1000))
expect("losing a little data does not abort", not should_abort_round(50, 1000))

# 6. Aggregation weights follow contribution.
weights = aggregation_weights([
    {"batch_id": "big", "samples": 800, "reliability": 1.0},
    {"batch_id": "small", "samples": 200, "reliability": 1.0},
])
expect("weights sum to one", abs(sum(weights.values()) - 1.0) < 1e-9)
expect("the larger shard carries more weight", weights["big"] > weights["small"])
expect("weights are proportional to samples", abs(weights["big"] - 0.8) < 1e-9)

penalised = aggregation_weights([
    {"batch_id": "trusted", "samples": 500, "reliability": 1.0},
    {"batch_id": "flaky", "samples": 500, "reliability": 0.4},
])
expect("an unreliable worker is damped", penalised["trusted"] > penalised["flaky"])

# 7. The throughput model learns from observation.
learner = node("learner", 1000)
first = update_throughput(learner, 100, 10.0)
expect("the first observation is taken at face value", abs(first - 10.0) < 1e-9)
learner["throughput_sps"] = first
second = update_throughput(learner, 100, 5.0)
expect("later observations are smoothed", 10.0 < second < 20.0, str(second))

expect("success raises reliability", update_reliability(node("n", 1, rel=0.5), True) > 0.5)
expect("failure lowers reliability", update_reliability(node("n", 1, rel=0.5), False) < 0.5)
expect("reliability stays bounded", update_reliability(node("n", 1, rel=0.99), True) <= 1.0)

# 8. An empty or offline mesh degrades safely rather than crashing.
expect("an empty mesh plans nothing", plan_round([], 500).assignments == [])
expect("an all-offline mesh plans nothing",
       plan_round([node("gone", 3000, active=False)], 500).assignments == [])
expect("a zero-sample dataset plans nothing", plan_round(pool, 0).assignments == [])

# 9. Policy is honoured.
strict = MeshPolicy(min_memory_mb=16000)
expect("a stricter memory floor rejects more devices",
       all(d.tier == "rejected" for d in admit(pool, strict).values()))

expect("efficiency is speedup over worker count", abs(efficiency(2.0, 4) - 0.5) < 1e-9)

print("")
if failures:
    print("%d assertion(s) failed" % len(failures))
    sys.exit(1)
print("scheduler suite passed")
`;

const python = venvReady() ? venvPython() : null;
const fallback = findSystemPython();

if (!python && !fallback) {
  console.error(paint("red", "No Python found. Run `npm run setup` first."));
  process.exit(1);
}

console.log(paint("bold", "\nGradMesh scheduler\n"));

// Writing the suite to a file rather than passing it with -c keeps Windows
// argument quoting out of the picture entirely.
const suitePath = path.join(tmpdir(), `gradmesh-scheduler-${process.pid}.py`);
writeFileSync(suitePath, SUITE, "utf8");

const interpreter = python ? { command: python, args: [] } : fallback;
const result = spawnSync(interpreter.command, [...interpreter.args, suitePath], {
  cwd: ENGINE_DIR,
  stdio: "inherit",
});

rmSync(suitePath, { force: true });
process.exit(result.status ?? 1);
