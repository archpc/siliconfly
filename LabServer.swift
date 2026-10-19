// LabServer.swift — headless experimental-control server for SiliconFly.
//
// Protocol: one JSON command per line on stdin, one JSON response per line on
// stdout. stderr keeps the sim's own diagnostics. The command loop is strictly
// single-threaded and every MetalSim operation is synchronous, so no additional
// locking is needed: GPU work is serialized by construction.
//
// Determinism: the sim is Q16 fixed-point with order-independent integer
// atomics, and `reset` recreates the MetalSim from scratch with the requested
// seed (the only state a fresh sim owns is seed-derived), so identical command
// sequences from an identical seed produce bit-identical results.
//
// Modes (set from main.swift):
//   SiliconFly --serve                JSON-lines server, real connectome
//   SiliconFly --serve --shuffle N    weights-shuffled control connectome
//
// Weight shuffle: permutes the int16 edge weights with a seeded LCG
// (Fisher-Yates). The multiset of weights is preserved exactly (global
// distribution, sign balance, fixed-point overflow bound), in/out degree
// sequences are untouched, and roles/groups are unchanged — only the wiring
// specificity is destroyed. This mirrors the connectivity-weight shuffle
// control in Shiu et al. 2024. A pure neuron relabeling would NOT be a valid
// control here: input targeting (loom drive, sensory input) is keyed on roles,
// which would permute along with everything else and leave dynamics unchanged.

import Foundation

let LAB_VERSION = "1.0.0"

// MARK: - helpers

private func respond(_ obj: [String: Any]) {
    if let data = try? JSONSerialization.data(withJSONObject: obj),
       let line = String(data: data, encoding: .utf8) {
        print(line)
        fflush(stdout)
    }
}

private func fail(_ id: Any?, _ code: String, _ message: String) {
    respond(["ok": false, "id": id as Any? ?? NSNull(), "error": ["code": code, "message": message]])
}

private func ok(_ id: Any?, _ fields: [String: Any]) {
    var out: [String: Any] = ["ok": true, "id": id as Any? ?? NSNull()]
    for (k, v) in fields { out[k] = v }
    respond(out)
}

private func num(_ d: [String: Any], _ k: String) -> Float? {
    (d[k] as? NSNumber)?.floatValue
}

private func clampF(_ v: Float, _ lo: Float, _ hi: Float) -> Float {
    return min(max(v, lo), hi)
}

// MARK: - weight shuffle

/// Seeded 32-bit LCG (Numerical Recipes constants). Deterministic across runs.
private func makeShuffleRng(_ seed: UInt32) -> () -> UInt32 {
    var s = seed == 0 ? 0x9E37_79B9 : seed
    return {
        s = s &* 1_664_525 &+ 1_013_904_223
        return s
    }
}

private func shuffleConnectomeWeights(_ c: Connectome, seed: UInt32) -> Connectome? {
    let e = c.e
    var w = c.weightData.withUnsafeBytes { buf -> [Int16] in
        [Int16](UnsafeBufferPointer(start: buf.baseAddress!.assumingMemoryBound(to: Int16.self),
                                    count: e))
    }
    let next = makeShuffleRng(seed)
    var i = w.count - 1
    while i > 0 {
        let j = Int(next() % UInt32(i + 1))
        w.swapAt(i, j)
        i -= 1
    }
    let data = w.withUnsafeBufferPointer { Data(buffer: $0) }
    return Connectome(n: c.n, e: c.e, positions: c.positions, superClass: c.superClass,
                      side: c.side, nt: c.nt, role: c.role, rootId: c.rootId,
                      rowStart: c.rowStart, colIdxData: c.colIdxData, weightData: data,
                      superClassNames: c.superClassNames, ntNames: c.ntNames,
                      isModulatory: c.isModulatory, roleIndices: c.roleIndices,
                      roleName: c.roleName, typeName: c.typeName, loadMs: c.loadMs)
}

// MARK: - server

private let GROUP_MAP: [String: (MetalSim) -> [Int]] = [
    "gf":     { $0.gf },
    "dnaL":   { $0.dnaL },
    "dnaR":   { $0.dnaR },
    "mdn":    { $0.mdn },
    "fwd":    { $0.fwd },
    "groom":  { $0.groom },
    "escw":   { $0.escw },
    "loomL":  { $0.loomLeft },
    "loomR":  { $0.loomRight },
    "loom":   { $0.loomLeft + $0.loomRight },
    "sens":   { $0.sens },
    "ascend": { $0.ascend },
]

private let MAX_STEP_MS = 120_000
private let MAX_STIM_MS = 5_000
private let MAX_TRIAL_MS = 10_000

private var sim: MetalSim? = nil
private var simSeed: UInt32 = 0
private var shuffled = false
private var shuffleSeed: UInt32 = 0
private var connectome: Connectome? = nil       // pristine (with shuffle if --shuffle)
private var activeConnectome: Connectome? = nil // pristine or ablated; reset uses this
private var ablationDesc: String? = nil
private var ablatedTargets: Set<Int> = []

// Group membership derived from the connectome arrays (NOT from the live sim's
// groups, which may already be ablated). loomL/R use the `side` table.
private func groupIndices(in c: Connectome, named g: String) -> [Int]? {
    var result: [Int] = []
    for i in 0..<c.n {
        let left = c.side[i] == 1
        switch g {
        case "gf":     if c.role[i] == Role.gf { result.append(i) }
        case "dnaL":   if (c.role[i] == Role.dna01 || c.role[i] == Role.dna02) && left { result.append(i) }
        case "dnaR":   if (c.role[i] == Role.dna01 || c.role[i] == Role.dna02) && !left { result.append(i) }
        case "mdn":    if c.role[i] == Role.mdn { result.append(i) }
        case "fwd":    if c.role[i] == Role.dnp09 { result.append(i) }
        case "groom":  if c.role[i] == Role.dng11 { result.append(i) }
        case "escw":   if c.role[i] == Role.escw { result.append(i) }
        case "loomL":  if (c.role[i] == Role.lc4 || c.role[i] == Role.lplc2) && left { result.append(i) }
        case "loomR":  if (c.role[i] == Role.lc4 || c.role[i] == Role.lplc2) && !left { result.append(i) }
        case "loom":   if c.role[i] == Role.lc4 || c.role[i] == Role.lplc2 { result.append(i) }
        case "sens":   if c.role[i] == Role.sens { result.append(i) }
        case "ascend": if c.role[i] == Role.ascend { result.append(i) }
        default: return nil
        }
    }
    return result
}

/// Ablated copy of the connectome: the target neurons' outgoing edge weights are
/// zeroed. Everything else (topology, other weights, fixed-point overflow bound
/// via preserved multiset) is untouched. Always derived from the PRISTINE
/// connectome so ablations never compound.
private func ablateConnectome(_ c: Connectome, targets: Set<Int>) -> Connectome {
    let e = c.e
    var w = c.weightData.withUnsafeBytes { buf -> [Int16] in
        [Int16](UnsafeBufferPointer(start: buf.baseAddress!.assumingMemoryBound(to: Int16.self),
                                    count: e))
    }
    var zeroed = 0
    for i in targets {
        for k in Int(c.rowStart[i])..<Int(c.rowStart[i + 1]) {
            w[k] = 0
            zeroed += 1
        }
    }
    let data = w.withUnsafeBufferPointer { Data(buffer: $0) }
    fputs("lab-server: ablation zeroed \(zeroed) outgoing edges of \(targets.count) neurons\n", stderr)
    return Connectome(n: c.n, e: c.e, positions: c.positions, superClass: c.superClass,
                      side: c.side, nt: c.nt, role: c.role, rootId: c.rootId,
                      rowStart: c.rowStart, colIdxData: c.colIdxData, weightData: data,
                      superClassNames: c.superClassNames, ntNames: c.ntNames,
                      isModulatory: c.isModulatory, roleIndices: c.roleIndices,
                      roleName: c.roleName, typeName: c.typeName, loadMs: c.loadMs)
}

private func makeSim(seed: UInt32) -> Bool {
    guard let c = activeConnectome else { return false }
    guard let s = MetalSim(connectome: c, spikeBus: nil, seed: seed) else { return false }
    s.perfLogIntervalMs = 0
    // Re-apply intrinsic-drive zeroing after every sim (re)creation: reset()
    // redraws baselines from the seed, which would otherwise resurrect the
    // ablated neurons' resting drive.
    if !ablatedTargets.isEmpty {
        var b = s.baselineValues()
        for i in ablatedTargets { b[i] = 0 }
        s.setBaseline(b)
    }
    sim = s
    simSeed = seed
    return true
}

private func ratesDict(_ s: MetalSim) -> [String: Any] {
    ["loom": s.rateLoom, "dnaL": s.rateDNaL, "dnaR": s.rateDNaR,
     "mdn": s.rateMDN, "fwd": s.rateFwd, "groom": s.rateGroom,
     "escw": s.rateEscW, "pop": s.ratePop]
}

private func groupSizes(_ s: MetalSim) -> [String: Any] {
    ["loomL": s.loomLeft.count, "loomR": s.loomRight.count, "gf": s.gf.count,
     "dnaL": s.dnaL.count, "dnaR": s.dnaR.count, "mdn": s.mdn.count,
     "fwd": s.fwd.count, "groom": s.groom.count, "escw": s.escw.count,
     "ascend": s.ascend.count, "sens": s.sens.count]
}

/// The escape-latency trial primitive, following --simtest's Phase-2 pattern:
/// settle (all sensory input zeroed), then hold the requested loom drive for
/// `duration_ms` one-ms steps, then `post_ms` more steps of silence, recording
/// per-step GF / LC / total-population spike counts the whole way.
private func runTrial(_ d: [String: Any], _ id: Any?) {
    guard let s = sim else { fail(id, "no_sim", "call reset first"); return }
    let settle = max(0, min(MAX_TRIAL_MS, Int((d["settle_ms"] as? NSNumber)?.intValue ?? 2000)))
    let duration = Int((d["duration_ms"] as? NSNumber)?.intValue ?? 400)
    guard duration >= 1, duration <= MAX_TRIAL_MS else {
        fail(id, "bad_duration", "duration_ms must be 1...\(MAX_TRIAL_MS)"); return
    }
    let post = max(0, min(MAX_TRIAL_MS, Int((d["post_ms"] as? NSNumber)?.intValue ?? 200)))
    let loomL = clampF(num(d, "loomL") ?? 1.0, 0, 1)
    let loomR = clampF(num(d, "loomR") ?? 0, 0, 1)

    // settle with clean sensory input
    s.loomL = 0; s.loomR = 0; s.airPuff = 0
    _ = s.consumeGF()
    if settle > 0 { s.step(settle) }
    _ = s.consumeGF()
    let preRates = ratesDict(s)
    let preSimMs = s.simMs

    var gfCounts = [Int](), loomCounts = [Int](), popCounts = [Int]()
    gfCounts.reserveCapacity(duration + post)
    loomCounts.reserveCapacity(duration + post)
    popCounts.reserveCapacity(duration + post)
    var spikesBefore = s.totalSpikes
    var firstGfMs = -1
    var gfTotal = 0
    var loomTotal = 0
    for ms in 0..<(duration + post) {
        if ms < duration {
            s.loomL = loomL
            s.loomR = loomR
        } else {
            s.loomL = 0
            s.loomR = 0
        }
        s.step(1)
        let hist = s.lastStepGroupCounts()
        let totalNow = s.totalSpikes
        popCounts.append(totalNow - spikesBefore)
        spikesBefore = totalNow
        let gf = Int(hist[2])           // Group.gf
        let lc = Int(hist[1])           // Group.loom
        gfCounts.append(gf)
        loomCounts.append(lc)
        if gf > 0 {
            gfTotal += gf
            if firstGfMs < 0 { firstGfMs = ms }
        }
        loomTotal += lc
    }

    let postRates = ratesDict(s)
    ok(id, ["seed": simSeed, "shuffled": shuffled, "shuffle_seed": shuffleSeed,
            "settle_ms": settle, "duration_ms": duration, "post_ms": post,
            "loomL": loomL, "loomR": loomR,
            "pre_sim_ms": preSimMs, "sim_ms": s.simMs,
            "pre_rates": preRates, "post_rates": postRates,
            "first_gf_ms": firstGfMs, "gf_total": gfTotal, "loom_total": loomTotal,
            "gf_counts": gfCounts, "loom_counts": loomCounts])
}

/// Stimulus trial primitive for explicitly targeted neurons (Operation Banana
/// Peel): settle, then apply `stimulate()` current drive to the requested
/// indices for `duration_ms`, then `post_ms` of silence. Records per-step
/// GF/LC/pop counts AND per-step spike membership of the watched neurons
/// (e.g. the MN9 readout). Uses the existing extInput drive path — no kernel
/// changes; stimulus is deterministic per-step drive, the same drive class the
/// loom port uses (NOT Shiu's 200 Hz Poisson draw; documented in the report).
private func runStimTrial(_ d: [String: Any], _ id: Any?) {
    guard let s = sim else { fail(id, "no_sim", "call reset first"); return }
    let settle = max(0, min(MAX_TRIAL_MS, Int((d["settle_ms"] as? NSNumber)?.intValue ?? 2000)))
    let duration = Int((d["duration_ms"] as? NSNumber)?.intValue ?? 400)
    guard duration >= 1, duration <= MAX_TRIAL_MS else {
        fail(id, "bad_duration", "duration_ms must be 1...\(MAX_TRIAL_MS)"); return
    }
    let post = max(0, min(MAX_TRIAL_MS, Int((d["post_ms"] as? NSNumber)?.intValue ?? 200)))
    let strength = clampF(num(d, "strength") ?? 0, 0, 2)
    guard let ns = d["neurons"] as? [NSNumber], !ns.isEmpty, ns.count <= 512 else {
        fail(id, "bad_target", "neurons required (1..512)"); return
    }
    let targets = ns.map { $0.intValue }
    guard targets.allSatisfy({ $0 >= 0 && $0 < s.n }) else {
        fail(id, "bad_neuron_id", "ids must be 0..<\(s.n)"); return
    }
    var watch = (d["watch"] as? [NSNumber])?.map { $0.intValue } ?? []
    guard watch.count <= 256, watch.allSatisfy({ $0 >= 0 && $0 < s.n }) else {
        fail(id, "bad_watch", "watch ids must be valid, at most 256"); return
    }

    // settle with zeroed sensory input
    s.loomL = 0; s.loomR = 0; s.airPuff = 0
    _ = s.consumeGF()
    if settle > 0 { s.step(settle) }
    _ = s.consumeGF()
    let preRates = ratesDict(s)
    let preSimMs = s.simMs

    var gfCounts = [Int](), loomCounts = [Int](), popCounts = [Int]()
    var watchPerStep = [Int]()
    var firstWatchMs = [Int](repeating: -1, count: watch.count)
    var watchTotals = [Int](repeating: 0, count: watch.count)
    var spikesBefore = s.totalSpikes
    var gfTotal = 0
    s.stimulate(targets, strength: strength, durationMs: duration)
    for ms in 0..<(duration + post) {
        s.step(1)
        let hist = s.lastStepGroupCounts()
        let totalNow = s.totalSpikes
        popCounts.append(totalNow - spikesBefore)
        spikesBefore = totalNow
        let gf = Int(hist[2])
        gfCounts.append(gf)
        gfTotal += gf
        loomCounts.append(Int(hist[1]))
        let spikeList = s.lastStepSpikes()
        var wThis = 0
        for (wi, wn) in watch.enumerated() where spikeList.contains(Int32(wn)) {
            wThis += 1
            watchTotals[wi] += 1
            if firstWatchMs[wi] < 0 { firstWatchMs[wi] = ms }
        }
        watchPerStep.append(wThis)
    }
    let postRates = ratesDict(s)
    ok(id, ["seed": simSeed, "shuffled": shuffled, "shuffle_seed": shuffleSeed,
            "settle_ms": settle, "duration_ms": duration, "post_ms": post,
            "strength": strength, "targeted": targets.count,
            "pre_sim_ms": preSimMs, "sim_ms": s.simMs,
            "pre_rates": preRates, "post_rates": postRates,
            "gf_total": gfTotal,
            "gf_counts": gfCounts, "loom_counts": loomCounts, "pop_counts": popCounts,
            "watch_ids": watch, "watch_totals": watchTotals,
            "watch_first_ms": firstWatchMs, "watch_per_step": watchPerStep])
}

private func handle(_ line: String) {
    guard let data = line.data(using: .utf8),
          let obj = try? JSONSerialization.jsonObject(with: data),
          let d = obj as? [String: Any],
          let cmd = d["cmd"] as? String else {
        fail(nil, "bad_request", "each stdin line must be a JSON object with a 'cmd' field")
        return
    }
    let id = d["id"] as Any? ?? NSNull()

    switch cmd {
    case "hello":
        guard let s = sim, let c = connectome else {
            fail(id, "no_sim", "sim not initialized"); return
        }
        ok(id, ["version": LAB_VERSION, "n": s.n, "e": c.e, "device": s.deviceName,
                "seed": simSeed, "shuffled": shuffled, "shuffle_seed": shuffleSeed,
                "ablation": ablationDesc ?? "none",
                "fixed_point_q": Int(log2(Double(s.fixedPointScale))),
                "groups": groupSizes(s)])

    case "reset":
        let seed = (d["seed"] as? NSNumber)?.uint32Value ?? simSeed
        guard makeSim(seed: seed) else { fail(id, "sim_init", "MetalSim init failed"); return }
        ok(id, ["seed": simSeed, "sim_ms": 0, "total_spikes": 0])

    case "set_input":
        guard let s = sim else { fail(id, "no_sim", "call reset first"); return }
        var applied: [String: Any] = [:]
        if let v = num(d, "loomL") { s.loomL = clampF(v, 0, 1); applied["loomL"] = s.loomL }
        if let v = num(d, "loomR") { s.loomR = clampF(v, 0, 1); applied["loomR"] = s.loomR }
        if let v = num(d, "airPuff") { s.airPuff = clampF(v, 0, 1); applied["airPuff"] = s.airPuff }
        if let v = num(d, "gaitDrive") { s.gaitDrive = clampF(v, 0, 1); applied["gaitDrive"] = s.gaitDrive }
        if let v = num(d, "gaitPhase") { s.gaitPhase = clampF(v, 0, 1); applied["gaitPhase"] = s.gaitPhase }
        if let v = num(d, "activityScale") { s.activityScale = clampF(v, 0, 1.5); applied["activityScale"] = s.activityScale }
        if let v = num(d, "sensoryGate") { s.sensoryGate = clampF(v, 0, 1); applied["sensoryGate"] = s.sensoryGate }
        ok(id, ["applied": applied, "sim_ms": s.simMs])

    case "stimulate":
        guard let s = sim else { fail(id, "no_sim", "call reset first"); return }
        let strength = clampF(num(d, "strength") ?? 0, 0, 2)
        let duration = Int((d["duration_ms"] as? NSNumber)?.intValue ?? 0)
        guard duration >= 1, duration <= MAX_STIM_MS else {
            fail(id, "bad_duration", "duration_ms must be 1...\(MAX_STIM_MS)"); return
        }
        var indices: [Int] = []
        if let g = d["group"] as? String {
            guard let map = GROUP_MAP[g] else {
                fail(id, "bad_group", "unknown group '\(g)'; valid: \(GROUP_MAP.keys.sorted().joined(separator: ","))")
                return
            }
            indices = map(s)
        } else if let ns = d["neurons"] as? [NSNumber] {
            indices = ns.map { $0.intValue }
        } else {
            fail(id, "bad_target", "provide 'group' or 'neurons'"); return
        }
        guard indices.allSatisfy({ $0 >= 0 && $0 < s.n }) else {
            fail(id, "bad_neuron_id", "neuron ids must be 0..<\(s.n)"); return
        }
        guard !indices.isEmpty else { fail(id, "empty_target", "target set is empty"); return }
        s.stimulate(indices, strength: strength, durationMs: duration)
        ok(id, ["targeted": indices.count, "strength": strength, "duration_ms": duration,
                "sim_ms": s.simMs])

    case "ablate":
        guard let c = connectome else { fail(id, "no_connectome", "not initialized"); return }
        var targets = Set<Int>()
        var desc = ""
        if let g = d["group"] as? String {
            guard let idxs = groupIndices(in: c, named: g), !idxs.isEmpty else {
                fail(id, "bad_group", "unknown or empty group '\(g)'"); return
            }
            targets.formUnion(idxs)
            desc = "ablate:\(g)(\(idxs.count))"
        } else if let rnd = d["random"] as? NSNumber {
            // Sham control: random size-matched slice of the unannotated ('other')
            // population, seeded so the same sham is reproducible.
            let count = max(1, min(c.n, rnd.intValue))
            let shamSeed = (d["seed"] as? NSNumber)?.uint32Value ?? 1
            let others = (0..<c.n).filter { c.role[$0] == Role.other }
            guard count <= others.count else {
                fail(id, "bad_count", "sham size \(count) exceeds \(others.count) 'other' neurons"); return
            }
            let rng = makeShuffleRng(shamSeed)
            var pool = others
            for _ in 0..<count {
                let j = Int(rng() % UInt32(pool.count))
                targets.insert(pool[j])
                pool.swapAt(j, pool.count - 1)
                pool.removeLast()
            }
            desc = "sham:random-other(\(count),seed\(shamSeed))"
        } else if let ns = d["neurons"] as? [NSNumber] {
            let ids = ns.map { $0.intValue }
            guard !ids.isEmpty, ids.allSatisfy({ $0 >= 0 && $0 < c.n }) else {
                fail(id, "bad_neuron_id", "ids must be nonempty and 0..<\(c.n)"); return
            }
            targets.formUnion(ids)
            desc = "ablate:neurons(\(targets.count))"
        } else {
            fail(id, "bad_target", "provide 'group', 'random'(+seed), or 'neurons'"); return
        }
        // Always derived from the PRISTINE connectome: ablations never compound.
        activeConnectome = ablateConnectome(c, targets: targets)
        ablationDesc = desc
        ablatedTargets = targets
        guard makeSim(seed: simSeed) else { fail(id, "sim_init", "ablated sim init failed"); return }
        ok(id, ["ablation": desc, "targeted": targets.count, "seed": simSeed])

    case "ablate_clear":
        activeConnectome = connectome
        ablationDesc = nil
        ablatedTargets = []
        guard makeSim(seed: simSeed) else { fail(id, "sim_init", "restore failed"); return }
        ok(id, ["ablation": "none", "seed": simSeed])

    case "step":
        guard let s = sim else { fail(id, "no_sim", "call reset first"); return }
        let ms = Int((d["ms"] as? NSNumber)?.intValue ?? 0)
        guard ms >= 1, ms <= MAX_STEP_MS else {
            fail(id, "bad_ms", "ms must be 1...\(MAX_STEP_MS)"); return
        }
        s.step(ms)
        ok(id, ["sim_ms": s.simMs, "total_spikes": s.totalSpikes,
                "avg_step_micros": s.avgStepMicros])

    case "trial":
        runTrial(d, id)

    case "stim_trial":
        runStimTrial(d, id)

    case "read":
        guard let s = sim else { fail(id, "no_sim", "call reset first"); return }
        var out: [String: Any] = ["sim_ms": s.simMs, "total_spikes": s.totalSpikes,
                                  "rates": ratesDict(s),
                                  "gf_latched": s.consumeGF() ? true : false,
                                  "groups": groupSizes(s)]
        if d["last_step"] as? Bool == true {
            let hist = s.lastStepGroupCounts()
            out["last_step"] = ["spikes": s.lastStepSpikes().count,
                                "loom": hist[1], "gf": hist[2],
                                "dnaL": hist[3], "dnaR": hist[4],
                                "mdn": hist[5], "fwd": hist[6],
                                "groom": hist[7], "escw": hist[8]]
        }
        ok(id, out)

    case "membrane":
        guard let s = sim else { fail(id, "no_sim", "call reset first"); return }
        let v = s.membrane()
        var out: [String: Any] = ["sim_ms": s.simMs]
        if let ns = d["neurons"] as? [NSNumber] {
            let ids = ns.map { $0.intValue }
            guard ids.allSatisfy({ $0 >= 0 && $0 < s.n }), ids.count <= 4096 else {
                fail(id, "bad_neuron_id", "ids must be valid, at most 4096"); return
            }
            out["values"] = ids.map { Double(v[$0]) }
        } else {
            fail(id, "bad_target", "membrane requires 'neurons' (max 4096)"); return
        }
        ok(id, out)

    case "shutdown":
        ok(id, ["bye": true])
        exit(0)

    default:
        fail(id, "unknown_cmd", "unknown command '\(cmd)'")
    }
}

func runLabServer() {
    // options: --shuffle N (control connectome), --seed N handled by SIM_SEED
    let a = CommandLine.arguments
    if let i = a.firstIndex(of: "--shuffle"), i + 1 < a.count {
        let t = a[i + 1]
        let hex = t.hasPrefix("0x") || t.hasPrefix("0X")
        shuffled = true
        shuffleSeed = UInt32(hex ? String(t.dropFirst(2)) : t, radix: hex ? 16 : 10) ?? 0
    }
    guard let c = loadConnectome() else {
        fputs("lab-server: no data/ — connectome load failed\n", stderr)
        exit(1)
    }
    if shuffled {
        guard let sc = shuffleConnectomeWeights(c, seed: shuffleSeed) else {
            fputs("lab-server: weight shuffle failed\n", stderr); exit(1)
        }
        connectome = sc
    } else {
        connectome = c
    }
    activeConnectome = connectome
    guard makeSim(seed: SIM_SEED) else {
        fputs("lab-server: MetalSim init failed\n", stderr); exit(1)
    }

    // Line-buffered stdout so the Python peer never stalls waiting for a flush.
    setvbuf(stdout, nil, _IOLBF, 16_384)

    if let s = sim {
        respond(["ok": true, "event": "ready", "version": LAB_VERSION,
                 "n": s.n, "e": connectome?.e ?? 0, "seed": simSeed,
                 "shuffled": shuffled, "shuffle_seed": shuffleSeed,
                 "device": s.deviceName, "groups": groupSizes(s)])
    }
    while let line = readLine() {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty { continue }
        handle(trimmed)
    }
    exit(0)
}
