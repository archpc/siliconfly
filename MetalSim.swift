// MetalSim.swift — the whole FlyWire v783 brain (139,255 neurons, 15,091,983
// signed edges) as a 1 kHz leaky-integrate-and-fire simulation running in two
// Metal compute kernels (LIF.metal). Public surface is the same as the
// 668-neuron CPU sim it replaces, so the body, the signal builder and the brain
// window are unchanged apart from the type name.
//
// Threading: step() is synchronous (encode a batch, commit, wait) and is called
// from the SceneKit render thread exactly like the CPU sim was. stimulate() is
// the only method safe to call from another thread.

import Foundation
import Metal
import simd

// MARK: - Tunables

/// Every knob the tuning phase needs. Changing any of these requires no shader
/// edit: they are either uploaded per-neuron or passed per-step. The measurements
/// behind the values (rate sweeps, rejected alternatives) are in notes/06-tuning-2.md.
struct SimParams {
    // membrane
    var decay: Float = 0.9512          // exp(-1/20): 20 ms tau at 1 ms steps
    var threshold: Float = 1.0
    var refractoryMs: Int = 2
    var floorV: Float = -2             // hyperpolarization clamp, uploaded in StepParams
    // synapses
    var weightScale: Float = 0.0032
    var modScale: Float = 0.5          // DA/SER/OCT are neuromodulatory, not fast drive
    // The GF's two electrical paths are stated relative to `weightScale` because the
    // escape race is calibrated on their ABSOLUTE strength: at 0.0032 a boost of 0.5
    // leaves LC->GF at 0.0016 (one synchronous LC volley = +1.6 threshold units).
    var gapJunctionBoost: Float = 0.5  // LC4/LPLC2 -> GF
    var sensGFBoost: Float = 1.5       // sensory (wind/tap) -> GF
    // Gain on the giant fiber's ORDINARY (chemical) in-synapses — its third and last
    // independent input route. It collects ~3,800 of them, so at one shared
    // `weightScale` it self-fires long before the rest of the brain is network-driven.
    var gfInputScale: Float = 0.12
    var inhDelayMs: Int = 4
    // Noise. The mean drive `pNoise * noiseKick` sets the resting membrane offset; how
    // it is DELIVERED decides how much the connectome matters. One kick is 25% of
    // threshold: small enough that the wiring competes with it, large enough to survive
    // the siesta (`activityScale` 0.84 costs ~0.144 of resting v).
    var pNoise: Float = 0.0050
    var noiseKick: Float = 0.25
    var burstFactor: Float = 6         // noise multiplier inside an arousal burst
    var burstMs: Int = 400
    var burstGapMs: ClosedRange<Int> = 15_000...40_000
    // sensory gains
    var loomGain: Float = 0.32
    var ascendGain: Float = 0.09
    var airPuffGain: Float = 0.05
    // outputs
    var rateAlpha: Float = 1.0 / 120.0
    // Resting drive, per FlyWire super class. A neuron rests at
    // `baseline * 20.49 + pNoise * noiseKick * 20.49` (= `baseline * 20.49 + 0.026`
    // here) against threshold 1, so with `noiseKick` at 0.25 the landmarks are
    //   >= 0.0354  one kick fires it        (rate pins at pNoise = 5 Hz)
    //   >= 0.0424  still one-kick after the siesta scales everything by 0.84
    //   >= 0.0475  self-fires on the mean noise drive alone
    //   >= 0.0488  self-fires with no noise at all
    // Every range straddles 0.0354 (the neurons below it fire only when the wiring
    // pushes them over — that is where the network contribution comes from) and every
    // top clears 0.0424 (that slice keeps the brain, and with it DNp09's drive, alive
    // through the siesta). Optic is 56% of the brain, so its range sets the population
    // rate; sensory/sensory_ascending are receptors and rest at 0.
    var baselineByClass: [String: ClosedRange<Float>] = [
        "optic": 0.021...0.043,
        "central": 0.023...0.047,
        "visual_projection": 0.021...0.043,
        "visual_centrifugal": 0.019...0.043,
        "descending": 0.019...0.043,
        "ascending": 0.019...0.043,
        "motor": 0.019...0.043,
        "endocrine": 0.019...0.043,
        "sensory": 0.000...0.004,
        "sensory_ascending": 0.000...0.006,
    ]
    var baselineFallback: ClosedRange<Float> = 0.019...0.043
    // per-role overrides
    // LC4 / LPLC2: near-silent at rest, and heterogeneous so a SUSTAINED loom
    // desynchronizes them (an abrupt loom step still drives one synchronous onset
    // volley into the giant fiber).
    var baselineLoom: ClosedRange<Float> = 0.008...0.028
    // Command DNs rest far below the one-kick point, so noise alone never fires them
    // and every spike they emit is the connectome's (0.00 Hz intrinsic, 3-68 Hz wired).
    var baselineCommand: Float = 0.022   // DNa01/02, MDN, DNg11, escape-wing DNs
    // DNp09 rests slightly higher: still below the one-kick point (0.50 Hz intrinsic vs
    // 15.5 Hz wired), close enough to keep drive when the siesta compresses the network.
    var baselineFwd: Float = 0.032       // DNp09
    var baselineGF: Float = 0.002        // DNp01: silent unless synaptically driven
}

// Group ids used by the kernel's per-step spike histogram (index 0 = ungrouped).
private enum Group {
    static let loom: UInt8 = 1, gf: UInt8 = 2, dnaL: UInt8 = 3, dnaR: UInt8 = 4
    static let mdn: UInt8 = 5, fwd: UInt8 = 6, groom: UInt8 = 7, escw: UInt8 = 8
    static let slots = 16   // padded, one histogram per batch slot
}

/// PCG-style 32-bit hash (Jarzynski & Olano 2020) — the exact function in
/// LIF.metal. Everything random in the sim (per-neuron baselines, gait phase
/// offsets, the per-step noise draw, the arousal-burst schedule) comes from it,
/// so a CPU reference reproduces the sim bit-for-bit from the seed alone.
@inline(__always) func pcgHash(_ v: UInt32) -> UInt32 {
    let state = v &* 747_796_405 &+ 2_891_336_453
    let word = ((state >> ((state >> 28) &+ 4)) ^ state) &* 277_803_737
    return (word >> 22) ^ word
}
/// The [0,1) draw the kernel makes: the top 24 bits of the hash.
@inline(__always) func pcgUnit(_ h: UInt32) -> Float { Float(h >> 8) * 5.9604645e-8 }
/// Stream `salt`, element `i`: `pcg(pcg(seed &+ salt) &+ i)`.
@inline(__always) func pcgDraw(_ seed: UInt32, _ salt: UInt32, _ i: UInt32) -> Float {
    pcgUnit(pcgHash(pcgHash(seed &+ salt) &+ i))
}

// Salts keep the streams independent. The noise salt is `stepIndex * 2654435761`
// (see LIF.metal); these are the CPU-side streams.
private let saltBaseline: UInt32 = 0x51ED_2701
private let saltPhase: UInt32 = 0x2F1B_3C4D
private let saltBurst: UInt32 = 0x7A3B_9F11

// Must match `struct StepParams` in LIF.metal field for field.
private struct StepParams {
    var n: UInt32 = 0, slot: UInt32 = 0, stepIndex: UInt32 = 0
    var curSlot: UInt32 = 0, inhSlot: UInt32 = 0, seed: UInt32 = 0, refractory: UInt32 = 0
    var invFx: Float = 0, decay: Float = 0, threshold: Float = 0
    var pNoise: Float = 0, noiseKick: Float = 0, activityScale: Float = 0
    var loomDriveL: Float = 0, loomDriveR: Float = 0
    var ascendDrive: Float = 0, gaitPh: Float = 0, airPuffDrive: Float = 0
    var floorV: Float = 0
    var tasteP: Float = 0       // per-step Poisson taste probability (0 = off)
    var tasteKick: Float = 0
    var tasteSeed: UInt32 = 0
}

// MARK: - Shared GPU resources

/// Device, compiled pipelines and the immutable CSR buffers. Built once per
/// connectome and shared by every `MetalSim` on it (--behaviortest makes one sim
/// per scenario; only the 121 MB of edge data is expensive, and it is read-only).
final class MetalShared {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let update: MTLComputePipelineState
    let propagate: MTLComputePipelineState
    let rowStart: MTLBuffer
    let colIdx: MTLBuffer
    let weightFx: MTLBuffer
    /// Weights are Int32 fixed point so synaptic accumulation can use integer
    /// atomics: order-independent, exactly reproducible, no float-atomic support
    /// needed. The scale is the largest power of two in [2^16, 2^20] for which
    /// `max|w| * scale * 4096` still fits in Int32.
    let fxScale: Float
    let compileMs: Double
    let buildMs: Double
    /// Every knob the load-time weight transform reads. A sim asking for different
    /// ones must not be handed these weights.
    let knobs: Knobs

    struct Knobs: Equatable {
        let weightScale, modScale, gapJunctionBoost, sensGFBoost, gfInputScale: Float
        init(_ p: SimParams) {
            weightScale = p.weightScale; modScale = p.modScale
            gapJunctionBoost = p.gapJunctionBoost; sensGFBoost = p.sensGFBoost
            gfInputScale = p.gfInputScale
        }
    }

    struct Err: Error, CustomStringConvertible {
        let description: String
        init(_ d: String) { description = d }
    }

    init(_ c: Connectome, _ p: SimParams) throws {
        guard let dev = MTLCreateSystemDefaultDevice() else { throw Err("no Metal device") }
        guard let q = dev.makeCommandQueue() else { throw Err("no Metal command queue") }
        device = dev
        queue = q
        knobs = Knobs(p)

        // ---- compile LIF.metal at runtime (no metallib build step) -------------
        let tCompile = DispatchTime.now()
        guard let src = findResource("LIF.metal"),
              let text = try? String(contentsOf: src, encoding: .utf8) else {
            throw Err("LIF.metal not found next to the executable or in the working directory")
        }
        let opts = MTLCompileOptions()
        opts.mathMode = .safe       // no reassociation (fma contraction still applies)
        let lib = try dev.makeLibrary(source: text, options: opts)
        guard let fUpdate = lib.makeFunction(name: "lif_update"),
              let fProp = lib.makeFunction(name: "lif_propagate") else {
            throw Err("LIF.metal is missing lif_update / lif_propagate")
        }
        update = try dev.makeComputePipelineState(function: fUpdate)
        propagate = try dev.makeComputePipelineState(function: fProp)
        compileMs = MetalShared.ms(since: tCompile)

        // ---- CSR buffers -------------------------------------------------------
        let tBuild = DispatchTime.now()
        let n = c.n, e = c.e
        guard let rs = dev.makeBuffer(bytes: c.rowStart, length: (n + 1) * 4,
                                      options: .storageModeShared),
              let ci = c.colIdxData.withUnsafeBytes({ raw in
                  dev.makeBuffer(bytes: raw.baseAddress!, length: e * 4, options: .storageModeShared)
              }),
              let wf = dev.makeBuffer(length: e * 4, options: .storageModeShared) else {
            throw Err("could not allocate the CSR buffers (\(e) edges)")
        }
        rowStart = rs; colIdx = ci; weightFx = wf

        // Load-time weight transform:
        //   w = synapses * weightScale
        //       * (pre nt in {DA,SER,OCT} ? modScale : 1)          <- row constant `g`
        //       * (post role == gf ? (pre lc4/lplc2 ? gapJunctionBoost
        //                                : pre sens ? sensGFBoost : gfInputScale) : 1)
        // Pass 1 measures the exact maximum so the fixed-point scale carries a
        // proven overflow bound; pass 2 quantizes with the same per-edge factors.
        var maxAbs: Float = 0
        var fx: Float = 1_048_576   // 2^20
        c.colIdxData.withUnsafeBytes { cb in
            c.weightData.withUnsafeBytes { wb in
                let col = cb.baseAddress!.assumingMemoryBound(to: UInt32.self)
                let w16 = wb.baseAddress!.assumingMemoryBound(to: Int16.self)
                let out = wf.contents().bindMemory(to: Int32.self, capacity: e)

                /// The GF-path gain on edge `k` out of neuron `i`: the giant fiber's
                /// three input routes are three independent gains, every other post
                /// takes the edge as it is. The row-constant part stays hoisted (the
                /// factor order is what the CPU reference reproduces bit for bit).
                @inline(__always) func gfGain(_ i: Int, _ k: Int) -> Float {
                    guard c.role[Int(col[k])] == Role.gf else { return 1 }
                    switch c.role[i] {
                    case Role.lc4, Role.lplc2: return p.gapJunctionBoost
                    case Role.sens:            return p.sensGFBoost
                    default:                   return p.gfInputScale
                    }
                }

                for i in 0..<n {
                    let g = p.weightScale * (c.isModulatory[Int(c.nt[i])] ? p.modScale : 1)
                    for k in Int(c.rowStart[i])..<Int(c.rowStart[i + 1]) {
                        maxAbs = max(maxAbs, Float(Int32(w16[k]).magnitude) * g * gfGain(i, k))
                    }
                }
                while fx > 65_536 && Double(maxAbs) * Double(fx) * 4096 > Double(Int32.max) {   // >= 2^16
                    fx /= 2
                }
                for i in 0..<n {
                    let g = p.weightScale * (c.isModulatory[Int(c.nt[i])] ? p.modScale : 1) * fx
                    for k in Int(c.rowStart[i])..<Int(c.rowStart[i + 1]) {
                        out[k] = Int32((Float(w16[k]) * g * gfGain(i, k)).rounded())
                    }
                }
            }
        }
        fxScale = fx
        guard Double(maxAbs) * Double(fx) * 4096 <= Double(Int32.max) else {
            throw Err(String(format: "fixed-point overflow: max|w| %.4f x scale %.0f x 4096 exceeds Int32",
                             maxAbs, fx))
        }
        buildMs = MetalShared.ms(since: tBuild)

        fputs(String(format: "metal-sim: %@ N=%d E=%d load %.0f ms compile %.0f ms\n",
                     dev.name, n, e, c.loadMs + buildMs, compileMs), stderr)
    }

    static func ms(since t: DispatchTime) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - t.uptimeNanoseconds) / 1e6
    }
}

// MARK: - MetalSim

final class MetalSim {
    // ---- topology (shared with the connectome, no copies) --------------------
    let n: Int
    let roles: [String]
    let types: [String]
    let positions: [SIMD3<Float>]
    private let roleId: [UInt8]
    private let classOf: [UInt8]                  // super class per neuron
    private let classRange: [ClosedRange<Float>]  // resting-drive range per super class

    // ---- groups --------------------------------------------------------------
    private(set) var loomLeft: [Int] = []
    private(set) var loomRight: [Int] = []
    private(set) var gf: [Int] = []
    private(set) var dnaL: [Int] = []      // DNa01 + DNa02, left
    private(set) var dnaR: [Int] = []      // DNa01 + DNa02, right
    private(set) var mdn: [Int] = []
    private(set) var fwd: [Int] = []       // DNp09
    private(set) var groom: [Int] = []     // DNg11
    private(set) var escw: [Int] = []      // DNp02/04/11 escape-maneuver (wing) DNs
    private(set) var ascend: [Int] = []    // ascending partners (leg proprioception)
    private(set) var sens: [Int] = []      // sensory partners (air-puff pathway)

    // ---- inputs (0..1), set each frame by the coordinator ---------------------
    var loomL: Float = 0
    var loomR: Float = 0
    var gaitDrive: Float = 0      // body walking intensity -> ascending neurons
    var gaitPhase: Float = 0      // body gait phase 0..1 -> rhythmic proprioception
    var airPuff: Float = 0        // fast cursor motion near the fly -> sensory neurons
    var sugarRate: Float = 0      // Poisson taste rate in Hz at the sugar GRNs (0 = off)
    var activityScale: Float = 1  // circadian / sleep neuromodulation of baseline+noise
    var sensoryGate: Float = 1    // sleep gates sensory input

    // ---- outputs -------------------------------------------------------------
    private(set) var rateLoom: Float = 0   // Hz per LC neuron (EMA)
    private(set) var rateDNaL: Float = 0
    private(set) var rateDNaR: Float = 0
    private(set) var rateMDN: Float = 0
    private(set) var rateFwd: Float = 0
    private(set) var rateGroom: Float = 0
    private(set) var rateEscW: Float = 0
    private(set) var ratePop: Float = 0    // whole-population Hz per neuron
    private var gfLatch = false
    private(set) var simMs: Int = 0
    private(set) var totalSpikes: Int = 0

    let spikeBus: SpikeBus?

    // ---- diagnostics ---------------------------------------------------------
    var deviceName: String { shared.device.name }
    let loadMs: Double            // connectome parse + CSR upload
    var compileMs: Double { shared.compileMs }
    var fixedPointScale: Float { shared.fxScale }
    /// The quantized edge weights this sim is stepping, one Int32 per CSR edge in
    /// `Connectome` order; divide by `fixedPointScale` for threshold units. The
    /// load-time transform (weightScale, modulator scaling, the giant fiber's three
    /// input gains) is already baked in — a reader must never re-derive it.
    var edgeWeightsFx: UnsafePointer<Int32> {
        let b = shared.weightFx
        return UnsafePointer(b.contents().bindMemory(to: Int32.self, capacity: b.length / 4))
    }
    private(set) var avgStepMicros: Double = 0   // EMA of wall microseconds per step
    var debugParams: SimParams { p }
    /// Sim-time interval between throughput log lines; 0 disables (tests do that).
    var perfLogIntervalMs = 30_000

    // ---- private state -------------------------------------------------------
    private let shared: MetalShared
    private var p: SimParams
    private var seed: UInt32
    private let maxBatch = 64
    private let ringSlots: Int

    private let vBuf, refrBuf, baselineBuf, inputKindBuf, phaseBuf, groupOfBuf: MTLBuffer
    private let extInputBuf, excBuf, inhBuf, spikeListBuf: MTLBuffer
    private let spikeCountBuf, groupCountBuf, sampleBuf: MTLBuffer
    private let extPtr: UnsafeMutablePointer<Float>
    private let spikeCountPtr, groupCountPtr, samplePtr, spikeListPtr: UnsafeMutablePointer<UInt32>
    private let gridN: MTLSize, tgUpdate: MTLSize, propGrid: MTLSize, tgProp: MTLSize

    private var burstUntil = 0
    private var burstNext = 12_000
    private var burstCounter: UInt32 = 0
    private var lastSpikeCount = 0
    private var lastGroupCounts = [UInt32](repeating: 0, count: Group.slots)

    private var logSteps = 0, logSpikes = 0, logWallMicros = 0.0, logNextMs = 30_000

    // EMA denominators
    private let nLoom, nDNaL, nDNaR, nMDN, nFwd, nGroom, nEscW, nPop: Float

    // "optogenetic" stimulation from brain-window clicks (any thread)
    private struct Stim { let idx: [Int]; let strength: Float; let durationMs: Int; var untilMs = 0 }
    private var pendingStims: [Stim] = []
    private var activeStims: [Stim] = []
    private var extDirty = Set<Int>()
    private let stimLock = NSLock()

    init?(connectome c: Connectome, spikeBus: SpikeBus?, seed: UInt32 = 0x5EED_1F1F,
          params: SimParams = SimParams(), tasteIndices: [Int]? = nil) {
        assert(MemoryLayout<StepParams>.stride == 88, "StepParams must match LIF.metal")
        self.spikeBus = spikeBus
        self.p = params
        self.seed = seed
        self.loadMs = c.loadMs
        self.ringSlots = params.inhDelayMs + 1
        n = c.n
        roles = c.roleName
        types = c.typeName
        positions = c.positions
        roleId = c.role

        // Shared, immutable GPU side. Rebuilt only if the weight knobs changed.
        let want = MetalShared.Knobs(params)
        var sh = c.gpu.shared
        if sh == nil || sh!.knobs != want {
            do { sh = try MetalShared(c, params) }
            catch { fputs("metal-sim: \(error)\n", stderr); return nil }
            c.gpu.shared = sh
        }
        guard let shared = sh else { return nil }
        self.shared = shared
        let dev = shared.device

        // ---- per-neuron state: group membership and input routing -------------
        // The seeded values (resting drive, gait phase) go straight into their
        // buffers in applySeed() below, which setSeed() reuses to reseed a live sim.
        var inputKind = [UInt8](repeating: 0, count: n)
        var groupOf = [UInt8](repeating: 0, count: n)
        classRange = c.superClassNames.map { params.baselineByClass[$0] ?? params.baselineFallback }
        classOf = c.superClass
        for i in 0..<n {
            let left = c.side[i] == 1
            switch c.role[i] {
            case Role.lc4, Role.lplc2:
                groupOf[i] = Group.loom
                inputKind[i] = left ? 1 : 2
                if left { loomLeft.append(i) } else { loomRight.append(i) }
            case Role.gf:
                groupOf[i] = Group.gf; gf.append(i)
            case Role.dna01, Role.dna02:
                groupOf[i] = left ? Group.dnaL : Group.dnaR
                if left { dnaL.append(i) } else { dnaR.append(i) }
            case Role.mdn:   groupOf[i] = Group.mdn; mdn.append(i)
            case Role.dnp09: groupOf[i] = Group.fwd; fwd.append(i)
            case Role.dng11: groupOf[i] = Group.groom; groom.append(i)
            case Role.escw:  groupOf[i] = Group.escw; escw.append(i)
            case Role.ascend: inputKind[i] = 3; ascend.append(i)
            case Role.sens:   inputKind[i] = 4; sens.append(i)
            default: break
            }
        }
        // Poisson taste targets (Operation Banana Peel / Second Helping): the
        // identified sugar-GRN indices get inputKind 5. Zero when not provided,
        // which leaves every existing mode untouched.
        if let ti = tasteIndices {
            for i in ti where i >= 0 && i < n { inputKind[i] = 5 }
        }

        func buf<T>(_ v: [T]) -> MTLBuffer? {
            v.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: $0.count,
                                               options: .storageModeShared) }
        }
        guard let vB = dev.makeBuffer(length: n * 4, options: .storageModeShared),
              let rB = dev.makeBuffer(length: n, options: .storageModeShared),
              let bB = dev.makeBuffer(length: n * 4, options: .storageModeShared),
              let phB = dev.makeBuffer(length: n * 4, options: .storageModeShared),
              let kB = buf(inputKind), let gB = buf(groupOf),
              let eiB = dev.makeBuffer(length: n * 4, options: .storageModeShared),
              let exB = dev.makeBuffer(length: n * 4, options: .storageModeShared),
              let inB = dev.makeBuffer(length: n * 4 * ringSlots, options: .storageModeShared),
              let slB = dev.makeBuffer(length: n * 4, options: .storageModeShared),
              let scB = dev.makeBuffer(length: maxBatch * 4, options: .storageModeShared),
              let gcB = dev.makeBuffer(length: maxBatch * Group.slots * 4, options: .storageModeShared),
              let smB = dev.makeBuffer(length: maxBatch * 32 * 4, options: .storageModeShared)
        else { fputs("metal-sim: could not allocate the per-neuron buffers\n", stderr); return nil }
        vBuf = vB; refrBuf = rB; baselineBuf = bB; inputKindBuf = kB; phaseBuf = phB
        groupOfBuf = gB; extInputBuf = eiB; excBuf = exB; inhBuf = inB; spikeListBuf = slB
        spikeCountBuf = scB; groupCountBuf = gcB; sampleBuf = smB
        extPtr = eiB.contents().bindMemory(to: Float.self, capacity: n)
        spikeCountPtr = scB.contents().bindMemory(to: UInt32.self, capacity: maxBatch)
        groupCountPtr = gcB.contents().bindMemory(to: UInt32.self, capacity: maxBatch * Group.slots)
        samplePtr = smB.contents().bindMemory(to: UInt32.self, capacity: maxBatch * 32)
        spikeListPtr = slB.contents().bindMemory(to: UInt32.self, capacity: n)
        for b in [vB, rB, eiB, exB, inB] { memset(b.contents(), 0, b.length) }

        let tgw = min(256, shared.update.maxTotalThreadsPerThreadgroup)
        gridN = MTLSize(width: n, height: 1, depth: 1)
        tgUpdate = MTLSize(width: tgw, height: 1, depth: 1)
        propGrid = MTLSize(width: 1024, height: 1, depth: 1)
        tgProp = MTLSize(width: 32, height: 1, depth: 1)

        nLoom = Float(max(1, loomLeft.count + loomRight.count))
        nDNaL = Float(max(1, dnaL.count)); nDNaR = Float(max(1, dnaR.count))
        nMDN = Float(max(1, mdn.count)); nFwd = Float(max(1, fwd.count))
        nGroom = Float(max(1, groom.count)); nEscW = Float(max(1, escw.count))
        nPop = Float(max(1, n))
        logNextMs = perfLogIntervalMs
        applySeed(seed)
    }

    /// Draws everything the seed decides — per-neuron resting drive and gait phase
    /// offsets, straight into their GPU buffers — and restarts the arousal-burst
    /// schedule. Group membership and input routing are seed-independent and stay
    /// in init. Roles with a fixed resting point are rewritten with the same
    /// constant, so reseeding a live sim only moves what the seed owns.
    private func applySeed(_ s: UInt32) {
        seed = s
        burstCounter = 0
        burstUntil = 0
        burstNext = simMs + 12_000
        let bp = baselineBuf.contents().bindMemory(to: Float.self, capacity: n)
        let ph = phaseBuf.contents().bindMemory(to: Float.self, capacity: n)
        func draw(_ r: ClosedRange<Float>, _ i: Int) -> Float {
            r.lowerBound + (r.upperBound - r.lowerBound) * pcgDraw(s, saltBaseline, UInt32(i))
        }
        for i in 0..<n {
            switch roleId[i] {
            case Role.lc4, Role.lplc2: bp[i] = draw(p.baselineLoom, i)
            case Role.gf:              bp[i] = p.baselineGF
            case Role.dnp09:           bp[i] = p.baselineFwd
            case Role.dna01, Role.dna02, Role.mdn, Role.dng11, Role.escw:
                bp[i] = p.baselineCommand
            default:
                // ascending / sensory partners included: they are ordinary neurons
                // that happen to be the circuit's input targets.
                bp[i] = draw(classRange[Int(classOf[i])], i)
            }
            ph[i] = roleId[i] == Role.ascend ? 2 * Float.pi * pcgDraw(s, saltPhase, UInt32(i)) : 0
        }
    }

    // MARK: - Stimulation

    func stimulate(_ indices: [Int], strength: Float, durationMs: Int) {
        guard !indices.isEmpty else { return }
        stimLock.lock()
        pendingStims.append(Stim(idx: indices, strength: strength, durationMs: durationMs))
        if pendingStims.count > 8 { pendingStims.removeFirst() }
        stimLock.unlock()
    }

    func consumeGF() -> Bool { let s = gfLatch; gfLatch = false; return s }

    /// Recomputes `extInput` from the active stim set. Only the touched indices
    /// are rewritten, and they are rebuilt (not incremented/decremented) so the
    /// value is exact no matter how stims overlap.
    private func rebuildExtInput() {
        for i in extDirty { extPtr[i] = 0 }
        extDirty.removeAll(keepingCapacity: true)
        for s in activeStims {
            for i in s.idx where i >= 0 && i < n {
                extPtr[i] += s.strength
                extDirty.insert(i)
            }
        }
    }

    /// A stim contributes while `simMs < untilMs`, so it is finished once the next
    /// step would reach `untilMs`. Dropping it here also bounds the sub-batch.
    private func dropFinishedStims() {
        let before = activeStims.count
        activeStims.removeAll { simMs + 1 >= $0.untilMs }
        if activeStims.count != before { rebuildExtInput() }
    }

    // MARK: - Step

    func step(_ ms: Int) {
        guard ms > 0 else { return }
        stimLock.lock()
        let fresh = pendingStims
        pendingStims.removeAll()
        stimLock.unlock()
        if !fresh.isEmpty {
            for var s in fresh { s.untilMs = simMs + s.durationMs; activeStims.append(s) }
            rebuildExtInput()
        }
        dropFinishedStims()

        var remaining = ms
        while remaining > 0 {
            // a sub-batch never crosses a stim expiry, so extInput is exact per step
            let room = activeStims.map(\.untilMs).min().map { $0 - simMs - 1 } ?? Int.max
            let k = min(remaining, maxBatch, max(1, room))
            runBatch(k)
            remaining -= k
            dropFinishedStims()
        }

        if perfLogIntervalMs > 0 && simMs >= logNextMs {
            fputs(String(format: "metal-sim: %d steps in %.0f ms (%.0f µs/step, %.0f spikes/step avg)\n",
                         logSteps, logWallMicros / 1000, logWallMicros / Double(max(1, logSteps)),
                         Double(logSpikes) / Double(max(1, logSteps))), stderr)
            logSteps = 0; logSpikes = 0; logWallMicros = 0
            logNextMs = simMs + perfLogIntervalMs
        }
    }

    private func runBatch(_ k: Int) {
        let t0 = DispatchTime.now()

        // Per-step parameters. Everything here is CPU state (input levels are held
        // constant across a step() call, the arousal-burst schedule is seeded), so
        // the whole batch can be encoded before any of it runs.
        var params = [StepParams](); params.reserveCapacity(k)
        for s in 0..<k {
            simMs += 1
            if simMs >= burstNext {
                burstUntil = simMs + p.burstMs
                burstCounter &+= 1
                let span = p.burstGapMs.upperBound - p.burstGapMs.lowerBound + 1
                burstNext = simMs + p.burstGapMs.lowerBound
                    + Int(pcgDraw(seed, saltBurst, burstCounter) * Float(span))
            }
            let noise = (simMs < burstUntil ? p.pNoise * p.burstFactor : p.pNoise) * activityScale
            params.append(StepParams(
                n: UInt32(n), slot: UInt32(s), stepIndex: UInt32(truncatingIfNeeded: simMs),
                curSlot: UInt32((simMs - 1) % ringSlots),
                inhSlot: UInt32((simMs - 1 + p.inhDelayMs) % ringSlots),
                seed: seed, refractory: UInt32(p.refractoryMs),
                invFx: 1 / shared.fxScale, decay: p.decay, threshold: p.threshold,
                pNoise: noise, noiseKick: p.noiseKick, activityScale: activityScale,
                loomDriveL: loomL > 0.001 ? loomL * p.loomGain * sensoryGate : 0,
                loomDriveR: loomR > 0.001 ? loomR * p.loomGain * sensoryGate : 0,
                ascendDrive: gaitDrive > 0.001 ? gaitDrive * p.ascendGain : 0,
                gaitPh: gaitPhase * 2 * Float.pi,
                airPuffDrive: airPuff > 0.001 ? airPuff * p.airPuffGain * sensoryGate : 0,
                floorV: p.floorV,
                tasteP: sugarRate > 0.001 ? sugarRate * 0.001 : 0,
                tasteKick: p.noiseKick,
                tasteSeed: seed &+ 0x7A57_1E7E))
        }

        memset(spikeCountBuf.contents(), 0, k * 4)
        memset(groupCountBuf.contents(), 0, k * Group.slots * 4)
        memset(sampleBuf.contents(), 0xFF, k * 32 * 4)   // 0xFFFFFFFF = empty slot

        guard let cb = shared.queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }
        // Buffer bindings are encoder state and the two kernels use disjoint
        // indices, so everything but StepParams is bound once per batch.
        for (i, b) in [vBuf, refrBuf, baselineBuf, inputKindBuf, phaseBuf, groupOfBuf,
                       extInputBuf, excBuf, inhBuf, spikeListBuf, spikeCountBuf,
                       groupCountBuf, sampleBuf, shared.rowStart, shared.colIdx,
                       shared.weightFx].enumerated() {
            enc.setBuffer(b, offset: 0, index: i)
        }
        for s in 0..<k {
            var sp = params[s]
            enc.setComputePipelineState(shared.update)
            enc.setBytes(&sp, length: MemoryLayout<StepParams>.stride, index: 16)
            enc.dispatchThreads(gridN, threadsPerThreadgroup: tgUpdate)
            enc.setComputePipelineState(shared.propagate)
            enc.setBytes(&sp, length: MemoryLayout<StepParams>.stride, index: 16)
            enc.dispatchThreadgroups(propGrid, threadsPerThreadgroup: tgProp)
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        // ---- fold the per-step histograms into the rate EMAs ------------------
        let a = p.rateAlpha
        var bus: [(Int, Bool)] = []
        for s in 0..<k {
            let total = Int(spikeCountPtr[s])
            totalSpikes += total
            logSpikes += total
            let g = groupCountPtr + s * Group.slots
            if g[Int(Group.gf)] > 0 { gfLatch = true }
            rateLoom += (Float(g[Int(Group.loom)]) * 1000 / nLoom - rateLoom) * a
            rateDNaL += (Float(g[Int(Group.dnaL)]) * 1000 / nDNaL - rateDNaL) * a
            rateDNaR += (Float(g[Int(Group.dnaR)]) * 1000 / nDNaR - rateDNaR) * a
            rateMDN  += (Float(g[Int(Group.mdn)])  * 1000 / nMDN  - rateMDN)  * a
            rateFwd  += (Float(g[Int(Group.fwd)])  * 1000 / nFwd  - rateFwd)  * a
            rateGroom += (Float(g[Int(Group.groom)]) * 1000 / nGroom - rateGroom) * a
            rateEscW += (Float(g[Int(Group.escw)]) * 1000 / nEscW - rateEscW) * a
            ratePop  += (Float(total) * 1000 / nPop - ratePop) * a

            if spikeBus != nil {
                var pushed = 0, t = 0
                while t < 32 && pushed < 12 {
                    let idx = samplePtr[s * 32 + t]
                    t += 1
                    if idx == 0xFFFF_FFFF { continue }
                    bus.append((Int(idx), roleId[Int(idx)] == Role.gf))
                    pushed += 1
                }
            }
        }
        lastSpikeCount = Int(spikeCountPtr[k - 1])
        for i in 0..<Group.slots { lastGroupCounts[i] = groupCountPtr[(k - 1) * Group.slots + i] }
        spikeBus?.push(bus)

        let micros = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1000
        logWallMicros += micros
        logSteps += k
        let per = micros / Double(k)
        avgStepMicros = avgStepMicros == 0 ? per : avgStepMicros + (per - avgStepMicros) * 0.1
    }

    // MARK: - Test hooks
    //
    // Cheap, synchronous reads of the shared (CPU-visible) GPU buffers. Only
    // valid between step() calls, which is all a single-threaded cross-check
    // needs. See notes/03-metal-sim.md for the CPU-reference recipe.

    /// Membrane potentials, one per neuron.
    func membrane() -> [Float] {
        let ptr = vBuf.contents().bindMemory(to: Float.self, capacity: n)
        return [Float](UnsafeBufferPointer(start: ptr, count: n))
    }

    /// Remaining refractory steps, one per neuron (0, 1 or 2).
    func debugRefr() -> [UInt8] {
        let ptr = refrBuf.contents().bindMemory(to: UInt8.self, capacity: n)
        return [UInt8](UnsafeBufferPointer(start: ptr, count: n))
    }

    /// Every neuron that spiked in the most recent step, in GPU-race order.
    func lastStepSpikes() -> [Int32] {
        (0..<lastSpikeCount).map { Int32(bitPattern: spikeListPtr[$0]) }
    }

    /// The most recent step's spike histogram, indexed by the kernel's group ids
    /// (1 loom, 2 gf, 3 dnaL, 4 dnaR, 5 mdn, 6 fwd, 7 groom, 8 escw).
    func lastStepGroupCounts() -> [UInt32] { lastGroupCounts }

    /// Reseeds every random stream: per-neuron baselines, gait phase offsets, the
    /// membrane-noise draw and the arousal-burst schedule. Membrane state is left
    /// alone; call it right after init for a deterministic run.
    func setSeed(_ s: UInt32) { applySeed(s) }

    /// Overwrites the per-neuron resting drive (tuning / cross-check hook).
    func setBaseline(_ values: [Float]) {
        guard values.count == n else { return }
        let ptr = baselineBuf.contents().bindMemory(to: Float.self, capacity: n)
        values.withUnsafeBufferPointer { ptr.update(from: $0.baseAddress!, count: n) }
    }

    /// Current per-neuron resting drive, straight from the baseline buffer
    /// (lab-server ablation hook: read-modify-zero without duplicating the
    /// seeded draw logic).
    func baselineValues() -> [Float] {
        let ptr = baselineBuf.contents().bindMemory(to: Float.self, capacity: n)
        return [Float](UnsafeBufferPointer(start: ptr, count: n))
    }
}
