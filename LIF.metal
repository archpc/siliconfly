// LIF.metal — the whole-brain leaky-integrate-and-fire step, two kernels per
// simulated millisecond. Compiled at RUNTIME by MetalSim.swift with
// MTLCompileOptions.mathMode = .safe (no fast-math reassociation). That does NOT
// disable fused multiply-add contraction: the leak below compiles to
// fma(v, decay, baseline*activityScale) — ONE rounding where a CPU's `a*b + c`
// has two. A bit-exact CPU reference must fuse the same way; in Swift that is
// `(baseline * scale).addingProduct(v, decay)`. One ulp here flips a spike
// within ~100 steps (the operating point is chaotic). See notes/04-gpucheck.md.
//
// Per-step order (identical to the CPU sim it replaces):
//   1. apply the excitation delivered by the PREVIOUS step's spikes (clamp floorV)
//   2. refractory ? (refr--, v *= decay) : (v = v*decay + baseline*scale, + noise)
//   3. external drive (loom / gait / air puff / click stim) — refractory or not
//   4. apply the inhibition scheduled for this step by spikes 4 ms ago (clamp floorV)
//   5. threshold: refr <= 0 && v >= 1 -> v = 0, refr = 2, spike
//   6. lif_propagate: this step's spikes -> excAcc (next step) / inhRing (+4 ms)
//
// Synaptic accumulation is Int32 fixed point (see MetalSim.fixedPointScale), so
// atomics are integer and the result is independent of thread order.

#include <metal_stdlib>
using namespace metal;

struct StepParams {
    uint  n;             // neuron count
    uint  slot;          // batch slot 0..S-1 (indexes spikeCount/groupCounts/sample)
    uint  stepIndex;     // 1-based global step (= simMs); the noise RNG stream
    uint  curSlot;       // inhibition ring slot consumed by this step
    uint  inhSlot;       // inhibition ring slot this step's spikes write into
    uint  seed;
    uint  refractory;    // refractory period in steps (2)
    float invFx;         // 1 / fixed-point scale
    float decay;         // exp(-1/tau)
    float threshold;
    float pNoise;        // already multiplied by activityScale and the burst factor
    float noiseKick;
    float activityScale;
    float loomDriveL;    // loomL * loomGain * sensoryGate (0 when gated off)
    float loomDriveR;
    float ascendDrive;   // gaitDrive * 0.09 (0 when gated off)
    float gaitPh;        // gaitPhase * 2pi
    float airPuffDrive;  // airPuff * 0.12 * sensoryGate (0 when gated off)
    float floorV;        // hyperpolarization clamp (SimParams.floorV, -2)
    float tasteP;        // per-step probability of a Poisson taste spike (0 = off)
    float tasteKick;     // charge deposited per Poisson taste spike
    uint  tasteSeed;     // independent PCG stream for the taste draw
};

// PCG-style 32-bit hash (Jarzynski & Olano 2020). Mirrored bit-for-bit in
// MetalSim.pcgHash so a CPU reference can reproduce the noise stream.
static inline uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// One thread per neuron: steps 1-5 above.
kernel void lif_update(
    device float*         v           [[buffer(0)]],
    device uchar*         refr        [[buffer(1)]],
    device const float*   baseline    [[buffer(2)]],
    device const uchar*   inputKind   [[buffer(3)]],
    device const float*   phase       [[buffer(4)]],
    device const uchar*   groupOf     [[buffer(5)]],
    device const float*   extInput    [[buffer(6)]],
    device atomic_int*    excAcc      [[buffer(7)]],
    device atomic_int*    inhRing     [[buffer(8)]],
    device uint*          spikeList   [[buffer(9)]],
    device atomic_uint*   spikeCount  [[buffer(10)]],
    device atomic_uint*   groupCounts [[buffer(11)]],
    device uint*          sampled     [[buffer(12)]],
    constant StepParams&  P           [[buffer(16)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= P.n) { return; }

    float vi = v[gid];

    // 1. excitation delivered by the previous step's spikes
    int ex = atomic_exchange_explicit(&excAcc[gid], 0, memory_order_relaxed);
    if (ex != 0) { vi = max(P.floorV, vi + float(ex) * P.invFx); }

    // one RNG draw per neuron per step: bits 8..31 -> noise, bits 3..7 -> sample slot
    uint h = pcg(pcg(P.seed + P.stepIndex * 2654435761u) + gid);

    // 2. leak; refractory neurons only leak (no baseline, no noise)
    uint r = refr[gid];
    if (r > 0u) {
        refr[gid] = uchar(r - 1u);
        vi *= P.decay;
    } else {
        vi = vi * P.decay + baseline[gid] * P.activityScale;
        if (float(h >> 8u) * 5.9604645e-8f < P.pNoise) { vi += P.noiseKick; }
    }

    // 3. external drive, applied whether or not the neuron is refractory
    uchar kind = inputKind[gid];
    if (kind == 1u) {
        vi += P.loomDriveL;
    } else if (kind == 2u) {
        vi += P.loomDriveR;
    } else if (kind == 3u) {
        vi += P.ascendDrive * (0.5f + 0.5f * precise::sin(P.gaitPh + phase[gid]));
    } else if (kind == 4u) {
        vi += P.airPuffDrive;
    } else if (kind == 5u && P.tasteP > 0.0f) {
        // Poisson taste spike: independent stream, so the noise draw above is
        // untouched and existing (taste-off) modes stay bit-identical.
        uint h2 = pcg(pcg(P.tasteSeed + P.stepIndex * 2654435761u) + gid);
        if (float(h2 >> 8u) * 5.9604645e-8f < P.tasteP) { vi += P.tasteKick; }
    }
    vi += extInput[gid];

    // 4. inhibition scheduled for this step (delivered 4 ms after the spike)
    int inh = atomic_exchange_explicit(&inhRing[P.curSlot * P.n + gid], 0, memory_order_relaxed);
    if (inh != 0) { vi = max(P.floorV, vi + float(inh) * P.invFx); }

    // 5. threshold. r is the pre-decrement value, so "refr <= 0 after step 2"
    //    means r == 0 or r == 1 — a neuron leaving refractoriness may fire.
    bool spike = (r <= 1u) && (vi >= P.threshold);
    if (spike) {
        vi = 0.0f;
        refr[gid] = uchar(P.refractory);
    }
    v[gid] = vi;

    if (spike) {
        uint k = atomic_fetch_add_explicit(&spikeCount[P.slot], 1u, memory_order_relaxed);
        spikeList[k] = gid;
        uchar g = groupOf[gid];
        if (g != 0u) {
            atomic_fetch_add_explicit(&groupCounts[P.slot * 16u + uint(g)], 1u, memory_order_relaxed);
        }
        // benign race: last writer wins, which is exactly the order-independent
        // random sample of spikers the brain window wants.
        sampled[P.slot * 32u + ((h >> 3u) & 31u)] = gid;
    }
}

// One SIMD group (32 lanes) per spiking neuron, grid-striding over the spike
// list; lanes stride over that neuron's CSR edge range.
kernel void lif_propagate(
    device atomic_int*    excAcc      [[buffer(7)]],
    device atomic_int*    inhRing     [[buffer(8)]],
    device const uint*    spikeList   [[buffer(9)]],
    device const uint*    spikeCount  [[buffer(10)]],
    device const uint*    rowStart    [[buffer(13)]],
    device const uint*    colIdx      [[buffer(14)]],
    device const int*     weightFx    [[buffer(15)]],
    constant StepParams&  P           [[buffer(16)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint ntg    [[threadgroups_per_grid]],
    uint lane   [[thread_position_in_threadgroup]],
    uint width  [[threads_per_threadgroup]])
{
    uint count = spikeCount[P.slot];
    uint inhBase = P.inhSlot * P.n;
    for (uint t = tgid; t < count; t += ntg) {
        uint i = spikeList[t];
        uint a = rowStart[i], b = rowStart[i + 1u];
        for (uint k = a + lane; k < b; k += width) {
            int w = weightFx[k];
            uint j = colIdx[k];
            if (w >= 0) {
                atomic_fetch_add_explicit(&excAcc[j], w, memory_order_relaxed);
            } else {
                atomic_fetch_add_explicit(&inhRing[inhBase + j], w, memory_order_relaxed);
            }
        }
    }
}
