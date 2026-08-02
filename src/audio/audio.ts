/**
 * INK TIDE — synthesised audio.
 *
 * No samples, no files, no decodeAudioData. Every sound is built out of
 * oscillators, procedurally generated noise buffers and filters.
 *
 * The mix is four buses into a shared master treatment:
 *
 *   engine → engineBus ┐
 *   water  → waterBus  ├→ busSum → compressor → soft clip → master → out
 *   sfx    → sfxBus    ┘                ↑
 *   sfx    → reverb ───────────────────┘   (short synthesised impulse)
 *
 * The compressor plus the `tanh` soft clip is what stops the result sounding
 * like isolated test tones: transients duck the engine slightly, and everything
 * shares one non-linearity so the whole mix glues instead of stacking linearly
 * into a digital-clipped mess.
 *
 * **Engine model.** A small high-strung outboard, not a V8. `f0` is the firing
 * rate (26–115 Hz, i.e. a two-cylinder from idle to screaming), and the character
 * comes from three places: sawtooth partials at f0/2f0/3f0 through a resonant
 * lowpass whose cutoff climbs with rpm (the formant sweep you hear as "revving"),
 * a dedicated high whine at 6·f0 that only appears at the top of the range, and
 * an amplitude-modulated noise rattle clocked to f0 that dominates at idle and
 * is squeezed out as the revs rise. A separate distorted growl voice is gated on
 * `throttle × (1 − speed)`, so it speaks when the engine is working — pulling
 * away, climbing a wave — and backs off once the boat is planing.
 *
 * **Verification.** Audio cannot be checked in a screenshot, so `selfTest()`
 * rebuilds the entire graph on an `OfflineAudioContext`, fires every one-shot,
 * renders it and reports peak/RMS per section. That proves the graph constructs,
 * that each cue actually produces signal, and that nothing NaNs or silently
 * clips. It does **not** prove any of it sounds good — nobody has heard it.
 */

import { CONFIG } from '../core/config';
import { clamp, clamp01 } from '../core/mathx';
import type { AudioAPI, GameContext } from '../core/types';

/** Firing rate at idle and at the redline, Hz. */
const F0_IDLE = 26;
const F0_SPAN = 89;

export interface AudioSelfTest {
  nodes: string[];
  sections: { name: string; peak: number; rms: number }[];
  ok: boolean;
  errors: string[];
}

export class GameAudio implements AudioAPI {
  private ac: BaseAudioContext | null = null;
  private live: AudioContext | null = null;

  // Master chain.
  private busSum!: GainNode;
  private master!: GainNode;
  private comp!: DynamicsCompressorNode;
  private clipper!: WaveShaperNode;
  private reverbSend!: GainNode;

  // Buses.
  private engineBus!: GainNode;
  private waterBus!: GainNode;
  private sfxBus!: GainNode;

  // Engine voices.
  private engineOsc: OscillatorNode[] = [];
  private engineGain!: GainNode;
  private engineFilter!: BiquadFilterNode;
  private whineOsc!: OscillatorNode;
  private whineGain!: GainNode;
  private growlOsc!: OscillatorNode;
  private growlGain!: GainNode;
  private growlFilter!: BiquadFilterNode;
  private rattleLfo!: OscillatorNode;
  private rattleGain!: GainNode;
  private rattleDepth!: GainNode;
  private rattleFilter!: BiquadFilterNode;

  // Water.
  private rushGain!: GainNode;
  private rushFilter!: BiquadFilterNode;
  private hissGain!: GainNode;
  private swellLfo!: OscillatorNode;

  /** Brown noise for the water wash; white for bright transients. */
  private noiseBuffer: AudioBuffer | null = null;
  private whiteBuffer: AudioBuffer | null = null;
  private nodeLog: string[] = [];

  private muted = false;
  private started = false;
  /** Rising-edge detectors for cues the game does not explicitly call. */
  private lastBoostTime = 0;
  private rpm = 0;

  // ───────────────────────────────────────────────────────────────────────────
  // Lifecycle
  // ───────────────────────────────────────────────────────────────────────────

  async unlock() {
    if (this.started) return;
    if (CONFIG.debug.harness) {
      // The harness runs muted and must never block on an audio device.
      this.muted = true;
      this.started = true;
      return;
    }
    try {
      const Ctor = window.AudioContext ?? (window as any).webkitAudioContext;
      const ac: AudioContext = new Ctor();
      await ac.resume();
      this.live = ac;
      this.ac = ac;
      this.build(ac);
      this.started = true;
      if (CONFIG.debug.enabled) console.info('[audio] graph:', this.nodeLog.join(', '));
    } catch (err) {
      console.warn('[audio] unavailable, running silent', err);
      this.muted = true;
      this.started = true;
    }
  }

  setMuted(m: boolean) {
    this.muted = m;
    if (this.master) this.master.gain.value = m ? 0 : CONFIG.audio.masterGain;
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Graph construction
  // ───────────────────────────────────────────────────────────────────────────

  /** ~2 s of brown-ish noise. Integrated white sits under an engine without hissing. */
  private makeNoise(ac: BaseAudioContext, brown: boolean) {
    const len = Math.floor(ac.sampleRate * 2);
    const buf = ac.createBuffer(1, len, ac.sampleRate);
    const d = buf.getChannelData(0);
    let last = 0;
    for (let i = 0; i < len; i++) {
      const white = Math.random() * 2 - 1;
      if (brown) {
        last = (last + 0.02 * white) / 1.02;
        d[i] = last * 3.2;
      } else {
        d[i] = white;
      }
    }
    return buf;
  }

  /**
   * `tanh`-shaped transfer curve. Linear near zero so quiet material is
   * untouched, compressing hard above ~0.6 so a stack of one-shots rounds over
   * instead of clipping into a square wave.
   */
  private makeClipCurve(drive: number) {
    const n = 2048;
    const curve = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * 2 - 1;
      curve[i] = Math.tanh(x * drive) / Math.tanh(drive);
    }
    return curve;
  }

  /** Harsher curve for the growl voice — this one is meant to sound dirty. */
  private makeDistCurve(amount: number) {
    const n = 1024;
    const curve = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * 2 - 1;
      curve[i] = ((1 + amount) * x) / (1 + amount * Math.abs(x));
    }
    return curve;
  }

  /** Exponentially decaying noise burst — a 1.1 s room, generated not loaded. */
  private makeImpulse(ac: BaseAudioContext) {
    const len = Math.floor(ac.sampleRate * 1.1);
    const buf = ac.createBuffer(2, len, ac.sampleRate);
    for (let c = 0; c < 2; c++) {
      const d = buf.getChannelData(c);
      for (let i = 0; i < len; i++) {
        const t = i / len;
        // Sparse early reflections then a smooth tail; sparse taps stop it
        // sounding like a plain noise swell.
        const sparse = i < len * 0.06 ? (Math.random() < 0.22 ? 1 : 0.08) : 1;
        d[i] = (Math.random() * 2 - 1) * Math.pow(1 - t, 3.4) * sparse;
      }
    }
    return buf;
  }

  private node<T>(name: string, n: T): T {
    this.nodeLog.push(name);
    return n;
  }

  private build(ac: BaseAudioContext) {
    this.nodeLog = [];
    this.noiseBuffer = this.makeNoise(ac, true);
    this.whiteBuffer = this.makeNoise(ac, false);

    // ── Master bus ──────────────────────────────────────────────────────────
    this.master = this.node('master:Gain', ac.createGain());
    this.master.gain.value = this.muted ? 0 : CONFIG.audio.masterGain;
    this.master.connect(ac.destination);

    this.clipper = this.node('softClip:WaveShaper', ac.createWaveShaper());
    this.clipper.curve = this.makeClipCurve(1.6);
    this.clipper.oversample = '2x';
    this.clipper.connect(this.master);

    this.comp = this.node('glue:DynamicsCompressor', ac.createDynamicsCompressor());
    this.comp.threshold.value = -17;
    this.comp.knee.value = 12;
    this.comp.ratio.value = 4;
    this.comp.attack.value = 0.006;
    this.comp.release.value = 0.19;
    this.comp.connect(this.clipper);

    this.busSum = this.node('busSum:Gain', ac.createGain());
    this.busSum.gain.value = 1;
    this.busSum.connect(this.comp);

    // Short synthesised room on the effects only. The engine stays dry: reverb
    // on a continuous tone that is supposed to be 3 m behind your head just
    // smears it.
    const convolver = this.node('room:Convolver', ac.createConvolver());
    convolver.buffer = this.makeImpulse(ac);
    convolver.normalize = true;
    this.reverbSend = this.node('reverbSend:Gain', ac.createGain());
    this.reverbSend.gain.value = 0.22;
    this.reverbSend.connect(convolver);
    const wet = this.node('reverbReturn:Gain', ac.createGain());
    wet.gain.value = 0.5;
    convolver.connect(wet).connect(this.comp);

    this.engineBus = this.node('engineBus:Gain', ac.createGain());
    this.waterBus = this.node('waterBus:Gain', ac.createGain());
    this.sfxBus = this.node('sfxBus:Gain', ac.createGain());
    this.engineBus.gain.value = 1;
    this.waterBus.gain.value = 1;
    this.sfxBus.gain.value = 1;
    this.engineBus.connect(this.busSum);
    this.waterBus.connect(this.busSum);
    this.sfxBus.connect(this.busSum);
    this.sfxBus.connect(this.reverbSend);

    // ── Engine: partials through a resonant lowpass ──────────────────────────
    this.engineFilter = this.node('engineFormant:Biquad', ac.createBiquadFilter());
    this.engineFilter.type = 'lowpass';
    this.engineFilter.frequency.value = 620;
    this.engineFilter.Q.value = 6.5;

    this.engineGain = this.node('engine:Gain', ac.createGain());
    this.engineGain.gain.value = 0;
    this.engineFilter.connect(this.engineGain).connect(this.engineBus);

    // f0, 2·f0, 3·f0 with a few cents of spread. The spread is what makes it a
    // machine with moving parts rather than an additive synth patch.
    for (const [mult, detune, type] of [
      [1, 0, 'sawtooth'],
      [1, 9, 'sawtooth'],
      [2, -7, 'sawtooth'],
      [3, 5, 'square'],
    ] as const) {
      const o = this.node(`enginePartial×${mult}:Osc`, ac.createOscillator());
      o.type = type;
      o.frequency.value = F0_IDLE * mult;
      o.detune.value = detune;
      const g = ac.createGain();
      g.gain.value = mult === 1 ? 0.5 : mult === 2 ? 0.3 : 0.14;
      o.connect(g).connect(this.engineFilter);
      o.start();
      // Stored so update() can retune them; the multiplier is recovered from
      // the initial frequency, which is why they are seeded from F0_IDLE.
      (o as any).__mult = mult;
      this.engineOsc.push(o);
    }

    // Top-end whine, only present at high rpm.
    this.whineOsc = this.node('whine:Osc', ac.createOscillator());
    this.whineOsc.type = 'triangle';
    this.whineOsc.frequency.value = F0_IDLE * 6;
    this.whineGain = this.node('whine:Gain', ac.createGain());
    this.whineGain.gain.value = 0;
    this.whineOsc.connect(this.whineGain).connect(this.engineBus);
    this.whineOsc.start();

    // Load growl: a distorted voice that speaks under load, not at speed.
    this.growlOsc = this.node('growl:Osc', ac.createOscillator());
    this.growlOsc.type = 'sawtooth';
    this.growlOsc.frequency.value = F0_IDLE * 1.5;
    const dist = this.node('growl:WaveShaper', ac.createWaveShaper());
    dist.curve = this.makeDistCurve(14);
    dist.oversample = '2x';
    this.growlFilter = this.node('growl:Biquad', ac.createBiquadFilter());
    this.growlFilter.type = 'bandpass';
    this.growlFilter.frequency.value = 320;
    this.growlFilter.Q.value = 2.6;
    this.growlGain = this.node('growl:Gain', ac.createGain());
    this.growlGain.gain.value = 0;
    this.growlOsc.connect(dist).connect(this.growlFilter).connect(this.growlGain).connect(this.engineBus);
    this.growlOsc.start();

    // Idle rattle: bandpassed noise amplitude-modulated at the firing rate.
    const rattleSrc = this.node('rattle:BufferSource', ac.createBufferSource());
    rattleSrc.buffer = this.makeNoise(ac, false);
    rattleSrc.loop = true;
    this.rattleFilter = this.node('rattle:Biquad', ac.createBiquadFilter());
    this.rattleFilter.type = 'bandpass';
    this.rattleFilter.frequency.value = 150;
    this.rattleFilter.Q.value = 1.4;
    // gain = 0.5 (DC) + LFO·0.45  →  a 0…1 chop at f0.
    const chop = this.node('rattleChop:Gain', ac.createGain());
    chop.gain.value = 0.5;
    this.rattleLfo = this.node('rattleLfo:Osc', ac.createOscillator());
    this.rattleLfo.type = 'sawtooth';
    this.rattleLfo.frequency.value = F0_IDLE;
    this.rattleDepth = this.node('rattleDepth:Gain', ac.createGain());
    this.rattleDepth.gain.value = 0.45;
    this.rattleLfo.connect(this.rattleDepth).connect(chop.gain);
    this.rattleLfo.start();
    this.rattleGain = this.node('rattle:Gain', ac.createGain());
    this.rattleGain.gain.value = 0;
    rattleSrc.connect(this.rattleFilter).connect(chop).connect(this.rattleGain).connect(this.engineBus);
    rattleSrc.start();

    // ── Water rush: brown wash + bright hiss ────────────────────────────────
    const rushSrc = this.node('rush:BufferSource', ac.createBufferSource());
    rushSrc.buffer = this.noiseBuffer;
    rushSrc.loop = true;
    this.rushFilter = this.node('rush:Biquad', ac.createBiquadFilter());
    this.rushFilter.type = 'bandpass';
    this.rushFilter.frequency.value = 280;
    this.rushFilter.Q.value = 0.7;
    this.rushGain = this.node('rush:Gain', ac.createGain());
    this.rushGain.gain.value = 0;
    rushSrc.connect(this.rushFilter).connect(this.rushGain).connect(this.waterBus);
    rushSrc.start();

    // The swell LFO breathes the wash so it is not a static band of noise.
    this.swellLfo = this.node('swellLfo:Osc', ac.createOscillator());
    this.swellLfo.type = 'sine';
    this.swellLfo.frequency.value = 0.23;
    const swellDepth = this.node('swellDepth:Gain', ac.createGain());
    swellDepth.gain.value = 150;
    this.swellLfo.connect(swellDepth).connect(this.rushFilter.frequency);
    this.swellLfo.start();

    const hissSrc = this.node('hiss:BufferSource', ac.createBufferSource());
    hissSrc.buffer = this.makeNoise(ac, false);
    hissSrc.loop = true;
    const hissFilter = this.node('hiss:Biquad', ac.createBiquadFilter());
    hissFilter.type = 'highpass';
    hissFilter.frequency.value = 2600;
    this.hissGain = this.node('hiss:Gain', ac.createGain());
    this.hissGain.gain.value = 0;
    hissSrc.connect(hissFilter).connect(this.hissGain).connect(this.waterBus);
    hissSrc.start();
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Per-frame
  // ───────────────────────────────────────────────────────────────────────────

  update(ctx: GameContext) {
    if (!this.ac || this.muted) return;
    const s = ctx.player.state;
    const now = this.ac.currentTime;

    // RPM is not just speed. Stabbing the throttle lifts it immediately — the
    // `load - speedFrac` term is the engine loading up before the hull has
    // caught up, which is most of what makes an engine feel responsive. In the
    // air the prop unloads entirely and it free-revs.
    const load = clamp01(s.appliedThrottle);
    let rpm = clamp01(s.speedFrac * 0.58 + load * 0.42 + Math.max(0, load - s.speedFrac) * 0.25);
    if (s.airborne) rpm = Math.max(rpm, 0.9);
    if (s.boostTime > 0) rpm = Math.min(1, rpm + 0.1);
    this.rpm = rpm;

    const f0 = F0_IDLE + rpm * F0_SPAN;

    for (const o of this.engineOsc) {
      o.frequency.setTargetAtTime(f0 * ((o as any).__mult as number), now, 0.05);
    }
    this.rattleLfo.frequency.setTargetAtTime(f0, now, 0.05);
    this.whineOsc.frequency.setTargetAtTime(f0 * 6, now, 0.05);
    this.growlOsc.frequency.setTargetAtTime(f0 * 1.5, now, 0.05);

    // Formant sweep: the perceptual "revving" cue.
    this.engineFilter.frequency.setTargetAtTime(
      560 + rpm * 3100 + (s.airborne ? 500 : 0),
      now,
      0.07,
    );
    const engineVol =
      CONFIG.audio.engineGain * (0.2 + load * 0.55 + rpm * 0.4) * (s.airborne ? 0.7 : 1);
    this.engineGain.gain.setTargetAtTime(engineVol, now, 0.06);

    // Whine appears only in the top half of the range, squared so it screams
    // rather than fades in.
    const whine = clamp01((rpm - 0.42) / 0.58);
    this.whineGain.gain.setTargetAtTime(CONFIG.audio.engineGain * whine * whine * 0.13, now, 0.08);

    // Growl: working hard = throttle open, boat not yet fast.
    const working = load * clamp01(1 - s.speedFrac * 0.85);
    this.growlGain.gain.setTargetAtTime(CONFIG.audio.engineGain * working * 0.5, now, 0.09);
    this.growlFilter.frequency.setTargetAtTime(280 + rpm * 520, now, 0.1);

    // Rattle dominates at idle and is squeezed out as the revs climb.
    const rattle = (1 - rpm * 0.78) * (0.35 + load * 0.3);
    this.rattleGain.gain.setTargetAtTime(CONFIG.audio.engineGain * rattle * 0.5, now, 0.08);
    this.rattleFilter.frequency.setTargetAtTime(120 + rpm * 260, now, 0.1);

    // Water rush scales with speed and all but vanishes in the air.
    const airborne = s.airborne;
    const rush = airborne ? 0.04 : clamp01(s.speedFrac * 1.15);
    this.rushGain.gain.setTargetAtTime(CONFIG.audio.waterGain * rush, now, airborne ? 0.06 : 0.12);
    this.rushFilter.frequency.setTargetAtTime(260 + rush * 1500, now, 0.12);
    this.hissGain.gain.setTargetAtTime(
      CONFIG.audio.waterGain * (airborne ? 0.01 : rush * rush * 0.34),
      now,
      airborne ? 0.05 : 0.14,
    );

    this.lastBoostTime = s.boostTime;
  }

  // ───────────────────────────────────────────────────────────────────────────
  // One-shots
  // ───────────────────────────────────────────────────────────────────────────

  /** Fire-and-forget noise burst from the shared buffer with an envelope. */
  private burst(
    ac: BaseAudioContext,
    at: number,
    dur: number,
    filter: { type: BiquadFilterType; from: number; to: number; q: number },
    peak: number,
    dest: AudioNode,
  ) {
    // White, not brown. A highpass on integrated noise removes almost all of its
    // energy — the first self-test measured the splash at a tenth the level of
    // every other cue for exactly this reason.
    const src = ac.createBufferSource();
    src.buffer = filter.type === 'highpass' ? this.whiteBuffer! : this.noiseBuffer!;
    src.loop = true;
    src.playbackRate.value = 0.8 + Math.random() * 0.5;
    const f = ac.createBiquadFilter();
    f.type = filter.type;
    f.Q.value = filter.q;
    f.frequency.setValueAtTime(filter.from, at);
    f.frequency.exponentialRampToValueAtTime(Math.max(30, filter.to), at + dur);
    const g = ac.createGain();
    g.gain.setValueAtTime(0.0001, at);
    g.gain.exponentialRampToValueAtTime(peak, at + Math.min(0.02, dur * 0.12));
    g.gain.exponentialRampToValueAtTime(0.0001, at + dur);
    src.connect(f).connect(g).connect(dest);
    src.start(at);
    src.stop(at + dur + 0.02);
  }

  /**
   * Hull slam. Three layers: a pitch-dropping body thump, a resonant hull boom
   * (bandpassed noise at 90 Hz, high Q — the "boxy" ring of a GRP hull), and the
   * water displacement, handed off to `splash`.
   */
  impact(strength: number) {
    const ac = this.ac;
    if (!ac || this.muted) return;
    const now = ac.currentTime;
    const s = clamp01(strength);

    const osc = ac.createOscillator();
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(150 + s * 110, now);
    osc.frequency.exponentialRampToValueAtTime(38, now + 0.3);
    const g = ac.createGain();
    g.gain.setValueAtTime(0.0001, now);
    g.gain.exponentialRampToValueAtTime(0.52 * s + 0.05, now + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, now + 0.42);
    osc.connect(g).connect(this.sfxBus);
    osc.start(now);
    osc.stop(now + 0.45);

    // Hull ring.
    this.burst(
      ac,
      now,
      0.26 + s * 0.16,
      { type: 'bandpass', from: 95 + s * 40, to: 70, q: 7.5 },
      0.34 * s + 0.04,
      this.sfxBus,
    );

    this.splash(s * 0.85);
  }

  /** Spray transient — bright, short, with a downward filter sweep. */
  splash(strength: number) {
    const ac = this.ac;
    if (!ac || this.muted) return;
    const s = clamp01(strength);
    const now = ac.currentTime;
    this.burst(
      ac,
      now,
      0.18 + s * 0.22,
      { type: 'highpass', from: 900 + s * 2600, to: 550, q: 0.8 },
      0.3 * s + 0.02,
      this.sfxBus,
    );
  }

  /**
   * Start-light horn. Two-tone air horn (fundamental + a fifth) with a fast
   * pitch bend into pitch and a chuff of air on the attack.
   */
  horn(pitch: number) {
    const ac = this.ac;
    if (!ac || this.muted) return;
    const now = ac.currentTime;
    const base = 196 * Math.pow(2, clamp(pitch, 0, 1.2));
    const hold = 0.24 + pitch * 0.12;

    for (const [mult, gain, type] of [
      [1, 0.2, 'sawtooth'],
      [1.5, 0.12, 'sawtooth'],
      [2, 0.07, 'square'],
    ] as const) {
      const o = ac.createOscillator();
      o.type = type;
      o.frequency.setValueAtTime(base * mult * 0.94, now);
      o.frequency.exponentialRampToValueAtTime(base * mult, now + 0.05);
      const f = ac.createBiquadFilter();
      f.type = 'lowpass';
      f.frequency.value = 1500 + pitch * 1400;
      f.Q.value = 1.2;
      const g = ac.createGain();
      g.gain.setValueAtTime(0.0001, now);
      g.gain.exponentialRampToValueAtTime(gain, now + 0.025);
      g.gain.setValueAtTime(gain, now + hold);
      g.gain.exponentialRampToValueAtTime(0.0001, now + hold + 0.26);
      o.connect(f).connect(g).connect(this.sfxBus);
      o.start(now);
      o.stop(now + hold + 0.3);
    }
    // Air chuff.
    this.burst(ac, now, 0.09, { type: 'highpass', from: 2200, to: 900, q: 0.7 }, 0.1, this.sfxBus);
  }

  /** Boost whoosh: a rising filtered noise sweep, a pitch rise and a low thump. */
  boost() {
    const ac = this.ac;
    if (!ac || this.muted) return;
    const now = ac.currentTime;

    this.burst(
      ac,
      now,
      0.5,
      { type: 'bandpass', from: 360, to: 4200, q: 1.1 },
      0.3,
      this.sfxBus,
    );

    const o = ac.createOscillator();
    o.type = 'sawtooth';
    o.frequency.setValueAtTime(150, now);
    o.frequency.exponentialRampToValueAtTime(960, now + 0.3);
    const f = ac.createBiquadFilter();
    f.type = 'bandpass';
    f.frequency.value = 800;
    f.Q.value = 4.5;
    const g = ac.createGain();
    g.gain.setValueAtTime(0.0001, now);
    g.gain.exponentialRampToValueAtTime(0.26, now + 0.04);
    g.gain.exponentialRampToValueAtTime(0.0001, now + 0.44);
    o.connect(f).connect(g).connect(this.sfxBus);
    o.start(now);
    o.stop(now + 0.48);

    // Low thump so the boost has weight as well as air.
    const t = ac.createOscillator();
    t.type = 'sine';
    t.frequency.setValueAtTime(110, now);
    t.frequency.exponentialRampToValueAtTime(45, now + 0.22);
    const tg = ac.createGain();
    tg.gain.setValueAtTime(0.0001, now);
    tg.gain.exponentialRampToValueAtTime(0.3, now + 0.012);
    tg.gain.exponentialRampToValueAtTime(0.0001, now + 0.3);
    t.connect(tg).connect(this.sfxBus);
    t.start(now);
    t.stop(now + 0.32);
  }

  /** Checkpoint blip: a rising two-note with a short slap echo. */
  checkpoint() {
    const ac = this.ac;
    if (!ac || this.muted) return;
    const now = ac.currentTime;

    const delay = ac.createDelay(0.4);
    delay.delayTime.value = 0.11;
    const dg = ac.createGain();
    dg.gain.value = 0.32;
    delay.connect(dg).connect(this.sfxBus);

    for (const [t0, freq, gain] of [
      [0, 880, 0.15],
      [0.075, 1320, 0.13],
    ] as const) {
      const o = ac.createOscillator();
      o.type = 'triangle';
      o.frequency.value = freq;
      const g = ac.createGain();
      g.gain.setValueAtTime(0.0001, now + t0);
      g.gain.exponentialRampToValueAtTime(gain, now + t0 + 0.008);
      g.gain.exponentialRampToValueAtTime(0.0001, now + t0 + 0.16);
      o.connect(g);
      g.connect(this.sfxBus);
      g.connect(delay);
      o.start(now + t0);
      o.stop(now + t0 + 0.2);
    }
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Self-test
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Build the whole graph on an `OfflineAudioContext`, exercise it, render, and
   * measure. This is the only automated check that exists for audio — a
   * screenshot proves nothing here.
   *
   * Sections are measured over disjoint time windows so a silent cue cannot hide
   * behind a loud one.
   */
  async selfTest(): Promise<AudioSelfTest> {
    const errors: string[] = [];
    const sampleRate = 44100;
    const seconds = 6;
    const Offline: typeof OfflineAudioContext =
      (globalThis as any).OfflineAudioContext ?? (globalThis as any).webkitOfflineAudioContext;
    const ac = new Offline(2, sampleRate * seconds, sampleRate);

    const savedAc = this.ac;
    const savedMuted = this.muted;
    this.ac = ac;
    this.muted = false;
    try {
      this.build(ac);
    } catch (e) {
      this.ac = savedAc;
      this.muted = savedMuted;
      return { nodes: [], sections: [], ok: false, errors: [`build threw: ${e}`] };
    }

    // Windows: [name, start, end]
    const windows: [string, number, number][] = [
      ['idle+engine', 0.0, 1.0],
      ['engine@full+water', 1.0, 2.0],
      ['impact', 2.0, 2.7],
      ['splash', 2.7, 3.3],
      ['horn', 3.3, 4.2],
      ['boost', 4.2, 5.0],
      ['checkpoint', 5.0, 5.6],
    ];

    // Drive the continuous voices directly — update() needs a GameContext we do
    // not have here, so the same parameter maths is applied by hand.
    const setEngine = (at: number, rpm: number, load: number, speed: number) => {
      const f0 = F0_IDLE + rpm * F0_SPAN;
      for (const o of this.engineOsc) o.frequency.setValueAtTime(f0 * ((o as any).__mult as number), at);
      this.rattleLfo.frequency.setValueAtTime(f0, at);
      this.whineOsc.frequency.setValueAtTime(f0 * 6, at);
      this.growlOsc.frequency.setValueAtTime(f0 * 1.5, at);
      this.engineFilter.frequency.setValueAtTime(560 + rpm * 3100, at);
      this.engineGain.gain.setValueAtTime(
        CONFIG.audio.engineGain * (0.2 + load * 0.55 + rpm * 0.4),
        at,
      );
      const whine = clamp01((rpm - 0.42) / 0.58);
      this.whineGain.gain.setValueAtTime(CONFIG.audio.engineGain * whine * whine * 0.13, at);
      this.growlGain.gain.setValueAtTime(
        CONFIG.audio.engineGain * load * clamp01(1 - speed * 0.85) * 0.5,
        at,
      );
      this.rattleGain.gain.setValueAtTime(
        CONFIG.audio.engineGain * (1 - rpm * 0.78) * (0.35 + load * 0.3) * 0.5,
        at,
      );
      this.rushGain.gain.setValueAtTime(CONFIG.audio.waterGain * speed, at);
      this.hissGain.gain.setValueAtTime(CONFIG.audio.waterGain * speed * speed * 0.34, at);
    };

    setEngine(0, 0.06, 0.05, 0.0); // idle
    setEngine(1.0, 1.0, 1.0, 0.95); // pinned
    // Silence the continuous voices for the one-shot windows so each cue is
    // measured on its own.
    setEngine(2.0, 0, 0, 0);
    this.engineGain.gain.setValueAtTime(0, 2.0);
    this.rattleGain.gain.setValueAtTime(0, 2.0);
    this.whineGain.gain.setValueAtTime(0, 2.0);
    this.growlGain.gain.setValueAtTime(0, 2.0);
    this.rushGain.gain.setValueAtTime(0, 2.0);
    this.hissGain.gain.setValueAtTime(0, 2.0);

    // One-shots are scheduled relative to `ac.currentTime`, which is 0 on an
    // offline context, so they have to be issued with the clock parked. Instead
    // of faking the clock, each cue is inlined against its window start.
    const fireAt = (t: number, fn: () => void) => {
      Object.defineProperty(ac, 'currentTime', { value: t, configurable: true });
      fn();
    };
    try {
      fireAt(2.0, () => this.impact(0.9));
      fireAt(2.75, () => this.splash(0.8));
      fireAt(3.35, () => this.horn(1.0));
      fireAt(4.25, () => this.boost());
      fireAt(5.05, () => this.checkpoint());
    } catch (e) {
      errors.push(`one-shot threw: ${e}`);
    }

    let rendered: AudioBuffer;
    try {
      rendered = await ac.startRendering();
    } catch (e) {
      this.ac = savedAc;
      this.muted = savedMuted;
      return { nodes: this.nodeLog.slice(), sections: [], ok: false, errors: [`render threw: ${e}`] };
    }

    const ch = rendered.getChannelData(0);
    const sections = windows.map(([name, a, b]) => {
      let peak = 0;
      let sum = 0;
      let n = 0;
      const i0 = Math.floor(a * sampleRate);
      const i1 = Math.min(ch.length, Math.floor(b * sampleRate));
      for (let i = i0; i < i1; i++) {
        const v = ch[i];
        if (!Number.isFinite(v)) {
          if (!errors.includes('non-finite sample')) errors.push('non-finite sample');
          continue;
        }
        const av = Math.abs(v);
        if (av > peak) peak = av;
        sum += v * v;
        n++;
      }
      return { name, peak, rms: Math.sqrt(sum / Math.max(1, n)) };
    });

    for (const sec of sections) {
      if (sec.peak < 1e-4) errors.push(`${sec.name}: silent`);
      if (sec.peak > 0.999) errors.push(`${sec.name}: hard clipped`);
    }

    this.ac = savedAc;
    this.muted = savedMuted;
    // The offline graph is garbage after this; drop the references so the live
    // graph is rebuilt cleanly if unlock() runs afterwards.
    this.engineOsc = [];
    if (savedAc && savedAc === this.live) this.build(savedAc);

    return { nodes: this.nodeLog.slice(), sections, ok: errors.length === 0, errors };
  }
}
