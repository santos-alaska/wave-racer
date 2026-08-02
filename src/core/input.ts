/**
 * Input. Keyboard + gamepad, normalised into a small analogue struct so the
 * boat physics never has to know where a control came from — the AI drives the
 * exact same struct, which is what keeps player and AI handling identical.
 */

export interface InputState {
  /** -1 (full left) … +1 (full right) */
  steer: number;
  /** 0 … 1 */
  throttle: number;
  /** 0 … 1 */
  brake: number;
  /** Powerslide held. */
  drift: boolean;
  /** Edge-triggered, consumed by the race state machine. */
  startPressed: boolean;
  restartPressed: boolean;
  cameraTogglePressed: boolean;
}

export function createInputState(): InputState {
  return {
    steer: 0,
    throttle: 0,
    brake: 0,
    drift: false,
    startPressed: false,
    restartPressed: false,
    cameraTogglePressed: false,
  };
}

const KEYS = {
  left: ['ArrowLeft', 'KeyA'],
  right: ['ArrowRight', 'KeyD'],
  fwd: ['ArrowUp', 'KeyW'],
  back: ['ArrowDown', 'KeyS'],
  drift: ['ShiftLeft', 'ShiftRight', 'Space'],
  start: ['Enter', 'Space'],
  restart: ['KeyR'],
  camera: ['KeyC'],
};

export class InputManager {
  readonly state = createInputState();
  private down = new Set<string>();
  private pressedThisFrame = new Set<string>();
  /** Smoothed analogue steer so keyboard input doesn't feel binary. */
  private steerSmooth = 0;

  constructor(private target: EventTarget = window) {
    target.addEventListener('keydown', this.onKeyDown);
    target.addEventListener('keyup', this.onKeyUp);
    target.addEventListener('blur', this.onBlur);
  }

  private onKeyDown = (ev: Event) => {
    const e = ev as KeyboardEvent;
    if (e.repeat) return;
    // Stop the page scrolling out from under the game.
    if (e.code.startsWith('Arrow') || e.code === 'Space') e.preventDefault();
    this.down.add(e.code);
    this.pressedThisFrame.add(e.code);
  };
  private onKeyUp = (ev: Event) => {
    this.down.delete((ev as KeyboardEvent).code);
  };
  private onBlur = () => {
    this.down.clear();
  };

  private any(list: string[]) {
    return list.some((k) => this.down.has(k));
  }
  private anyPressed(list: string[]) {
    return list.some((k) => this.pressedThisFrame.has(k));
  }

  update(dt: number) {
    const s = this.state;
    const pad = navigator.getGamepads?.().find((g) => g && g.connected) ?? null;

    // ── Steering ──────────────────────────────────────────────────────────
    let rawSteer = (this.any(KEYS.right) ? 1 : 0) - (this.any(KEYS.left) ? 1 : 0);
    if (pad) {
      const ax = pad.axes[0] ?? 0;
      if (Math.abs(ax) > 0.12) rawSteer = ax; // analogue stick wins over keys
    }
    // Ramp toward the raw value; snappy to start, slightly slower to centre.
    const rate = rawSteer === 0 ? 12 : 9;
    this.steerSmooth += (rawSteer - this.steerSmooth) * (1 - Math.exp(-rate * dt));
    s.steer = Math.abs(this.steerSmooth) < 1e-3 ? 0 : this.steerSmooth;

    // ── Throttle / brake ──────────────────────────────────────────────────
    s.throttle = this.any(KEYS.fwd) ? 1 : 0;
    s.brake = this.any(KEYS.back) ? 1 : 0;
    if (pad) {
      s.throttle = Math.max(s.throttle, pad.buttons[7]?.value ?? 0, pad.buttons[0]?.value ?? 0);
      s.brake = Math.max(s.brake, pad.buttons[6]?.value ?? 0);
    }

    // ── Drift + edge-triggered actions ────────────────────────────────────
    s.drift = this.any(KEYS.drift) || !!(pad && (pad.buttons[1]?.pressed || pad.buttons[5]?.pressed));
    s.startPressed = this.anyPressed(KEYS.start) || !!(pad && pad.buttons[9]?.pressed);
    s.restartPressed = this.anyPressed(KEYS.restart);
    s.cameraTogglePressed = this.anyPressed(KEYS.camera);

    this.pressedThisFrame.clear();
  }

  dispose() {
    this.target.removeEventListener('keydown', this.onKeyDown);
    this.target.removeEventListener('keyup', this.onKeyUp);
    this.target.removeEventListener('blur', this.onBlur);
  }
}
