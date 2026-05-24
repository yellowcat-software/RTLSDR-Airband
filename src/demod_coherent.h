/*
 * demod_coherent.h
 *
 * Costas-style PLL for synchronous AM demodulation. Locks to the AM carrier
 * (small residual offset after FFT-bin pickoff + sliding-window derotation),
 * coherently demodulates by multiplying with the recovered carrier, and DC-
 * blocks the result. Output is normalized against a tracked carrier level so
 * that 100%-modulated AM lands near ±0.66 (same scale as the envelope path).
 *
 * Copyright (C) 2026 RTLSDR-Airband contributors
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <https://www.gnu.org/licenses/>.
 */

#ifndef _DEMOD_COHERENT_H
#define _DEMOD_COHERENT_H 1

class CoherentAmDemod {
   public:
    CoherentAmDemod(void);
    CoherentAmDemod(int sample_rate, float loop_bandwidth_hz, float damping);

    // Returns the next demodulated audio sample (DC-blocked, AGC-normalized).
    // Must be called once per IQ sample even when squelch is closed, so the
    // PLL keeps tracking the carrier across silent intervals.
    float demodulate(float re, float im);

    // Tracked carrier amplitude — surface so callers can see the AGC reference.
    float carrier_level(void) const { return carrier_level_; }

    bool enabled(void) const { return enabled_; }

   private:
    bool enabled_;

    // PLL state
    float phi_;    // NCO phase, radians, kept in [0, 2π)
    float freq_;   // NCO instantaneous frequency, radians per sample
    float integ_;  // integrator state of the PI loop filter
    float k_p_;    // proportional gain
    float k_i_;    // integral gain

    // Output-side trackers
    float dc_;             // slow IIR for DC blocker on I
    float carrier_level_;  // slow IIR for carrier amplitude (AGC reference)
};

#endif /* _DEMOD_COHERENT_H */
