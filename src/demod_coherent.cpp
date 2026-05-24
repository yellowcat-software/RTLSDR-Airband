/*
 * demod_coherent.cpp
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

#include <cmath>

#include "config.h"  // SINCOSF

#include "demod_coherent.h"
#include "logging.h"  // debug_print

CoherentAmDemod::CoherentAmDemod(void) : enabled_(false), phi_(0.0f), freq_(0.0f), integ_(0.0f), k_p_(0.0f), k_i_(0.0f), dc_(0.0f), carrier_level_(0.0f) {}

CoherentAmDemod::CoherentAmDemod(int sample_rate, float loop_bandwidth_hz, float damping)
    : enabled_(true), phi_(0.0f), freq_(0.0f), integ_(0.0f), dc_(0.0f), carrier_level_(0.0f) {
    if (sample_rate <= 0 || loop_bandwidth_hz <= 0.0f || damping <= 0.0f) {
        debug_print("Invalid coherent AM parameters (sample_rate=%d, loop_bandwidth_hz=%f, damping=%f), disabling\n", sample_rate, loop_bandwidth_hz, damping);
        enabled_ = false;
        k_p_ = 0.0f;
        k_i_ = 0.0f;
        return;
    }

    // Second-order PLL gains. omega_n is the natural frequency in radians/sample;
    // the standard PI-loop mapping is k_p = 2·ζ·ω_n, k_i = ω_n².
    const float omega_n = 2.0f * (float)M_PI * loop_bandwidth_hz / (float)sample_rate;
    k_p_ = 2.0f * damping * omega_n;
    k_i_ = omega_n * omega_n;

    debug_print("Coherent AM demod: sample_rate=%d loop_bw=%f damping=%f omega_n=%f k_p=%f k_i=%f\n", sample_rate, loop_bandwidth_hz, damping, omega_n, k_p_, k_i_);
}

float CoherentAmDemod::demodulate(float re, float im) {
    if (!enabled_) {
        return 0.0f;
    }

    float sin_phi, cos_phi;
    SINCOSF(phi_, &sin_phi, &cos_phi);

    // Multiply IQ by conj(NCO): (re + j·im) · (cos − j·sin)
    const float i_demod = re * cos_phi + im * sin_phi;
    const float q_demod = im * cos_phi - re * sin_phi;

    // Costas-style phase error for AM: sign(I)·Q drives Q → 0. Robust under
    // deep modulation because sign(I) follows the AM envelope, not the carrier.
    const float err = (i_demod >= 0.0f ? 1.0f : -1.0f) * q_demod;

    // PI loop filter
    integ_ += k_i_ * err;
    freq_ = integ_ + k_p_ * err;

    // NCO advance, wrapped to [0, 2π). A while-loop handles large freq excursions.
    phi_ += freq_;
    const float two_pi = 2.0f * (float)M_PI;
    while (phi_ >= two_pi) {
        phi_ -= two_pi;
    }
    while (phi_ < 0.0f) {
        phi_ += two_pi;
    }

    // Carrier amplitude tracker: |I| converges to E[|1 + m(t)|·A] ≈ A.
    // Time constant ≈ 1000 samples (62 ms at 16 kHz, 125 ms at 8 kHz).
    carrier_level_ = 0.999f * carrier_level_ + 0.001f * fabsf(i_demod);

    // DC blocker: separate, slower IIR so it removes the carrier-amplitude
    // mean without dragging on the audio. Time constant ≈ 2000 samples.
    dc_ = 0.9995f * dc_ + 0.0005f * i_demod;
    const float audio = i_demod - dc_;

    // Normalize against the carrier so 100%-modulated AM lands near ±0.66,
    // matching the envelope path's `(wavein - agcavgfast) / (agcavgfast · 1.5)`
    // output range. Floor protects against div-by-zero at startup / dead air.
    if (carrier_level_ < 1e-6f) {
        return 0.0f;
    }
    return audio / (carrier_level_ * 1.5f);
}
