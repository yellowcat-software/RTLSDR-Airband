/*
 * audio_compressor.cpp
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

#include <algorithm>
#include <cmath>

#include "audio_compressor.h"
#include "logging.h"  // debug_print

namespace {

constexpr float kEnvelopeFloor = 1e-9f;       // avoids log10(0) on silent input
constexpr float kSilenceShortCircuit = 1e-7f;  // below this, treat envelope as silence

// Maps a time constant in ms to a one-pole IIR coefficient at the given
// sample rate. α = 1 - exp(-1 / (τ_samples)) where τ_samples = ms/1000 · fs.
float ms_to_alpha(float ms, int sample_rate) {
    if (ms <= 0.0f || sample_rate <= 0) {
        return 1.0f;
    }
    const float tau_samples = ms * 1e-3f * (float)sample_rate;
    return 1.0f - std::exp(-1.0f / tau_samples);
}

}  // namespace

AudioCompressor::AudioCompressor(void)
    : enabled_(false),
      alpha_attack_(0.0f),
      alpha_release_(0.0f),
      threshold_db_(0.0f),
      ratio_(1.0f),
      ratio_inverse_(1.0f),
      knee_db_(0.0f),
      knee_lower_db_(0.0f),
      knee_upper_db_(0.0f),
      makeup_gain_db_(0.0f),
      envelope_(0.0f),
      gain_(1.0f) {}

AudioCompressor::AudioCompressor(int sample_rate, float threshold_db, float ratio, float attack_ms, float release_ms, float knee_db, float makeup_gain_db)
    : enabled_(true),
      alpha_attack_(ms_to_alpha(attack_ms, sample_rate)),
      alpha_release_(ms_to_alpha(release_ms, sample_rate)),
      threshold_db_(threshold_db),
      ratio_(ratio),
      ratio_inverse_(1.0f / ratio),
      knee_db_(knee_db),
      knee_lower_db_(threshold_db - knee_db * 0.5f),
      knee_upper_db_(threshold_db + knee_db * 0.5f),
      makeup_gain_db_(makeup_gain_db),
      envelope_(0.0f),
      gain_(std::pow(10.0f, makeup_gain_db * 0.05f)) {
    if (sample_rate <= 0 || ratio < 1.0f || threshold_db > 0.0f || attack_ms <= 0.0f || release_ms <= 0.0f || knee_db < 0.0f) {
        debug_print("Invalid compressor params (rate=%d threshold_db=%f ratio=%f attack_ms=%f release_ms=%f knee_db=%f makeup_db=%f), disabling\n", sample_rate, threshold_db, ratio, attack_ms, release_ms, knee_db, makeup_gain_db);
        enabled_ = false;
        return;
    }
    debug_print("AudioCompressor: rate=%d threshold=%fdB ratio=%f:1 attack=%fms release=%fms knee=%fdB makeup=%fdB\n", sample_rate, threshold_db, ratio, attack_ms, release_ms, knee_db, makeup_gain_db);
}

float AudioCompressor::step(float magnitude) {
    // Envelope follower (asymmetric one-pole IIR): fast attack on rising edges,
    // slow release on falling.
    const float alpha = (magnitude > envelope_) ? alpha_attack_ : alpha_release_;
    envelope_ = (1.0f - alpha) * envelope_ + alpha * magnitude;

    // Target gain in dB based on the smoothed envelope.
    float target_gain_db;
    if (envelope_ < kSilenceShortCircuit) {
        // Silent input: just apply makeup, don't try to log10(very small).
        target_gain_db = makeup_gain_db_;
    } else {
        const float env_db = 20.0f * std::log10(envelope_ + kEnvelopeFloor);
        if (env_db >= knee_upper_db_) {
            // Above the knee: full compression.
            const float over = env_db - threshold_db_;
            target_gain_db = -over * (1.0f - ratio_inverse_) + makeup_gain_db_;
        } else if (env_db > knee_lower_db_ && knee_db_ > 0.0f) {
            // Inside the soft-knee region: quadratic transition from 0 → full.
            const float knee_over = env_db - knee_lower_db_;
            const float reduction = (1.0f - ratio_inverse_) * (knee_over * knee_over) / (2.0f * knee_db_);
            target_gain_db = -reduction + makeup_gain_db_;
        } else {
            // Below threshold (and below the knee): no compression, just makeup.
            target_gain_db = makeup_gain_db_;
        }
    }

    const float target_gain = std::pow(10.0f, target_gain_db * 0.05f);

    // Smooth the gain change toward target with the same attack/release pair
    // (attack used when target_gain is reducing — input got louder).
    const float alpha_g = (target_gain < gain_) ? alpha_attack_ : alpha_release_;
    gain_ = (1.0f - alpha_g) * gain_ + alpha_g * target_gain;

    return gain_;
}

void AudioCompressor::apply(float* samples, int n) {
    if (!enabled_) {
        return;
    }
    for (int i = 0; i < n; ++i) {
        const float g = step(std::fabs(samples[i]));
        samples[i] *= g;
    }
}

void AudioCompressor::apply_stereo(float* l, float* r, int n) {
    if (!enabled_) {
        return;
    }
    for (int i = 0; i < n; ++i) {
        const float magnitude = std::max(std::fabs(l[i]), std::fabs(r[i]));
        const float g = step(magnitude);
        l[i] *= g;
        r[i] *= g;
    }
}
