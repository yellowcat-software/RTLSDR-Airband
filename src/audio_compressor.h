/*
 * audio_compressor.h
 *
 * Feed-forward dynamic-range compressor for the output-side audio chain.
 * Brings up quiet transmissions and holds back loud ones so recorded MP3s
 * sit at a consistent perceived volume regardless of per-station mic levels
 * or AM modulation depth.
 *
 * Standard threshold + ratio + soft-knee design with asymmetric attack/release
 * envelope follower and makeup gain. State is POD floats so the class survives
 * the XCALLOC-then-assign pattern used for `output_t` slots.
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

#ifndef _AUDIO_COMPRESSOR_H
#define _AUDIO_COMPRESSOR_H 1

class AudioCompressor {
   public:
    AudioCompressor(void);
    AudioCompressor(int sample_rate, float threshold_db, float ratio, float attack_ms, float release_ms, float knee_db, float makeup_gain_db);

    // Mono in-place compression. Operates on samples[0..n-1].
    void apply(float* samples, int n);

    // Linked-stereo compression: one envelope drives a single gain applied
    // to both channels, keeping the stereo image stable.
    void apply_stereo(float* l, float* r, int n);

    bool enabled(void) const { return enabled_; }

   private:
    bool enabled_;

    // Precomputed coefficients
    float alpha_attack_;
    float alpha_release_;
    float threshold_db_;
    float ratio_;
    float ratio_inverse_;   // 1 / ratio
    float knee_db_;         // 0 = hard knee
    float knee_lower_db_;   // threshold_db - knee_db/2
    float knee_upper_db_;   // threshold_db + knee_db/2
    float makeup_gain_db_;

    // State (post-XCALLOC zeros are the "freshly constructed disabled" state).
    float envelope_;
    float gain_;

    // Per-sample core: updates envelope_/gain_ for the given input magnitude
    // and returns the gain to multiply this sample by.
    float step(float magnitude);
};

#endif /* _AUDIO_COMPRESSOR_H */
