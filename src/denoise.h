/*
 * denoise.h
 *
 * Post-demodulation audio denoise stages. WienerDenoise is a classical
 * MMSE-LSA estimator (Ephraim-Malah 1984/85) with a minimum-statistics
 * noise tracker (Martin 2001). RnnoiseDenoise is a stub in this commit;
 * commit 3 fills it in behind WITH_RNNOISE.
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

#ifndef _DENOISE_H
#define _DENOISE_H 1

#include <fftw3.h>
#include <vector>

#include "config.h"  // WITH_RNNOISE — must be visible identically in every TU
                     // that sees this header, since RnnoiseDenoise's member
                     // layout is gated on it.

class WienerDenoise {
   public:
    WienerDenoise(void);
    WienerDenoise(int sample_rate, int fft_size, float overlap, float alpha_dd, float noise_window_s, float noise_bias, float min_gain_db);
    ~WienerDenoise(void);

    // No copy — FFTW plans and ring buffers are owned per instance. Move is
    // supported so the config parser can build a configured instance and move
    // it into an existing slot (e.g. one returned by XCALLOC).
    WienerDenoise(const WienerDenoise&) = delete;
    WienerDenoise& operator=(const WienerDenoise&) = delete;
    WienerDenoise(WienerDenoise&& other) noexcept;
    WienerDenoise& operator=(WienerDenoise&& other) noexcept;

    // Apply denoise to *samples in-place. Introduces fft_size_ samples of
    // pipeline latency; the first fft_size_ output samples after construction
    // are zeros (warmup). No-op if !enabled().
    void apply(float* samples, int n);

    bool enabled(void) const { return enabled_; }

   private:
    void process_frame(void);

    bool enabled_;
    int fft_size_;
    int hop_;
    int n_bins_;
    float alpha_dd_;
    float noise_bias_;
    float gain_floor_;  // linear, derived from min_gain_db
    int min_window_frames_;

    fftwf_plan fwd_plan_;
    fftwf_plan rev_plan_;
    float* time_buf_;          // length fft_size_
    fftwf_complex* freq_buf_;  // length n_bins_
    std::vector<float> window_;

    // Input ring: fft_size_ samples wide. Holds the most recent fft_size_ inputs.
    std::vector<float> in_ring_;
    int in_w_;       // next write index
    int in_pending_;  // samples accumulated since last process_frame()

    // Output ring: fft_size_ samples wide. Holds the OLA buffer.
    std::vector<float> out_ring_;
    int out_r_;  // next read index

    // Per-bin running state
    std::vector<float> power_smooth_;  // smoothed |Y_k|² (Martin 2001)
    std::vector<float> noise_psd_;     // current bias-corrected noise PSD λ_d
    std::vector<float> s_prev_sq_;     // |Ŝ_{k-1}|² for decision-directed estimator

    // Minimum-statistics ring buffer: min_window_frames_ × n_bins_
    std::vector<std::vector<float>> min_ring_;
    int min_pos_;
    int min_filled_;
};

// Forward declares so the header doesn't drag in <rnnoise.h>/<samplerate.h>
// for every translation unit that includes us. The real types live in
// denoise_rnnoise.cpp.
struct DenoiseState;
typedef struct SRC_STATE_tag SRC_STATE;

class RnnoiseDenoise {
   public:
    RnnoiseDenoise(void);
#ifdef WITH_RNNOISE
    RnnoiseDenoise(int sample_rate, float wet_mix);
#endif
    ~RnnoiseDenoise(void);

    RnnoiseDenoise(const RnnoiseDenoise&) = delete;
    RnnoiseDenoise& operator=(const RnnoiseDenoise&) = delete;
    RnnoiseDenoise(RnnoiseDenoise&& other) noexcept;
    RnnoiseDenoise& operator=(RnnoiseDenoise&& other) noexcept;

    void apply(float* samples, int n);
    bool enabled(void) const { return enabled_; }

   private:
    bool enabled_;
#ifdef WITH_RNNOISE
    DenoiseState* st_;
    SRC_STATE* up_;
    SRC_STATE* down_;
    int sample_rate_;
    double up_ratio_;
    float wet_;
    std::vector<float> in_48k_;
    std::vector<float> out_wave_;
#endif
};

#endif /* _DENOISE_H */
