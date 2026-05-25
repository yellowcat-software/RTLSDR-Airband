/*
 * denoise_wiener.cpp
 *
 * STFT-based MMSE-LSA denoiser. Speech enhancement pipeline:
 *   1. Slide an analysis window (sqrt-Hann) over the input
 *   2. FFT, compute |Y_k|² (smoothed)
 *   3. Per-bin noise PSD via minimum-statistics over a sliding window of
 *      ~1-2 s of frames (Martin 2001), bias-corrected
 *   4. Decision-directed a-priori SNR estimator (Ephraim-Malah 1984)
 *   5. MMSE-LSA gain (Ephraim-Malah 1985):
 *        G(ξ,γ) = ξ/(1+ξ) · exp(½ E₁(v)),   v = ξ/(1+ξ)·γ
 *      where E₁ is the exponential integral (A&S 5.1.53 / 5.1.56)
 *   6. Apply gain, IFFT, synthesis window, overlap-add
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

#include "denoise.h"
#include "logging.h"  // debug_print

namespace {

// Exponential integral E₁(v) for v > 0, polynomial approximations from
// Abramowitz & Stegun 5.1.53 (small v) and 5.1.56 (large v). Error < 2e-7.
float e1_approx(float v) {
    if (v <= 0.0f) {
        return 1e30f;  // very large; treats as infinite gain correction (clamped downstream)
    }
    if (v <= 1.0f) {
        // A&S 5.1.53
        return -0.57721566f - logf(v) + v * (0.99999193f + v * (-0.24991055f + v * (0.05519968f + v * (-0.00976004f + v * 0.00107857f))));
    }
    // A&S 5.1.56:
    //   x * exp(x) * E₁(x) ≈ (x⁴ + a₃x³ + a₂x² + a₁x + a₀)
    //                       / (x⁴ + b₃x³ + b₂x² + b₁x + b₀)
    const float a0 = 0.2677737343f, a1 = 8.6347608925f, a2 = 18.059016973f, a3 = 8.5733287401f;
    const float b0 = 3.9584969228f, b1 = 21.0996530827f, b2 = 25.6329561486f, b3 = 9.5733223454f;
    const float num = (((v + a3) * v + a2) * v + a1) * v + a0;
    const float den = (((v + b3) * v + b2) * v + b1) * v + b0;
    return num / (den * v * expf(v));
}

constexpr float kFrameEnergyEpsilon = 1e-9f;  // below this, freeze noise estimate

}  // namespace

WienerDenoise::WienerDenoise(void)
    : enabled_(false),
      fft_size_(0),
      hop_(0),
      n_bins_(0),
      alpha_dd_(0.0f),
      noise_bias_(0.0f),
      gain_floor_(0.0f),
      min_window_frames_(0),
      fwd_plan_(nullptr),
      rev_plan_(nullptr),
      time_buf_(nullptr),
      freq_buf_(nullptr),
      in_w_(0),
      in_pending_(0),
      out_r_(0),
      min_pos_(0),
      min_filled_(0) {}

WienerDenoise::WienerDenoise(int sample_rate, int fft_size, float overlap, float alpha_dd, float noise_window_s, float noise_bias, float min_gain_db)
    : enabled_(true),
      fft_size_(fft_size),
      hop_(0),
      n_bins_(fft_size / 2 + 1),
      alpha_dd_(alpha_dd),
      noise_bias_(noise_bias),
      gain_floor_(powf(10.0f, min_gain_db / 20.0f)),
      min_window_frames_(0),
      fwd_plan_(nullptr),
      rev_plan_(nullptr),
      time_buf_(nullptr),
      freq_buf_(nullptr),
      in_w_(0),
      in_pending_(0),
      out_r_(0),
      min_pos_(0),
      min_filled_(0) {
    if (sample_rate <= 0 || fft_size <= 0 || (fft_size & (fft_size - 1)) != 0 || overlap <= 0.0f || overlap >= 1.0f || alpha_dd < 0.0f || alpha_dd >= 1.0f || noise_window_s <= 0.0f || noise_bias <= 0.0f) {
        debug_print("Invalid wiener parameters (sample_rate=%d fft_size=%d overlap=%f alpha_dd=%f noise_window_s=%f noise_bias=%f), disabling\n", sample_rate, fft_size, overlap, alpha_dd, noise_window_s, noise_bias);
        enabled_ = false;
        return;
    }

    hop_ = (int)((float)fft_size_ * (1.0f - overlap) + 0.5f);
    if (hop_ <= 0 || hop_ > fft_size_) {
        enabled_ = false;
        return;
    }

    const float frame_rate = (float)sample_rate / (float)hop_;
    min_window_frames_ = std::max(2, (int)(noise_window_s * frame_rate + 0.5f));

    // FFTW plans + buffers
    time_buf_ = (float*)fftwf_malloc(sizeof(float) * (size_t)fft_size_);
    freq_buf_ = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (size_t)n_bins_);
    if (!time_buf_ || !freq_buf_) {
        enabled_ = false;
        return;
    }
    fwd_plan_ = fftwf_plan_dft_r2c_1d(fft_size_, time_buf_, freq_buf_, FFTW_ESTIMATE);
    rev_plan_ = fftwf_plan_dft_c2r_1d(fft_size_, freq_buf_, time_buf_, FFTW_ESTIMATE);
    if (!fwd_plan_ || !rev_plan_) {
        enabled_ = false;
        return;
    }

    // sqrt-Hann analysis = sqrt-Hann synthesis; the product is Hann, which
    // has constant overlap-add at 50% overlap (and other Nyquist hops).
    window_.resize((size_t)fft_size_);
    for (int n = 0; n < fft_size_; ++n) {
        const float w = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)n / (float)(fft_size_ - 1));
        window_[(size_t)n] = sqrtf(w);
    }

    in_ring_.assign((size_t)fft_size_, 0.0f);
    out_ring_.assign((size_t)fft_size_, 0.0f);
    power_smooth_.assign((size_t)n_bins_, 0.0f);
    noise_psd_.assign((size_t)n_bins_, 1e-6f);  // small but non-zero
    s_prev_sq_.assign((size_t)n_bins_, 0.0f);

    min_ring_.assign((size_t)min_window_frames_, std::vector<float>((size_t)n_bins_, 1e30f));

    debug_print("WienerDenoise: rate=%d fft=%d hop=%d bins=%d alpha_dd=%f min_frames=%d bias=%f floor=%fdB(%g)\n", sample_rate, fft_size_, hop_, n_bins_, alpha_dd_, min_window_frames_, noise_bias_, min_gain_db, gain_floor_);
}

WienerDenoise::~WienerDenoise(void) {
    if (fwd_plan_) {
        fftwf_destroy_plan(fwd_plan_);
    }
    if (rev_plan_) {
        fftwf_destroy_plan(rev_plan_);
    }
    if (time_buf_) {
        fftwf_free(time_buf_);
    }
    if (freq_buf_) {
        fftwf_free(freq_buf_);
    }
}

WienerDenoise::WienerDenoise(WienerDenoise&& other) noexcept
    : enabled_(other.enabled_),
      fft_size_(other.fft_size_),
      hop_(other.hop_),
      n_bins_(other.n_bins_),
      alpha_dd_(other.alpha_dd_),
      noise_bias_(other.noise_bias_),
      gain_floor_(other.gain_floor_),
      min_window_frames_(other.min_window_frames_),
      fwd_plan_(other.fwd_plan_),
      rev_plan_(other.rev_plan_),
      time_buf_(other.time_buf_),
      freq_buf_(other.freq_buf_),
      window_(std::move(other.window_)),
      in_ring_(std::move(other.in_ring_)),
      in_w_(other.in_w_),
      in_pending_(other.in_pending_),
      out_ring_(std::move(other.out_ring_)),
      out_r_(other.out_r_),
      power_smooth_(std::move(other.power_smooth_)),
      noise_psd_(std::move(other.noise_psd_)),
      s_prev_sq_(std::move(other.s_prev_sq_)),
      min_ring_(std::move(other.min_ring_)),
      min_pos_(other.min_pos_),
      min_filled_(other.min_filled_) {
    other.enabled_ = false;
    other.fwd_plan_ = nullptr;
    other.rev_plan_ = nullptr;
    other.time_buf_ = nullptr;
    other.freq_buf_ = nullptr;
}

WienerDenoise& WienerDenoise::operator=(WienerDenoise&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (fwd_plan_) {
        fftwf_destroy_plan(fwd_plan_);
    }
    if (rev_plan_) {
        fftwf_destroy_plan(rev_plan_);
    }
    if (time_buf_) {
        fftwf_free(time_buf_);
    }
    if (freq_buf_) {
        fftwf_free(freq_buf_);
    }
    enabled_ = other.enabled_;
    fft_size_ = other.fft_size_;
    hop_ = other.hop_;
    n_bins_ = other.n_bins_;
    alpha_dd_ = other.alpha_dd_;
    noise_bias_ = other.noise_bias_;
    gain_floor_ = other.gain_floor_;
    min_window_frames_ = other.min_window_frames_;
    fwd_plan_ = other.fwd_plan_;
    rev_plan_ = other.rev_plan_;
    time_buf_ = other.time_buf_;
    freq_buf_ = other.freq_buf_;
    window_ = std::move(other.window_);
    in_ring_ = std::move(other.in_ring_);
    in_w_ = other.in_w_;
    in_pending_ = other.in_pending_;
    out_ring_ = std::move(other.out_ring_);
    out_r_ = other.out_r_;
    power_smooth_ = std::move(other.power_smooth_);
    noise_psd_ = std::move(other.noise_psd_);
    s_prev_sq_ = std::move(other.s_prev_sq_);
    min_ring_ = std::move(other.min_ring_);
    min_pos_ = other.min_pos_;
    min_filled_ = other.min_filled_;
    other.enabled_ = false;
    other.fwd_plan_ = nullptr;
    other.rev_plan_ = nullptr;
    other.time_buf_ = nullptr;
    other.freq_buf_ = nullptr;
    return *this;
}

void WienerDenoise::process_frame(void) {
    // Read fft_size_ samples from in_ring_ in chronological order, windowed.
    // in_w_ is the next-write index → in_w_ % fft_size_ is the oldest sample.
    for (int i = 0; i < fft_size_; ++i) {
        const int idx = (in_w_ + i) % fft_size_;
        time_buf_[i] = in_ring_[(size_t)idx] * window_[(size_t)i];
    }

    // Quick energy check: when the input frame is essentially zero (e.g.,
    // squelch-closed muting upstream), freeze the noise estimate so the
    // min-statistics tracker doesn't learn "noise floor = 0" and produce
    // huge gain on the next audible frame.
    float frame_energy = 0.0f;
    for (int i = 0; i < fft_size_; ++i) {
        frame_energy += time_buf_[i] * time_buf_[i];
    }
    const bool active_frame = frame_energy > kFrameEnergyEpsilon;

    fftwf_execute(fwd_plan_);

    // Per-bin smoothed power.
    const float kPowerSmoothing = 0.8f;
    for (int k = 0; k < n_bins_; ++k) {
        const float re = freq_buf_[k][0];
        const float im = freq_buf_[k][1];
        const float power = re * re + im * im;
        power_smooth_[(size_t)k] = kPowerSmoothing * power_smooth_[(size_t)k] + (1.0f - kPowerSmoothing) * power;
    }

    // Push current frame's smoothed power into min ring, then re-derive λ_d
    // as bias × min over the ring per bin. Skip the push on silent frames so
    // the tracker doesn't drift to zero during squelch-closed periods.
    if (active_frame) {
        std::vector<float>& slot = min_ring_[(size_t)min_pos_];
        for (int k = 0; k < n_bins_; ++k) {
            slot[(size_t)k] = power_smooth_[(size_t)k];
        }
        min_pos_ = (min_pos_ + 1) % min_window_frames_;
        if (min_filled_ < min_window_frames_) {
            ++min_filled_;
        }

        // Recompute per-bin minimum across the filled portion of the ring.
        for (int k = 0; k < n_bins_; ++k) {
            float mn = min_ring_[0][(size_t)k];
            for (int m = 1; m < min_filled_; ++m) {
                if (min_ring_[(size_t)m][(size_t)k] < mn) {
                    mn = min_ring_[(size_t)m][(size_t)k];
                }
            }
            // Floor avoids div-by-zero in the gain calc on early frames before
            // any signal has been seen.
            noise_psd_[(size_t)k] = std::max(noise_bias_ * mn, 1e-12f);
        }
    }

    // Decision-directed a-priori SNR + MMSE-LSA gain per bin.
    for (int k = 0; k < n_bins_; ++k) {
        const float re = freq_buf_[k][0];
        const float im = freq_buf_[k][1];
        const float Y2 = re * re + im * im;
        const float lambda_d = noise_psd_[(size_t)k];

        const float gamma = Y2 / lambda_d;                                            // a-posteriori SNR
        const float xi_inst = (gamma > 1.0f) ? (gamma - 1.0f) : 0.0f;                 // half-wave rectified
        float xi = alpha_dd_ * (s_prev_sq_[(size_t)k] / lambda_d) + (1.0f - alpha_dd_) * xi_inst;
        if (xi < 1e-6f) {
            xi = 1e-6f;
        }

        const float ratio = xi / (1.0f + xi);
        const float v = ratio * gamma;
        float gain = ratio * expf(0.5f * e1_approx(v));
        if (gain < gain_floor_) {
            gain = gain_floor_;
        } else if (gain > 1.0f) {
            gain = 1.0f;  // suppression only — never amplify
        }

        freq_buf_[k][0] = re * gain;
        freq_buf_[k][1] = im * gain;
        s_prev_sq_[(size_t)k] = (re * gain) * (re * gain) + (im * gain) * (im * gain);
    }

    fftwf_execute(rev_plan_);  // freq_buf_ → time_buf_ (unnormalized)

    // Overlap-add into output ring with synthesis window. FFTW's c2r is
    // unnormalized — divide by fft_size_ to get unit-scale samples.
    const float norm = 1.0f / (float)fft_size_;
    for (int i = 0; i < fft_size_; ++i) {
        const float val = time_buf_[i] * norm * window_[(size_t)i];
        const int idx = (out_r_ + i) % fft_size_;
        out_ring_[(size_t)idx] += val;
    }
}

void WienerDenoise::apply(float* samples, int n) {
    if (!enabled_) {
        return;
    }
    for (int i = 0; i < n; ++i) {
        in_ring_[(size_t)in_w_] = samples[i];
        in_w_ = (in_w_ + 1) % fft_size_;
        ++in_pending_;

        if (in_pending_ >= hop_) {
            process_frame();
            in_pending_ -= hop_;
        }

        samples[i] = out_ring_[(size_t)out_r_];
        out_ring_[(size_t)out_r_] = 0.0f;
        out_r_ = (out_r_ + 1) % fft_size_;
    }
}
