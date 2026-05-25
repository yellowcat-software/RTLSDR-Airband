/*
 * denoise_rnnoise.cpp
 *
 * Wraps Mozilla/JM Valin's RNNoise (recurrent-neural-network speech
 * denoiser) for use on demodulated audio. RNNoise is fixed at 48 kHz,
 * 480-sample frames; we upsample WAVE_RATE → 48 kHz with libsamplerate
 * (SRC_SINC_FASTEST), process frames, and downsample back. Output is
 * delayed by the resampler + framing latency (~30-40 ms).
 *
 * Built and exercised only when `-DRNNOISE=ON` was passed at CMake time
 * (sets WITH_RNNOISE). When not built with RNNoise, the methods below
 * compile to no-ops via the default constructor and #ifdef guards in the
 * header, so the class can still appear in `freq_t` and be passed around
 * harmlessly.
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

#include "denoise.h"

#ifdef WITH_RNNOISE

#include <algorithm>

#include <rnnoise.h>
#include <samplerate.h>

#include "logging.h"  // debug_print

namespace {

constexpr int kRnnoiseFrameSize = 480;  // 10 ms @ 48 kHz
constexpr int kRnnoiseRateHz = 48000;
constexpr float kPcm16Scale = 32768.0f;

}  // namespace

RnnoiseDenoise::RnnoiseDenoise(void) : enabled_(false), st_(nullptr), up_(nullptr), down_(nullptr), sample_rate_(0), up_ratio_(0.0), wet_(0.0f) {}

RnnoiseDenoise::RnnoiseDenoise(int sample_rate, float wet_mix) : enabled_(true), st_(nullptr), up_(nullptr), down_(nullptr), sample_rate_(sample_rate), up_ratio_((double)kRnnoiseRateHz / (double)sample_rate), wet_(wet_mix) {
    if (sample_rate <= 0 || sample_rate >= kRnnoiseRateHz || wet_mix < 0.0f || wet_mix > 1.0f) {
        debug_print("Invalid rnnoise parameters (sample_rate=%d wet=%f), disabling\n", sample_rate, wet_mix);
        enabled_ = false;
        return;
    }

    st_ = rnnoise_create(NULL);
    int err = 0;
    up_ = src_new(SRC_SINC_FASTEST, 1, &err);
    if (!st_ || !up_) {
        debug_print("rnnoise/samplerate up state alloc failed (err=%d), disabling\n", err);
        enabled_ = false;
        return;
    }
    err = 0;
    down_ = src_new(SRC_SINC_FASTEST, 1, &err);
    if (!down_) {
        debug_print("samplerate down state alloc failed (err=%d), disabling\n", err);
        enabled_ = false;
        return;
    }

    in_48k_.reserve((size_t)kRnnoiseFrameSize * 4);
    out_wave_.reserve(2048);
    debug_print("RnnoiseDenoise: sample_rate=%d up_ratio=%.6f wet=%f\n", sample_rate_, up_ratio_, wet_);
}

RnnoiseDenoise::~RnnoiseDenoise(void) {
    if (st_) {
        rnnoise_destroy(st_);
    }
    if (up_) {
        src_delete(up_);
    }
    if (down_) {
        src_delete(down_);
    }
}

RnnoiseDenoise::RnnoiseDenoise(RnnoiseDenoise&& other) noexcept
    : enabled_(other.enabled_),
      st_(other.st_),
      up_(other.up_),
      down_(other.down_),
      sample_rate_(other.sample_rate_),
      up_ratio_(other.up_ratio_),
      wet_(other.wet_),
      in_48k_(std::move(other.in_48k_)),
      out_wave_(std::move(other.out_wave_)) {
    other.enabled_ = false;
    other.st_ = nullptr;
    other.up_ = nullptr;
    other.down_ = nullptr;
}

RnnoiseDenoise& RnnoiseDenoise::operator=(RnnoiseDenoise&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (st_) {
        rnnoise_destroy(st_);
    }
    if (up_) {
        src_delete(up_);
    }
    if (down_) {
        src_delete(down_);
    }
    enabled_ = other.enabled_;
    st_ = other.st_;
    up_ = other.up_;
    down_ = other.down_;
    sample_rate_ = other.sample_rate_;
    up_ratio_ = other.up_ratio_;
    wet_ = other.wet_;
    in_48k_ = std::move(other.in_48k_);
    out_wave_ = std::move(other.out_wave_);
    other.enabled_ = false;
    other.st_ = nullptr;
    other.up_ = nullptr;
    other.down_ = nullptr;
    return *this;
}

void RnnoiseDenoise::apply(float* samples, int n) {
    if (!enabled_ || n <= 0) {
        return;
    }

    // Step 1: upsample input to 48 kHz, append to in_48k_.
    const size_t up_room = (size_t)((double)n * up_ratio_) + 64;
    const size_t prev_in = in_48k_.size();
    in_48k_.resize(prev_in + up_room);
    SRC_DATA up_data;
    up_data.data_in = samples;
    up_data.input_frames = n;
    up_data.data_out = in_48k_.data() + prev_in;
    up_data.output_frames = (long)up_room;
    up_data.end_of_input = 0;
    up_data.src_ratio = up_ratio_;
    up_data.input_frames_used = 0;
    up_data.output_frames_gen = 0;
    src_process(up_, &up_data);
    in_48k_.resize(prev_in + (size_t)up_data.output_frames_gen);

    // Step 2: drain 480-sample frames through rnnoise. Emit cleaned 48 kHz
    // samples into a local staging buffer that we then downsample.
    std::vector<float> staged_48k;
    staged_48k.reserve(in_48k_.size());
    size_t consumed = 0;
    while (in_48k_.size() - consumed >= (size_t)kRnnoiseFrameSize) {
        float in_frame[kRnnoiseFrameSize];
        float out_frame[kRnnoiseFrameSize];
        for (int i = 0; i < kRnnoiseFrameSize; ++i) {
            in_frame[i] = in_48k_[consumed + (size_t)i] * kPcm16Scale;
        }
        rnnoise_process_frame(st_, out_frame, in_frame);
        for (int i = 0; i < kRnnoiseFrameSize; ++i) {
            const float clean = out_frame[i] / kPcm16Scale;
            const float dry = in_48k_[consumed + (size_t)i];
            staged_48k.push_back(wet_ * clean + (1.0f - wet_) * dry);
        }
        consumed += (size_t)kRnnoiseFrameSize;
    }
    if (consumed > 0) {
        in_48k_.erase(in_48k_.begin(), in_48k_.begin() + (std::ptrdiff_t)consumed);
    }

    // Step 3: downsample staged 48 kHz back to WAVE_RATE, append to out_wave_.
    if (!staged_48k.empty()) {
        const size_t down_room = (size_t)((double)staged_48k.size() / up_ratio_) + 64;
        const size_t prev_out = out_wave_.size();
        out_wave_.resize(prev_out + down_room);
        SRC_DATA down_data;
        down_data.data_in = staged_48k.data();
        down_data.input_frames = (long)staged_48k.size();
        down_data.data_out = out_wave_.data() + prev_out;
        down_data.output_frames = (long)down_room;
        down_data.end_of_input = 0;
        down_data.src_ratio = 1.0 / up_ratio_;
        down_data.input_frames_used = 0;
        down_data.output_frames_gen = 0;
        src_process(down_, &down_data);
        out_wave_.resize(prev_out + (size_t)down_data.output_frames_gen);
    }

    // Step 4: emit n samples from the head of out_wave_; pad with zeros if
    // we don't have enough yet (resampler/framing warmup).
    const int avail = (int)std::min((size_t)n, out_wave_.size());
    for (int i = 0; i < avail; ++i) {
        samples[i] = out_wave_[(size_t)i];
    }
    for (int i = avail; i < n; ++i) {
        samples[i] = 0.0f;
    }
    if (avail > 0) {
        out_wave_.erase(out_wave_.begin(), out_wave_.begin() + avail);
    }
}

#else  // !WITH_RNNOISE

RnnoiseDenoise::RnnoiseDenoise(void) : enabled_(false) {}
RnnoiseDenoise::~RnnoiseDenoise(void) {}
RnnoiseDenoise::RnnoiseDenoise(RnnoiseDenoise&& other) noexcept : enabled_(other.enabled_) {
    other.enabled_ = false;
}
RnnoiseDenoise& RnnoiseDenoise::operator=(RnnoiseDenoise&& other) noexcept {
    if (this != &other) {
        enabled_ = other.enabled_;
        other.enabled_ = false;
    }
    return *this;
}
void RnnoiseDenoise::apply(float*, int) {}

#endif  // WITH_RNNOISE
