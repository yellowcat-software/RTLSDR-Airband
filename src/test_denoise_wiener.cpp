/*
 * test_denoise_wiener.cpp
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
#include <random>
#include <vector>

#include "test_base_class.h"

#include "denoise.h"

namespace {

constexpr int kSampleRate = 8000;
constexpr int kFftSize = 256;
constexpr float kOverlap = 0.5f;
constexpr float kAlphaDD = 0.98f;
constexpr float kNoiseWindowS = 1.0f;
constexpr float kNoiseBias = 1.5f;
constexpr float kMinGainDb = -18.0f;
constexpr float kTonePassMin = 0.5f;  // tone preserved at ≥ 50% RMS after Wiener

float rms(const std::vector<float>& v, int start, int end) {
    if (end <= start) {
        return 0.0f;
    }
    double s = 0;
    for (int i = start; i < end; ++i) {
        s += (double)v[(size_t)i] * (double)v[(size_t)i];
    }
    return (float)std::sqrt(s / (double)(end - start));
}

std::vector<float> make_tone(int n, float hz, float amplitude) {
    std::vector<float> y((size_t)n);
    const float two_pi_fs = 2.0f * (float)M_PI / (float)kSampleRate;
    for (int i = 0; i < n; ++i) {
        y[(size_t)i] = amplitude * std::sin(two_pi_fs * hz * (float)i);
    }
    return y;
}

std::vector<float> make_white_noise(int n, float sigma, uint32_t seed) {
    std::vector<float> y((size_t)n);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, sigma);
    for (int i = 0; i < n; ++i) {
        y[(size_t)i] = nd(rng);
    }
    return y;
}

WienerDenoise make_default(void) {
    return WienerDenoise(kSampleRate, kFftSize, kOverlap, kAlphaDD, kNoiseWindowS, kNoiseBias, kMinGainDb);
}

}  // namespace

class WienerDenoiseTest : public TestBaseClass {
   protected:
    void SetUp(void) { TestBaseClass::SetUp(); }
    void TearDown(void) { TestBaseClass::TearDown(); }
};

TEST_F(WienerDenoiseTest, default_constructed_is_disabled) {
    WienerDenoise dn;
    EXPECT_FALSE(dn.enabled());

    std::vector<float> in = {0.1f, -0.2f, 0.3f, -0.4f, 0.5f};
    std::vector<float> copy = in;
    dn.apply(copy.data(), (int)copy.size());
    EXPECT_EQ(in, copy) << "disabled denoiser must leave samples untouched";
}

TEST_F(WienerDenoiseTest, invalid_parameters_disable) {
    EXPECT_FALSE(WienerDenoise(0, kFftSize, kOverlap, kAlphaDD, kNoiseWindowS, kNoiseBias, kMinGainDb).enabled());
    EXPECT_FALSE(WienerDenoise(kSampleRate, 0, kOverlap, kAlphaDD, kNoiseWindowS, kNoiseBias, kMinGainDb).enabled());
    EXPECT_FALSE(WienerDenoise(kSampleRate, 255, kOverlap, kAlphaDD, kNoiseWindowS, kNoiseBias, kMinGainDb).enabled()) << "non-power-of-two FFT must be rejected";
    EXPECT_FALSE(WienerDenoise(kSampleRate, kFftSize, 0.0f, kAlphaDD, kNoiseWindowS, kNoiseBias, kMinGainDb).enabled());
    EXPECT_FALSE(WienerDenoise(kSampleRate, kFftSize, 1.0f, kAlphaDD, kNoiseWindowS, kNoiseBias, kMinGainDb).enabled());
    EXPECT_FALSE(WienerDenoise(kSampleRate, kFftSize, kOverlap, 1.0f, kNoiseWindowS, kNoiseBias, kMinGainDb).enabled());
}

TEST_F(WienerDenoiseTest, attenuates_white_noise) {
    WienerDenoise dn = make_default();
    ASSERT_TRUE(dn.enabled());

    // 5 seconds of pure white noise. Wiener's noise estimator should learn
    // the floor and the gain should drop to near `gain_floor`.
    const int n = kSampleRate * 5;
    auto x = make_white_noise(n, 0.1f, 42);
    auto y = x;
    dn.apply(y.data(), n);

    // After the noise estimate matures (~2 s), output power should be
    // substantially below input power.
    const int start = 2 * kSampleRate;
    const float rms_in = rms(x, start, n);
    const float rms_out = rms(y, start, n);
    EXPECT_LT(rms_out, 0.4f * rms_in) << "White noise insufficiently attenuated: out=" << rms_out << " in=" << rms_in;
}

TEST_F(WienerDenoiseTest, improves_snr_on_brief_tone_in_noise) {
    // Realistic ATC workload: noise floor learned during long noise-only
    // intervals, then a brief speech-band tone arrives. The min-statistics
    // tracker is trained on the noise, so when the tone hits, the noise bins
    // get suppressed while the tone bin passes (gain ≈ 1). Mirrors how a real
    // transmission punches through trained noise.
    WienerDenoise dn = make_default();
    ASSERT_TRUE(dn.enabled());

    const int n_noise = kSampleRate * 3;  // long enough to fill the min-statistics window
    const int n_tone = kSampleRate / 3;   // ~0.33 s tone burst, brief enough that
                                          // tone power doesn't dominate the noise ring
    const int n_total = n_noise + n_tone + n_noise;

    auto noise = make_white_noise(n_total, 0.1f, 13);
    auto tone = make_tone(n_total, 1000.0f, 0.3f);

    std::vector<float> in_signal((size_t)n_total);
    for (int i = 0; i < n_total; ++i) {
        const bool tone_active = (i >= n_noise && i < n_noise + n_tone);
        in_signal[(size_t)i] = noise[(size_t)i] + (tone_active ? tone[(size_t)i] : 0.0f);
    }
    auto out_signal = in_signal;
    dn.apply(out_signal.data(), n_total);

    // Measure noise rms in the noise-only tail (after some settling).
    const int tail_start = n_noise + n_tone + kSampleRate / 4;
    const float noise_in_rms = rms(in_signal, tail_start, n_total);
    const float noise_out_rms = rms(out_signal, tail_start, n_total);
    EXPECT_LT(noise_out_rms, 0.5f * noise_in_rms) << "Wiener did not suppress trained noise floor: in=" << noise_in_rms << " out=" << noise_out_rms;

    // Measure tone preservation: RMS during the tone burst.
    const int tone_meas_start = n_noise + kSampleRate / 10;  // skip 100ms attack
    const int tone_meas_end = n_noise + n_tone;
    const float tone_in_rms = rms(in_signal, tone_meas_start, tone_meas_end);
    const float tone_out_rms = rms(out_signal, tone_meas_start, tone_meas_end);
    EXPECT_GT(tone_out_rms, kTonePassMin * tone_in_rms) << "Wiener over-suppressed tone burst riding on trained noise: in=" << tone_in_rms << " out=" << tone_out_rms;
}
