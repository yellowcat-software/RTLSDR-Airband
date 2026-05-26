/*
 * test_denoise_rnnoise.cpp
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

#include "test_base_class.h"

#include "denoise.h"

class RnnoiseDenoiseTest : public TestBaseClass {
   protected:
    void SetUp(void) { TestBaseClass::SetUp(); }
    void TearDown(void) { TestBaseClass::TearDown(); }
};

// Always-on test: the no-op default constructor must report disabled and
// leave samples untouched in apply(). Holds in both WITH_RNNOISE and
// !WITH_RNNOISE builds.
TEST_F(RnnoiseDenoiseTest, default_constructed_is_disabled) {
    RnnoiseDenoise dn;
    EXPECT_FALSE(dn.enabled());

    float samples[] = {0.1f, -0.2f, 0.3f, -0.4f, 0.5f};
    float copy[5];
    for (int i = 0; i < 5; ++i) {
        copy[i] = samples[i];
    }
    dn.apply(samples, 5);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(samples[i], copy[i]) << "disabled denoiser must leave samples untouched at index " << i;
    }
}

#ifdef WITH_RNNOISE

#include <cmath>
#include <vector>

#include "rtl_airband.h"  // WAVE_RATE

TEST_F(RnnoiseDenoiseTest, parameterized_constructor_enables) {
    RnnoiseDenoise dn(WAVE_RATE, 1.0f);
    EXPECT_TRUE(dn.enabled());
}

TEST_F(RnnoiseDenoiseTest, invalid_parameters_disable) {
    EXPECT_FALSE(RnnoiseDenoise(0, 1.0f).enabled());
    EXPECT_FALSE(RnnoiseDenoise(48001, 1.0f).enabled()) << "input above 48 kHz can't be handled (RNNoise is fixed at 48 kHz)";
    EXPECT_FALSE(RnnoiseDenoise(WAVE_RATE, -0.1f).enabled());
    EXPECT_FALSE(RnnoiseDenoise(WAVE_RATE, 1.1f).enabled());
}

TEST_F(RnnoiseDenoiseTest, passthrough_at_48khz_skips_resampler) {
    // At exactly 48 kHz the wrapper feeds RNNoise directly — no libsamplerate.
    RnnoiseDenoise dn(48000, 1.0f);
    EXPECT_TRUE(dn.enabled());
}

TEST_F(RnnoiseDenoiseTest, silence_in_produces_silence_out) {
    RnnoiseDenoise dn(WAVE_RATE, 1.0f);
    ASSERT_TRUE(dn.enabled());

    // 1 second of zeros — output (after warmup) must be near zero.
    const int n = WAVE_RATE;
    std::vector<float> samples((size_t)n, 0.0f);
    dn.apply(samples.data(), n);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(samples[(size_t)i], 0.0f, 1e-3f);
    }
}

TEST_F(RnnoiseDenoiseTest, dry_passthrough_with_wet_zero) {
    // wet=0 → output should be the (resampled-then-resampled-back) dry signal,
    // which is essentially the input minus some small resampler distortion.
    RnnoiseDenoise dn(WAVE_RATE, 0.0f);
    ASSERT_TRUE(dn.enabled());

    const int n = WAVE_RATE * 2;
    std::vector<float> tone((size_t)n);
    const float two_pi_fs = 2.0f * (float)M_PI / (float)WAVE_RATE;
    for (int i = 0; i < n; ++i) {
        tone[(size_t)i] = 0.2f * std::sin(two_pi_fs * 1000.0f * (float)i);
    }
    auto out = tone;
    dn.apply(out.data(), n);

    // Beyond the resampler warmup (~50 ms), tone power should be preserved.
    const int start = WAVE_RATE / 4;
    double in_e = 0;
    double out_e = 0;
    for (int i = start; i < n; ++i) {
        in_e += (double)tone[(size_t)i] * (double)tone[(size_t)i];
        out_e += (double)out[(size_t)i] * (double)out[(size_t)i];
    }
    EXPECT_GT(out_e, 0.5 * in_e) << "wet=0 dry-passthrough lost too much energy";
}

#endif  // WITH_RNNOISE
