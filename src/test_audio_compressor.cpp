/*
 * test_audio_compressor.cpp
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
#include <vector>

#include "test_base_class.h"

#include "audio_compressor.h"

namespace {

constexpr int kSampleRate = 16000;
constexpr float kToneHz = 1000.0f;

std::vector<float> tone(int n, float amplitude) {
    std::vector<float> y((size_t)n);
    const float w = 2.0f * (float)M_PI * kToneHz / (float)kSampleRate;
    for (int i = 0; i < n; ++i) {
        y[(size_t)i] = amplitude * std::sin(w * (float)i);
    }
    return y;
}

float peak(const std::vector<float>& v, int start, int end) {
    float p = 0.0f;
    for (int i = start; i < end; ++i) {
        const float a = std::fabs(v[(size_t)i]);
        if (a > p) {
            p = a;
        }
    }
    return p;
}

// Default tunables matching the plan's example.
AudioCompressor make_default(float attack_ms = 5.0f, float release_ms = 200.0f) {
    return AudioCompressor(kSampleRate, /*threshold_db*/ -20.0f, /*ratio*/ 4.0f, attack_ms, release_ms, /*knee_db*/ 6.0f, /*makeup_db*/ 6.0f);
}

}  // namespace

class AudioCompressorTest : public TestBaseClass {
   protected:
    void SetUp(void) { TestBaseClass::SetUp(); }
    void TearDown(void) { TestBaseClass::TearDown(); }
};

TEST_F(AudioCompressorTest, default_constructed_is_disabled) {
    AudioCompressor c;
    EXPECT_FALSE(c.enabled());

    std::vector<float> samples = {0.1f, -0.2f, 0.3f, -0.4f, 0.5f};
    std::vector<float> copy = samples;
    c.apply(samples.data(), (int)samples.size());
    EXPECT_EQ(samples, copy) << "disabled compressor must leave samples untouched";
}

TEST_F(AudioCompressorTest, invalid_parameters_disable) {
    EXPECT_FALSE(AudioCompressor(0, -20.0f, 4.0f, 5.0f, 200.0f, 6.0f, 6.0f).enabled());
    EXPECT_FALSE(AudioCompressor(kSampleRate, -20.0f, 0.5f, 5.0f, 200.0f, 6.0f, 6.0f).enabled()) << "ratio < 1 is expansion, not compression";
    EXPECT_FALSE(AudioCompressor(kSampleRate, 1.0f, 4.0f, 5.0f, 200.0f, 6.0f, 6.0f).enabled()) << "threshold > 0 dBFS rejected";
    EXPECT_FALSE(AudioCompressor(kSampleRate, -20.0f, 4.0f, 0.0f, 200.0f, 6.0f, 6.0f).enabled());
    EXPECT_FALSE(AudioCompressor(kSampleRate, -20.0f, 4.0f, 5.0f, 0.0f, 6.0f, 6.0f).enabled());
    EXPECT_FALSE(AudioCompressor(kSampleRate, -20.0f, 4.0f, 5.0f, 200.0f, -1.0f, 6.0f).enabled()) << "knee_db must be >= 0";
}

TEST_F(AudioCompressorTest, sustained_loud_signal_is_attenuated) {
    // 0.8 peak (≈ -2 dBFS) on a -20 dB threshold with 4:1 ratio + 6 dB makeup.
    // Steady-state target peak (above the knee): threshold + (peak_dB - threshold)/ratio + makeup
    //   = -20 + (-2 + 20)/4 + 6 = -9.5 dBFS ≈ 0.335 linear.
    AudioCompressor c = make_default(/*attack_ms*/ 2.0f);
    ASSERT_TRUE(c.enabled());

    const int n = kSampleRate;  // 1 second
    auto y = tone(n, 0.8f);
    c.apply(y.data(), n);

    // Measure peak in the last 200 ms — well past the attack settling time.
    const int start = n - kSampleRate / 5;
    const float p = peak(y, start, n);
    EXPECT_NEAR(p, 0.335f, 0.06f) << "Expected loud tone compressed to ~0.335, got " << p;
}

TEST_F(AudioCompressorTest, quiet_signal_below_knee_only_gets_makeup) {
    // 0.05 peak (≈ -26 dBFS) is well below threshold − knee/2 = -23 dB.
    // Output should be input × makeup_gain = 0.05 × 10^(6/20) ≈ 0.0998.
    AudioCompressor c = make_default();
    ASSERT_TRUE(c.enabled());

    const int n = kSampleRate;
    auto y = tone(n, 0.05f);
    c.apply(y.data(), n);

    const int start = n - kSampleRate / 5;
    const float p = peak(y, start, n);
    EXPECT_NEAR(p, 0.0998f, 0.01f) << "Expected quiet tone × makeup ≈ 0.10, got " << p;
}

TEST_F(AudioCompressorTest, attack_pulls_loud_onset_down) {
    // Step from silence to 0.8 amplitude. Right after the step the compressor
    // sees a sudden loud signal; gain should fall over ≈ attack_ms.
    const float attack_ms = 5.0f;
    AudioCompressor c = make_default(attack_ms);
    ASSERT_TRUE(c.enabled());

    const int n = kSampleRate;
    auto y = tone(n, 0.8f);
    c.apply(y.data(), n);

    // After ~3× attack (15 ms ≈ 240 samples) the gain should have settled
    // close to its steady-state. Peak in [200..400) should be near 0.335.
    const int settle_start = (int)(0.003f * (float)kSampleRate * 3.0f);  // ~144
    const float p_settled = peak(y, settle_start + 50, settle_start + 250);
    EXPECT_LT(p_settled, 0.45f) << "After ~3·attack the compressor should have clamped peaks";
}

TEST_F(AudioCompressorTest, linked_stereo_applies_one_gain) {
    // Left channel loud, right channel silent. Stereo apply must use the
    // louder side for the envelope and apply the resulting gain to both,
    // so the right channel stays silent and the left matches mono behavior.
    AudioCompressor c = make_default(/*attack_ms*/ 2.0f);
    ASSERT_TRUE(c.enabled());

    const int n = kSampleRate;
    auto l = tone(n, 0.8f);
    std::vector<float> r((size_t)n, 0.0f);

    c.apply_stereo(l.data(), r.data(), n);

    // Right channel must remain bit-zero (0 × gain = 0).
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(r[(size_t)i], 0.0f) << "stereo-linked right channel must remain zero at index " << i;
    }
    // Left channel should match the mono compressed result.
    const int start = n - kSampleRate / 5;
    const float p = peak(l, start, n);
    EXPECT_NEAR(p, 0.335f, 0.06f);
}
