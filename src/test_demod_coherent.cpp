/*
 * test_demod_coherent.cpp
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

#include "demod_coherent.h"

namespace {

constexpr int kSampleRate = 16000;
constexpr float kCarrier = 0.5f;
constexpr float kModFreq = 1000.0f;  // 1 kHz speech-band tone
constexpr float kModDepth = 0.7f;    // 70% modulation depth

// Generate a baseband AM IQ signal: s(n) = A · (1 + m·cos(2π·f_m·n/fs))
//                                          · exp(j(2π·f_off·n/fs + φ_0))
// f_off is the residual carrier offset after FFT-bin pickoff + derotation
// (the PLL has to track this out).
std::vector<std::pair<float, float>> synth_am(int n_samples, float f_off_hz, float phi_0, float carrier = kCarrier, float mod_freq = kModFreq, float mod_depth = kModDepth) {
    std::vector<std::pair<float, float>> iq;
    iq.reserve(n_samples);
    const float two_pi_fs = 2.0f * (float)M_PI / (float)kSampleRate;
    for (int n = 0; n < n_samples; ++n) {
        const float envelope = carrier * (1.0f + mod_depth * std::cos(two_pi_fs * mod_freq * (float)n));
        const float phase = two_pi_fs * f_off_hz * (float)n + phi_0;
        iq.emplace_back(envelope * std::cos(phase), envelope * std::sin(phase));
    }
    return iq;
}

// RMS correlation between two equal-length signals, both zero-meaned, both
// unit-rms-normalized. Returns value in [-1, 1].
float corr(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0f;
    }
    double ma = 0, mb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        ma += a[i];
        mb += b[i];
    }
    ma /= (double)a.size();
    mb /= (double)a.size();
    double num = 0, da = 0, db = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double xa = a[i] - ma;
        const double xb = b[i] - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    if (da <= 0 || db <= 0) {
        return 0.0f;
    }
    return (float)(num / std::sqrt(da * db));
}

std::vector<float> reference_mod(int n_samples) {
    std::vector<float> ref;
    ref.reserve(n_samples);
    const float two_pi_fs = 2.0f * (float)M_PI / (float)kSampleRate;
    for (int n = 0; n < n_samples; ++n) {
        ref.push_back(kModDepth * std::cos(two_pi_fs * kModFreq * (float)n));
    }
    return ref;
}

}  // namespace

class CoherentAmTest : public TestBaseClass {
   protected:
    void SetUp(void) { TestBaseClass::SetUp(); }
    void TearDown(void) { TestBaseClass::TearDown(); }
};

TEST_F(CoherentAmTest, default_constructed_is_disabled) {
    CoherentAmDemod demod;
    EXPECT_FALSE(demod.enabled());
    // demodulate() on a disabled instance must return 0 (don't trip undefined behavior).
    EXPECT_EQ(demod.demodulate(0.5f, 0.5f), 0.0f);
}

TEST_F(CoherentAmTest, invalid_parameters_disable) {
    CoherentAmDemod bad_rate(0, 50.0f, 0.707f);
    EXPECT_FALSE(bad_rate.enabled());

    CoherentAmDemod bad_bw(kSampleRate, 0.0f, 0.707f);
    EXPECT_FALSE(bad_bw.enabled());

    CoherentAmDemod bad_damp(kSampleRate, 50.0f, 0.0f);
    EXPECT_FALSE(bad_damp.enabled());
}

TEST_F(CoherentAmTest, locks_and_recovers_audio_with_zero_offset) {
    CoherentAmDemod demod(kSampleRate, 50.0f, 0.707f);
    ASSERT_TRUE(demod.enabled());

    const int n_samples = kSampleRate;  // 1 second
    auto iq = synth_am(n_samples, /*f_off=*/0.0f, /*phi_0=*/0.0f);

    std::vector<float> out;
    out.reserve(n_samples);
    for (const auto& s : iq) {
        out.push_back(demod.demodulate(s.first, s.second));
    }

    // Discard the first ~200 ms — let PLL/DC/AGC trackers settle.
    const int skip = kSampleRate / 5;
    auto ref = reference_mod(n_samples);
    std::vector<float> out_tail(out.begin() + skip, out.end());
    std::vector<float> ref_tail(ref.begin() + skip, ref.end());
    EXPECT_GT(corr(out_tail, ref_tail), 0.95f) << "Coherent demod should recover modulating tone with >0.95 correlation";

    // Carrier level should converge near the synthesized amplitude.
    EXPECT_NEAR(demod.carrier_level(), kCarrier, 0.10f);
}

TEST_F(CoherentAmTest, locks_with_small_frequency_offset) {
    CoherentAmDemod demod(kSampleRate, 50.0f, 0.707f);
    ASSERT_TRUE(demod.enabled());

    const int n_samples = 2 * kSampleRate;  // 2 seconds — give the PLL room to lock
    auto iq = synth_am(n_samples, /*f_off=*/30.0f, /*phi_0=*/0.7f);

    std::vector<float> out;
    out.reserve(n_samples);
    for (const auto& s : iq) {
        out.push_back(demod.demodulate(s.first, s.second));
    }

    // Skip 500 ms for the PLL to lock on a 50 Hz loop-bw filter.
    const int skip = kSampleRate / 2;
    auto ref = reference_mod(n_samples);
    std::vector<float> out_tail(out.begin() + skip, out.end());
    std::vector<float> ref_tail(ref.begin() + skip, ref.end());
    EXPECT_GT(corr(out_tail, ref_tail), 0.9f) << "Coherent demod should track 30 Hz residual offset";
}

TEST_F(CoherentAmTest, recovers_audio_under_awgn) {
    CoherentAmDemod demod(kSampleRate, 50.0f, 0.707f);
    ASSERT_TRUE(demod.enabled());

    const int n_samples = 2 * kSampleRate;
    auto iq = synth_am(n_samples, /*f_off=*/10.0f, /*phi_0=*/0.0f);

    // Additive complex Gaussian noise at ~6 dB CNR relative to the carrier.
    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, kCarrier * 0.5f);
    for (auto& s : iq) {
        s.first += noise(rng);
        s.second += noise(rng);
    }

    std::vector<float> out;
    out.reserve(n_samples);
    for (const auto& s : iq) {
        out.push_back(demod.demodulate(s.first, s.second));
    }

    const int skip = kSampleRate / 2;
    auto ref = reference_mod(n_samples);
    std::vector<float> out_tail(out.begin() + skip, out.end());
    std::vector<float> ref_tail(ref.begin() + skip, ref.end());
    // Noise drags correlation down but coherent demod should still beat random.
    EXPECT_GT(corr(out_tail, ref_tail), 0.5f) << "Coherent demod should still correlate with audio under +6 dB CNR";
}
