"""
test_rnnoise_denoise.py — AM signal demodulated with RNNoise denoise.

Regression test: with `rnnoise.enabled = true`, the AM channel must still
produce an MP3 output of the expected duration. Asserts the AUD-8 path
doesn't break the basic AM signal flow.

Skipped automatically when --rnnoise-binary is not provided, since the
RNNoise path is only compiled in when `-DRNNOISE=ON`.
"""

from pathlib import Path

import pytest

from conftest import CACHE_DIR, BinaryUnderTest, run_rtl_airband
from helpers import config_writer, iq_generator, output_validator, stats_validator

SAMPLE_RATE = 2_048_000
CENTERFREQ_HZ = 120_000_000
CHANNEL_OFFSET_HZ = 25_000
AUDIO_TONE_HZ = 1_000
DURATION_S = 10.0
TOTAL_IQ_DURATION_S = DURATION_S + 2 * iq_generator.NOISE_PAD_S
TIMEOUT_S = TOTAL_IQ_DURATION_S * 3 + 30


def pytest_generate_tests(metafunc):
    """Parametrize test_rnnoise_denoise over --rnnoise-binary only."""
    if "binary_under_test" in metafunc.fixturenames:
        rnnoise_bins: list[BinaryUnderTest] = (
            metafunc.config._rtlsdr_rnnoise_binaries
        )
        if not rnnoise_bins:
            metafunc.parametrize(
                "binary_under_test",
                [
                    pytest.param(
                        None,
                        marks=pytest.mark.skip(
                            reason="--rnnoise-binary not provided"
                        ),
                    )
                ],
                ids=["rnnoise-not-built"],
            )
        else:
            metafunc.parametrize(
                "binary_under_test",
                rnnoise_bins,
                ids=[b.label for b in rnnoise_bins],
            )


def test_rnnoise_denoise(
    binary_under_test: BinaryUnderTest,
    test_output_dir: Path,
    mp3_tolerance: float,
    max_overrun_count: int,
    speedup_factor: float,
) -> None:
    """AM signal with RNNoise denoise → MP3 must contain ≈10s of audio."""
    iq_file = iq_generator.get_or_generate_am(
        offset_hz=CHANNEL_OFFSET_HZ,
        audio_hz=AUDIO_TONE_HZ,
        duration_s=DURATION_S,
        cache_dir=CACHE_DIR,
    )

    config_path = test_output_dir / "rtl_airband.conf"
    filename_template = "rnnoise_denoise"

    config_writer.write_config(
        config_path=config_path,
        iq_filepath=iq_file,
        sample_rate=SAMPLE_RATE,
        centerfreq_hz=CENTERFREQ_HZ,
        channels=[
            {
                "freq_hz": CENTERFREQ_HZ + CHANNEL_OFFSET_HZ,
                "output_filename_template": filename_template,
                "rnnoise": {
                    "enabled": True,
                    "wet": 1.0,
                },
            }
        ],
        output_dir=test_output_dir,
        speedup_factor=speedup_factor,
        mode="multichannel",
        mp3_tmp_dir=test_output_dir,
        stats_filepath=test_output_dir / "stats.txt",
    )

    run_rtl_airband(binary_under_test.path, config_path, timeout_s=TIMEOUT_S)

    output_validator.validate_mp3(
        mp3_dir=test_output_dir,
        filename_template=filename_template,
        expected_duration_s=DURATION_S,
        tolerance=mp3_tolerance,
    )

    stats = stats_validator.load(test_output_dir / "stats.txt")
    freq_hz = CENTERFREQ_HZ + CHANNEL_OFFSET_HZ
    assert (
        stats.channel("channel_activity_counter", freq_hz) > 0
    ), "Expected non-zero activity counter when RNNoise denoise is active"
    assert (
        stats.device("buffer_overflow_count") == 0
    ), "Unexpected device buffer overflow"
    stats_validator.assert_no_excessive_overruns(stats, max_overrun_count)
