import subprocess


def convert_audio_to_wav(input_file, output_file):
    # fmt: off
    command = [
        'ffmpeg',
        '-i', input_file,
        '-ar', '16000',  # Set the audio sample rate to 16kHz
        '-ac', '1',      # Set the number of audio channels to 1 (mono)
        '-c:a', 'pcm_s16le',
        '-loglevel', 'warning',
        '-hide_banner',
        '-nostats',
        '-nostdin',
        output_file
    ]
    # fmt: on
    subprocess.run(command)
