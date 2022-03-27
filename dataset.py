import csv
import os,sys
import glob
import string
import subprocess
from enum import Enum
from typing import Tuple

import inflect
import soundfile


class Datasets(Enum):
    CUSTOM = 'CUSTOM'
    COMMON_VOICE = 'COMMON_VOICE'
    LIBRI_SPEECH_TEST_CLEAN = 'LIBRI_SPEECH_TEST_CLEAN'
    LIBRI_SPEECH_TEST_OTHER = 'LIBRI_SPEECH_TEST_OTHER'
    TED_LIUM = 'TED_LIUM'


class Dataset(object):
    def size(self) -> int:
        raise NotImplementedError()

    def get(self, index: int) -> Tuple[str, str]:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

    @classmethod
    def create(cls, x: Datasets, folder: str, start: int, end: int):
        if x == Datasets.COMMON_VOICE:
            return CommonVoiceDataset(folder, start, end)
        elif x == Datasets.CUSTOM:
            return CustomDataset(folder, start, end)
        elif x == Datasets.LIBRI_SPEECH_TEST_CLEAN:
            return LibriSpeechTestCleanDataset(folder)
        elif x == Datasets.LIBRI_SPEECH_TEST_OTHER:
            return LibriSpeechTestOtherDataset(folder)
        elif x == Datasets.TED_LIUM:
            return TEDLIUMDataset(folder)
        else:
            raise ValueError(f"Cannot create {cls.__name__} of type `{x}`")

    @staticmethod
    def _normalize(text: str) -> str:
        p = inflect.engine()

        text = text.lower()

        for c in '-/–—':
            text = text.replace(c, ' ')

        for c in '‘!",.:;?“”`':
            text = text.replace(c, '')

        text = text.replace("’", "'").replace('&', 'and')

        def num2txt(y):
            return p.number_to_words(y).replace('-', ' ').replace(',', '') if any(x.isdigit() for x in y) else y

        text = ' '.join(num2txt(x) for x in text.split())

        if not all(c in " '" + string.ascii_lowercase for c in text):
            raise RuntimeError()
        if any(x.startswith("'") for x in text.split()):
            raise RuntimeError()

        return text


class CustomDataset(Dataset):
    def __init__(self, folder: str, start: int, end: int):
        self._data = list()

        file_all = os.path.join(folder, 'all.tsv')
        with open(file_all) as f:
            for i, row in enumerate(csv.DictReader(f, delimiter='\t')):
                if i < start:
                    continue
                if end != -1 and i >= end:
                    break
                
                if int(row['up_votes']) > 0 and int(row['down_votes']) == 0:

                    flac_path = os.path.join(folder, row['path']).replace('.mp3', '.flac')
                    try:
                        if soundfile.read(flac_path)[0].size > 16000 * 60:
                            continue
                    except ValueError:
                        print(f'Value error in reading: {flac_path}', file=sys.stderr)
                        continue

                    except RuntimeError:
                        print(f'Runtime error in reading: {flac_path}', file=sys.stderr)
                        continue

                    try:
                        normalized_sentence = self._normalize(row['sentence'])
                    except RuntimeError:
                        continue

                    print(flac_path)
                    self._data.append((flac_path, normalized_sentence))

        print("Loaded {} files".format(len(self._data)))

    def size(self) -> int:
        return len(self._data)

    def get(self, index: int) -> Tuple[str, str]:
        return self._data[index]

    def __str__(self) -> str:
        return 'Custom'


class CommonVoiceDataset(Dataset):
    def __init__(self, folder: str, start: int, end: int):
        self._data = list()

        header = 'client_id	path	sentence	up_votes	down_votes	age	gender	accents	locale	segment'
        
        file_all = os.path.join(folder, 'all.tsv')
        if not os.path.exists(file_all):
            lines = set()
            for tsv in ['test', 'train', 'validated', 'dev', 'other']:
                with open(os.path.join(folder, tsv+'.tsv')) as f:
                    for line in f:
                        lines.add(line.rstrip())
            lines.remove(header)
            lines = sorted(lines)
            with open(file_all, 'w') as f:
                print(header, file=f)
                for line in lines:
                    print(line, file=f)

        with open(file_all) as f:
            for i, row in enumerate(csv.DictReader(f, delimiter='\t')):
                if i < start or (end != -1 and i >= end):
                    continue
                if int(row['up_votes']) > 0 and int(row['down_votes']) == 0:
                    mp3_path = os.path.join(folder, 'clips', row['path'])
                    flac_path = mp3_path.replace('.mp3', '.flac')
                    if not os.path.exists(flac_path):
                        args = [
                            'ffmpeg',
                            '-i',
                            mp3_path,
                            '-ac', '1',
                            '-ar', '16000',
                            flac_path,
                        ]
                        subprocess.check_output(args)
                    elif soundfile.read(flac_path)[0].size > 16000 * 60:
                        continue

                    try:
                        print(flac_path)
                        self._data.append((flac_path, self._normalize(row['sentence'])))
                    except RuntimeError:
                        continue

        print("Loaded {} files".format(len(self._data)))

    def size(self) -> int:
        return len(self._data)

    def get(self, index: int) -> Tuple[str, str]:
        return self._data[index]

    def __str__(self) -> str:
        return 'CommonVoice'


class LibriSpeechTestCleanDataset(Dataset):
    def __init__(self, folder: str):
        self._data = list()

        for speaker_id in os.listdir(folder):
            speaker_folder = os.path.join(folder, speaker_id)
            for chapter_id in os.listdir(speaker_folder):
                chapter_folder = os.path.join(speaker_folder, chapter_id)

                with open(os.path.join(chapter_folder, f'{speaker_id}-{chapter_id}.trans.txt'), 'r') as f:
                    transcripts = dict(x.split(' ', maxsplit=1) for x in f.readlines())

                for x in os.listdir(chapter_folder):
                    if x.endswith('.flac'):
                        self._data.append((os.path.join(chapter_folder, x), transcripts[x.replace('.flac', '')]))

    def size(self) -> int:
        return len(self._data)

    def get(self, index: int) -> Tuple[str, str]:
        return self._data[index]

    def __str__(self) -> str:
        return 'LibriSpeech `test-clean`'


class LibriSpeechTestOtherDataset(LibriSpeechTestCleanDataset):
    def __init__(self, folder: str):
        super().__init__(folder)

    def __str__(self) -> str:
        return 'LibriSpeech `test-other`'


class TEDLIUMDataset(Dataset):
    def __init__(self, folder: str):
        self._data = list()

        test_folder = os.path.join(folder, 'test')
        audio_folder = os.path.join(test_folder, 'sph')
        caption_folder = os.path.join(test_folder, 'stm')
        for x in os.listdir(caption_folder):
            with open(os.path.join(caption_folder, x)) as f:
                for row in csv.reader(f, delimiter=" "):
                    if row[2] == "inter_segment_gap":
                        continue
                    start_sec = float(row[3])
                    end_sec = float(row[4])

                    try:
                        transcript = self._normalize(" ".join(row[6:]).replace(" '", "'"))
                    except RuntimeError:
                        continue

                    sph_path = os.path.join(audio_folder, x.replace('.stm', '.sph'))
                    flac_path = sph_path.replace('.sph', f'_{start_sec:.3f}_{end_sec:.3f}.flac')

                    if not os.path.exists(flac_path):
                        args = [
                            'ffmpeg',
                            '-i',
                            sph_path,
                            '-ac', '1',
                            '-ar', '16000',
                            '-ss', f'{start_sec:.3f}',
                            '-to', f'{end_sec:.3f}',
                            flac_path,
                        ]
                        subprocess.check_output(args)

                    self._data.append((flac_path, transcript))

    def size(self) -> int:
        return len(self._data)

    def get(self, index: int) -> Tuple[str, str]:
        return self._data[index]

    def __str__(self) -> str:
        return 'TED-LIUM'


__all__ = ['Datasets', 'Dataset']
