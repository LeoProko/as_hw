import json
import logging
import os
import shutil
from pathlib import Path

import torchaudio
from src.base.base_dataset import BaseDataset
from src.utils import ROOT_PATH
from speechbrain.utils.data_utils import download_file
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class AsDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "LA"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []

        if part == "train":
            split_dir = (
                self._data_dir
                / f"ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{part}.trn.txt"
            )
        else:
            split_dir = (
                self._data_dir
                / f"ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{part}.trl.txt"
            )
        audio_base_path = self._data_dir / f"ASVspoof2019_LA_{part}/flac"

        with split_dir.open("r") as fin:
            for line in tqdm(fin.readlines()):
                line = line.strip().split(" ")
                if line[-1] == "spoof":
                    target = 1
                else:
                    target = 0

                flac_path = audio_base_path / (line[1] + ".flac")
                t_info = torchaudio.info(str(flac_path))
                length = t_info.num_frames / t_info.sample_rate

                index.append(
                    {
                        "path": str(flac_path.absolute().resolve()),
                        "audio_len": length,
                        "target": target,
                    }
                )

        return index

        # index = []
        # split_dir = self._data_dir / part
        # if not split_dir.exists():
        #     self._load_dataset()

        # wav_dirs = set()
        # for dirpath, dirnames, filenames in os.walk(str(split_dir)):
        #     if any([f.endswith(".wav") for f in filenames]):
        #         wav_dirs.add(dirpath)
        # for wav_dir in tqdm(list(wav_dirs), desc=f"Preparing ljspeech folders: {part}"):
        #     wav_dir = Path(wav_dir)
        #     trans_path = list(self._data_dir.glob("*.csv"))[0]
        #     with trans_path.open() as f:
        #         for line in f:
        #             w_id = line.split("|")[0]
        #             wav_path = wav_dir / f"{w_id}.wav"
        #             if not wav_path.exists():  # elem in another part
        #                 continue
        #             t_info = torchaudio.info(str(wav_path))
        #             length = t_info.num_frames / t_info.sample_rate
        #             index.append(
        #                 {
        #                     "path": str(wav_path.absolute().resolve()),
        #                     "audio_len": length,
        #                 }
        #             )
        # return index
