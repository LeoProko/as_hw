import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio
from torch.nn.functional import pad
from tqdm.auto import tqdm
import numpy as np

import src.model as module_model
from src.trainer import Trainer
from src.utils import ROOT_PATH
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser
from src.utils import calculate_eer
from src.utils import make_mel

DEFAULT_CHECKPOINT_PATH = ROOT_PATH
TEST_DATA_DIR = ROOT_PATH / "test_data"


def load_audio(path, target_sr):
    audio_tensor, sr = torchaudio.load(path)
    audio_tensor = audio_tensor[0:1, :]
    if sr != target_sr:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
    return audio_tensor


def main(config, out_file, ckpt_name):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info(
        "Loading checkpoint: {} ...".format(DEFAULT_CHECKPOINT_PATH / Path(ckpt_name))
    )
    checkpoint = torch.load(
        DEFAULT_CHECKPOINT_PATH / Path(ckpt_name), map_location=device
    )
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    # wav2spec = torchaudio.transforms.LFCC(
    #     *config["preprocessing"]["spectrogram"]["args"]
    # )
    wav2spec = config.init_obj(
        config["preprocessing"]["spectrogram"],
        torchaudio.transforms,
    )
    res = []

    with torch.no_grad():
        for fname in os.listdir(str(TEST_DATA_DIR.absolute().resolve())):
            audio = load_audio(
                str((TEST_DATA_DIR / Path(fname)).absolute().resolve()), config["sr"]
            )
            audio = audio[:, : config["max_audio_len"]]
            audio = pad(audio, (0, config["max_audio_len"] - audio.size(-1)))
            spec = wav2spec(audio).to(device)
            logits = model(spectrogram=spec)
            pred = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()[0]
            res.append((fname, pred))

    preds = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(dataloaders["eval"]):
            batch = Trainer.move_batch_to_device(batch, device)
            logits = model(**batch)
            preds.extend(torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist())
            targets.extend(batch["target"].detach().cpu().tolist())

    preds = np.array(preds)
    targets = np.array(targets)
    err, thr = calculate_eer.compute_eer(preds[targets == 1], preds[targets == 0])

    with Path("test_res").open("w") as fout:
        res_str = f"test err: {err}\ntest thr: {thr}\n\n"
        for fname, pred in res:
            res_str += f"fname: {fname}, pred: {pred:.3f}\n"
        fout.write(res_str)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    # model_config = Path(args.resume).parent / "config.json"
    with open(args.config, "r") as f:
        config = ConfigParser(json.load(f))

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    # if args.test_data_folder is not None:
    #     test_data_folder = Path(args.test_data_folder).absolute().resolve()
    #     assert test_data_folder.exists()
    #     config.config["data"] = {
    #         "test": {
    #             "batch_size": args.batch_size,
    #             "num_workers": args.jobs,
    #             "datasets": [
    #                 {
    #                     "type": "CustomDirAudioDataset",
    #                     "args": {
    #                         "audio_dir": str(test_data_folder / "audio"),
    #                         "transcription_dir": str(
    #                             test_data_folder / "transcriptions"
    #                         ),
    #                     },
    #                 }
    #             ],
    #         }
    #     }

    assert config.config.get("data", {}).get("eval", None) is not None
    # config["data"]["eval"]["batch_size"] = args.batch_size
    config["data"]["eval"]["n_jobs"] = args.jobs

    main(config, args.output, args.resume)
