from .utils.nemo_logging import suppress_logging; suppress_logging()

try:
    from nemo.utils import logging
    from nemo.collections.asr.models.ctc_models import EncDecCTCModel
    from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel
    from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR
    from nemo.collections.asr.parts.utils.transcribe_utils import setup_model
    from nemo.collections.asr.parts.utils.aligner_utils import Segment, Word
    from nemo.collections.asr.parts.utils.aligner_utils import (
        add_t_start_end_to_utt_obj,
        get_batch_variables,
        viterbi_decoding,
    )
except ImportError:
    raise ImportError(
        "Missing required dependency for NFA. "
        "Install NeMo with NFA utilities support:\n"
        "  pip install 'nemo_toolkit[asr]==2.7.2'\n"
        "Or install the latest development version:\n"
        "  pip install git+https://github.com/NVIDIA/NeMo.git"
    )

import copy
import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from omegaconf import OmegaConf
from .utils.data_prep import (
    get_batch_starts_ends,
    get_manifest_lines_batch,
)

import json
import shutil
import tempfile
from pathlib import Path


"""
Align the utterances in audios and texts.
Utterances results are returned.

Arguments:
    pretrained_name: string specifying the name of a CTC NeMo ASR model which will be automatically downloaded
        from NGC and used for generating the log-probs which we will use to do alignment.
        Note: NFA can only use CTC models (not Transducer models) at the moment.
    model_path: string specifying the local filepath to a CTC NeMo ASR model which will be used to generate the
        log-probs which we will use to do alignment.
        Note: NFA can only use CTC models (not Transducer models) at the moment.
        Note: if a model_path is provided, it will override the pretrained_name.
    align_using_pred_text: if True, will transcribe the audio using the specified model and then use that transcription 
        as the reference text for the forced alignment. 
    transcribe_device: None, or a string specifying the device that will be used for generating log-probs (i.e. "transcribing").
        The string needs to be in a format recognized by torch.device(). If None, NFA will set it to 'cuda' if it is available 
        (otherwise will set it to 'cpu').
    viterbi_device: None, or string specifying the device that will be used for doing Viterbi decoding. 
        The string needs to be in a format recognized by torch.device(). If None, NFA will set it to 'cuda' if it is available 
        (otherwise will set it to 'cpu').
    batch_size: int specifying batch size that will be used for generating log-probs and doing Viterbi decoding.
    use_local_attention: boolean flag specifying whether to try to use local attention for the ASR Model (will only
        work if the ASR Model is a Conformer model). If local attention is used, we will set the local attention context 
        size to [64,64].
    additional_segment_grouping_separator: an optional string or list of strings used to separate the text into smaller segments. 
        If this is not specified, then the whole text will be treated as a single segment.
    use_buffered_infer: False, if set True, using streaming to do get the logits for alignment
                        This flag is useful when aligning large audio file.
                        However, currently the chunk streaming inference does not support batch inference,
                        which means even you set batch_size > 1, it will only infer one by one instead of doing
                        the whole batch inference together.
    chunk_len_in_secs: float chunk length in seconds
    total_buffer_in_secs: float  Length of buffer (chunk + left and right padding) in seconds
    chunk_batch_size: int batch size for buffered chunk inference,
                      which will cut one audio into segments and do inference on chunk_batch_size segments at a time

    simulate_cache_aware_streaming: False, if set True, using cache aware streaming to do get the logits for alignment
"""


# from spacy.lang.en.tokenizer_exceptions import TOKENIZER_EXCEPTIONS
TOKENIZER_EXCEPTIONS = ["a.", "b.", "c.", "d.", "e.", "f.", "g.", "h.", "i.", "j.", "k.", "l.", "m.", "n.", "o.", "p.", "q.", "r.", "s.", "t.", "u.", "v.", "w.", "x.", "y.", "z.", "ä.", "ö.", "ü.", "._.", "°c.", "°f.", "°k.", "1a.m.", "1p.m.", "2a.m.", "2p.m.", "3a.m.", "3p.m.", "4a.m.", "4p.m.", "5a.m.", "5p.m.", "6a.m.", "6p.m.", "7a.m.", "7p.m.", "8a.m.", "8p.m.", "9a.m.", "9p.m.", "10a.m.", "10p.m.", "11a.m.", "11p.m.", "12a.m.", "12p.m.", "mt.", "ak.", "ala.", "apr.", "ariz.", "ark.", "aug.", "calif.", "colo.", "conn.", "dec.", "del.", "feb.", "fla.", "ga.", "ia.", "id.", "ill.", "ind.", "jan.", "jul.", "jun.", "kan.", "kans.", "ky.", "la.", "mar.", "mass.", "mich.", "minn.", "miss.", "n.c.", "n.d.", "n.h.", "n.j.", "n.m.", "n.y.", "neb.", "nebr.", "nev.", "nov.", "oct.", "okla.", "ore.", "pa.", "s.c.", "sep.", "sept.", "tenn.", "va.", "wash.", "wis.", "a.m.", "adm.", "bros.", "co.", "corp.", "d.c.", "dr.", "e.g.", "gen.", "gov.", "i.e.", "inc.", "jr.", "ltd.", "md.", "messrs.", "mo.", "mont.", "mr.", "mrs.", "ms.", "p.m.", "ph.d.", "prof.", "rep.", "rev.", "sen.", "st.", "vs.", "v.s."]


@dataclass
class AlignmentConfig:
    # Required configs
    # model_name = "stt_en_fastconformer_ctc_xxlarge"  		# Top 2 (1.1b)
    # model_name = "stt_en_fastconformer_ctc_xlarge"  		# Top 3 (0.6b)
    # model_name = "stt_en_fastconformer_hybrid_large_pc"   # Top 4 (110m)
    # ---
    # model_name = "nvidia/parakeet-tdt_ctc-1.1b"  			# Top 1
    # model_name = "nvidia/parakeet-ctc-1.1b"  				# top 5
    # model_name = "nvidia/parakeet-tdt_ctc-110m"  			# Top 6
    # model_name = "nvidia/parakeet-ctc-0.6b"  				# Top 7
    pretrained_name: Optional[str] = "stt_en_fastconformer_hybrid_large_pc"
    model_path: Optional[str] = None

    # General configs
    align_using_pred_text: bool = False
    transcribe_device: Optional[str] = None
    viterbi_device: Optional[str] = None
    batch_size: int = 1
    use_local_attention: bool = True
    additional_segment_grouping_separator: Optional[List[str]] = field(default_factory=lambda: ['.', '?', '!', '...'])

    # Buffered chunked streaming configs
    use_buffered_chunked_streaming: bool = False
    chunk_len_in_secs: float = 1.6
    total_buffer_in_secs: float = 4.0
    chunk_batch_size: int = 32

    # Cache aware streaming configs
    simulate_cache_aware_streaming: Optional[bool] = False


class ForcedAligner:
    def __init__(self, *args, **kwargs):
        if not kwargs.get("pretrained_name"):
            kwargs.pop("pretrained_name", None)

        cfg = AlignmentConfig(*args, **kwargs)

        self.cfg = OmegaConf.structured(cfg)

        # Validate config
        if self.cfg.model_path is None and self.cfg.pretrained_name is None:
            raise ValueError("Both self.cfg.model_path and self.cfg.pretrained_name cannot be None")

        if self.cfg.model_path is not None and self.cfg.pretrained_name is not None:
            raise ValueError("One of self.cfg.model_path and self.cfg.pretrained_name must be None")

        if self.cfg.batch_size < 1:
            raise ValueError("self.cfg.batch_size cannot be zero or a negative number")

        if self.cfg.additional_segment_grouping_separator == "" or self.cfg.additional_segment_grouping_separator == " ":
            raise ValueError("self.cfg.additional_grouping_separator cannot be empty string or space character")
        elif self.cfg.additional_segment_grouping_separator is not None and self.cfg.additional_segment_grouping_separator != []:
            logging.warning(
                f"`additional_segment_grouping_separator` is set to {self.cfg.additional_segment_grouping_separator}. "
                "BEHAVIOR CHANGE: Starting in NeMo 2.5.0, separators are preserved in segment text after splitting. "
                "In previous versions, separators were removed. This affects the behavior of NFA."
            )

        # init devices
        if self.cfg.transcribe_device is None:
            self.transcribe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.transcribe_device = torch.device(self.cfg.transcribe_device)
        # logging.info(f"Device to be used for transcription step (`transcribe_device`) is {transcribe_device}")

        if self.cfg.viterbi_device is None:
            self.viterbi_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.viterbi_device = torch.device(self.cfg.viterbi_device)
        # logging.info(f"Device to be used for viterbi step (`viterbi_device`) is {viterbi_device}")

        if self.transcribe_device.type == 'cuda' and self.viterbi_device.type == 'cuda':
            logging.warning(
                'Both of transcribe_device and viterbi_device are GPUs. If you run into OOM errors '
                'it may help to change one or both devices to be the CPU.'
            )

        # load model
        self.model, _ = setup_model(self.cfg, self.transcribe_device)
        self.model.eval()

        if isinstance(self.model, EncDecHybridRNNTCTCModel):
            self.model.change_decoding_strategy(decoder_type="ctc", decoding_strategy="greedy_batch")

        if self.cfg.use_local_attention:
            # logging.info(
            #     "Flag use_local_attention is set to True => will try to use local attention for model if it allows it"
            # )
            self.model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[64, 64])

        if not (isinstance(self.model, EncDecCTCModel) or isinstance(self.model, EncDecHybridRNNTCTCModel)):
            raise NotImplementedError(
                f"Model is not an instance of NeMo EncDecCTCModel or ENCDecHybridRNNTCTCModel."
                " Currently only instances of these models are supported"
            )

        self.buffered_chunk_params = {}
        if self.cfg.use_buffered_chunked_streaming:
            model_cfg = copy.deepcopy(self.model._cfg)

            OmegaConf.set_struct(model_cfg.preprocessor, False)
            # some changes for streaming scenario
            model_cfg.preprocessor.dither = 0.0
            model_cfg.preprocessor.pad_to = 0

            if model_cfg.preprocessor.normalize != "per_feature":
                logging.error(
                    "Only EncDecCTCModelBPE models trained with per_feature normalization are supported currently"
                )
            # Disable config overwriting
            OmegaConf.set_struct(model_cfg.preprocessor, True)

            feature_stride = model_cfg.preprocessor['window_stride']
            model_stride_in_secs = feature_stride * self.cfg.model_downsample_factor
            total_buffer = self.cfg.total_buffer_in_secs
            chunk_len = float(self.cfg.chunk_len_in_secs)
            tokens_per_chunk = math.ceil(chunk_len / model_stride_in_secs)
            mid_delay = math.ceil((chunk_len + (total_buffer - chunk_len) / 2) / model_stride_in_secs)
            # logging.info(f"tokens_per_chunk is {tokens_per_chunk}, mid_delay is {mid_delay}")

            self.model = FrameBatchASR(
                asr_model=self.model,
                frame_len=chunk_len,
                total_buffer=self.cfg.total_buffer_in_secs,
                batch_size=self.cfg.chunk_batch_size,
            )
            self.buffered_chunk_params = {
                "delay": mid_delay,
                "model_stride_in_secs": model_stride_in_secs,
                "tokens_per_chunk": tokens_per_chunk,
            }

        # init output_timestep_duration = None and we will calculate and update it during the first batch
        self.output_timestep_duration = None
        self.audio_filepath_parts_in_utt_id = 1

    def align(self, audio_paths: str | list[str], text_paths: str | list[str] | None = None, alignment_level: str = "word"):

        alignment_level = alignment_level.lower()
        assert alignment_level in {"segment", "word", "token"}, \
            "alignment_level must be one of: 'segment', 'word', or 'token'"

        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]
        if isinstance(text_paths, str):
            text_paths = [text_paths]
        assert len(audio_paths) == len(text_paths)
        # audio_data, text_data = [], []
        # for audio_path, text_path in zip(audio_paths, text_paths):
        #     # audio = sf.read(audio_path, dtype = "float32", always_2d = False)
        #     with open(text_path, "rt") as f:
        #         text = f.read().strip()
        #     # audio_data.append(audio)
        #     text_data.append(text)

        self._temp_dir = Path(tempfile.mkdtemp())
        manifest_filepath = self._temp_dir / "manifest.jsonl"

        manifest = []
        for audio_path, transcript in zip(audio_paths, text_paths):
            if Path(transcript).exists():
                with open(transcript, "rt", encoding = "utf-8") as f:
                    transcript = f.read().strip()
            transcript = self.make_script(transcript)
            manifest.append(
                {"audio_filepath": audio_path, "text": transcript}
            )

        with open(manifest_filepath, "wt") as f:
            for i, item in enumerate(manifest):
                f.write(json.dumps(item))
                if i < len(manifest)-1:
                    f.write("\n")

        # get start and end line IDs of batches
        starts, ends = get_batch_starts_ends(manifest_filepath, self.cfg.batch_size)

        utt_data = []

        # get alignment and save in CTM batch-by-batch
        for start, end in zip(starts, ends):
            manifest_lines_batch = get_manifest_lines_batch(manifest_filepath, start, end)

            if not self.cfg.align_using_pred_text:
                gt_text_batch = [line.get("text", "") for line in manifest_lines_batch]
            else:
                gt_text_batch = None

            (
                log_probs_batch,
                y_batch,
                T_batch,
                U_batch,
                utt_obj_batch,
                self.output_timestep_duration,
            ) = get_batch_variables(
                audio=[line["audio_filepath"] for line in manifest_lines_batch],
                model=self.model,
                segment_separators=self.cfg.additional_segment_grouping_separator,
                align_using_pred_text=self.cfg.align_using_pred_text,
                audio_filepath_parts_in_utt_id=self.audio_filepath_parts_in_utt_id,
                gt_text_batch=gt_text_batch,
                output_timestep_duration=self.output_timestep_duration,
                simulate_cache_aware_streaming=self.cfg.simulate_cache_aware_streaming,
                use_buffered_chunked_streaming=self.cfg.use_buffered_chunked_streaming,
                buffered_chunk_params=self.buffered_chunk_params,
            )

            alignments_batch = viterbi_decoding(log_probs_batch, y_batch, T_batch, U_batch, self.viterbi_device)

            for utt_obj, alignment_utt in zip(utt_obj_batch, alignments_batch):

                utt_obj = add_t_start_end_to_utt_obj(utt_obj, alignment_utt, self.output_timestep_duration)

                # utt_obj = make_ctm_files(
                #     utt_obj,
                #     self.cfg.output_dir,
                #     self.cfg.ctm_file_config,
                # )

                utt_obj = self.make_parts(utt_obj, alignment_level)

                utt_data.append(utt_obj)

        shutil.rmtree(self._temp_dir)

        return utt_data

    def make_script(self, text: str, norm: bool = False) -> str:
        text = text.replace("-", " - ").replace("—", " — ")
        text = "".join(e for e in text if e.isalnum() or e.isspace())
        if norm:
            text = text.lower()
        return " ".join(text.split())

    def make_parts(self, utt_obj, alignment_level: str = "word"):
        boundary_info_utt = []
        is_valid = lambda part: part.t_start >= 0 and part.t_end >= 0
        for segment_or_token in utt_obj.segments_and_tokens:
            if isinstance(segment_or_token, Segment):
                segment = segment_or_token
                if alignment_level == "segment" and is_valid(segment):
                    boundary_info_utt.append(segment)
                else:
                    for word_or_token in segment.words_and_tokens:
                        if isinstance(word_or_token, Word):
                            word = word_or_token
                            if alignment_level == "word" and is_valid(word):
                                boundary_info_utt.append(word)
                            else:
                                for token in word.tokens:
                                    if alignment_level == "token" and is_valid(token):
                                        boundary_info_utt.append(token)
                        else:
                            token = word_or_token
                            if alignment_level == "token" and is_valid(token):
                                boundary_info_utt.append(token)
            else:
                token = segment_or_token
                if alignment_level == "token" and is_valid(token):
                    boundary_info_utt.append(token)

        return boundary_info_utt

    def simplify(self, utt_data):
        return [
            [{"text": e.text, "start": e.t_start, "end": e.t_end} for e in utt] for utt in utt_data
        ]
