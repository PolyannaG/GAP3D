import os
import io
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List, Any
import time
import torch, gc
import glob
import transformers
import tokenizers
import random
from blip3o.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_IDX
from torch.utils.data import Dataset
from blip3o.train.blip3o_trainer import blip3oTrainer
from blip3o import conversation as conversation_lib
from blip3o.model import *
from blip3o.mm_utils import tokenizer_image_token
from PIL import Image, ImageFile
from datasets import load_dataset, concatenate_datasets
from pathlib import Path
from datasets.utils.logging import set_verbosity_info
from transformers import logging as tf_logging, TrainerCallback, TrainerState, TrainerControl
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoProcessor
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from blip3o.train.eval_retrieval import (
    batch_ground_truth_latents,
    per_image_text_predicted_latents_via_loader,
    _cache_prefix, try_load_cached, save_cache,
    write_memmap, memmaps_exist, open_memmap_shards,
    retrieval_recall_at_k, retrieval_q_vs_db_shards,
    load_coco_as_hfds,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True
transform_und_images = T.Compose([T.Resize(448, interpolation=InterpolationMode.BICUBIC, antialias=True), T.CenterCrop(448)])

set_verbosity_info()
tf_logging.set_verbosity_info()

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=True)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    gen_vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_gen_mlp_adapter: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    gen_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    n_query: Optional[int] = field(default=729)  # clip 576, siglip 729, radio: 512/16=32, 32*32=1024, should we do 432-->729?
    n_und_query: Optional[int] = field(default=729)  # clip 576, siglip 729
    gen_pooling: Optional[str] = field(default="all")  # options are: pool2d_3, pool2d_9, seq_3, seq_9, seq_27
    predict_summary_token: bool = field(default=False)
    predict_dino_grid: bool = field(default=False)
    num_register_tokens: Optional[int] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    journeyDB_folder: Optional[str] = field(default=None)
    shortcaption_image_folder: Optional[str] = field(default=None)
    data_type: Optional[str] = field(default="mix")
    image_aspect_ratio: str = "square"

    eval_coco_root: Optional[str] = None     # e.g. "/path/to/coco" (has annotations/, val2017/)
    eval_coco_split: str = "val2017"         # "train2017" or "val2017"

    eval_mapper_image_folder: Optional[str] = None  # folder of .tar shards for mapper eval
    eval_mapper_num_samples: Optional[int] = None   # cap N examples (None = all)



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."},
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."},
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

    
    # DB-only caching (memmap or in-memory npy)
    eval_use_memmap: bool = True
    eval_mmap_dir: str = field(default_factory=lambda: os.environ.get("KNN_MMAP_DIR", "."))
    eval_recompute_db_cache: bool = False
    eval_cache_dir: str = field(default_factory=lambda: os.environ.get("ENCODER_CACHE_DIR", ".encoder_cache"))

    eval_workers: int = 8
    eval_q_chunk: int = 4096
    eval_db_chunk: int = 65536



def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_vision_tower_state_maybe_zero_3(named_params, keys_to_match=[""]):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, vision_tower: str):
    """Collects the state dict and dump to disk."""

    # if getattr(trainer.args, "tune_vision_model", False):

    if trainer.deepspeed:
        torch.cuda.synchronize()
    

    # Only save Adapter
    keys_to_match = ["mm_projector"]
    if getattr(trainer.args, "use_im_start_end", False):
        keys_to_match.extend(["embed_tokens", "embed_in"])

    weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    trainer.model.config.save_pretrained(output_dir)

    current_folder = output_dir.split("/")[-1]
    parent_folder = os.path.dirname(output_dir)
    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if current_folder.startswith("checkpoint-"):
            mm_projector_folder = os.path.join(parent_folder, "mm_projector")
            os.makedirs(mm_projector_folder, exist_ok=True)
            torch.save(
                weight_to_save,
                os.path.join(mm_projector_folder, f"{current_folder}.bin"),
            )
        else:
            torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))

    keys_to_match = ["gen_projector"]
    if getattr(trainer.args, "use_im_start_end", False):
        keys_to_match.extend(["embed_tokens", "embed_in"])

    weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    trainer.model.config.save_pretrained(output_dir)

    current_folder = output_dir.split("/")[-1]
    parent_folder = os.path.dirname(output_dir)
    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if current_folder.startswith("checkpoint-"):
            mm_projector_folder = os.path.join(parent_folder, "gen_projector")
            os.makedirs(mm_projector_folder, exist_ok=True)
            torch.save(
                weight_to_save,
                os.path.join(mm_projector_folder, f"{current_folder}.bin"),
            )
        else:
            torch.save(weight_to_save, os.path.join(output_dir, f"gen_projector.bin"))

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):


    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation



def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    und_placeholder = "<|vision_start|>" + "<|image_pad|>" * data_args.n_und_query + "<|vision_end|>"
    gen_placeholder = ""
    # "[IMG]" + "<image>" * data_args.n_query + "[/IMG]"
    inst_type = None
    for source in sources:  # [instance]
        for sentence in source:
            if sentence["from"] == "human" and "<image>" in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, und_placeholder).strip()
                inst_type = "und"
            elif sentence["from"] == "gpt" and "<image>" in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, gen_placeholder).strip()
                inst_type = "gen"
    return sources, inst_type



def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"

        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            # First is bos token we don't need here
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )



def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        # assert DEFAULT_IMAGE_TOKEN in source[0]['value'] or DEFAULT_IMAGE_TOKEN in source[1]['value']
        conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.version == "llama3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)



class LazySupervisedMixDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        super(LazySupervisedMixDataset, self).__init__()

        self.data_args = data_args
        list_data_dict = []


        # load journeyDB_T2I data with json file 
        # train_dataset = load_dataset("json", data_files='/fsx/sfr/data/jiuhai/hub/datasets--JourneyDB--JourneyDB/snapshots/e191aa61ca37e5e4418707ade4df5deb5c6d5d8f/data/train/train_caption_only.jsonl', split="train", num_proc=64)
        # if args.journeyDB_folder is not None:
        #     train_dataset = load_dataset("json", data_files=os.path.join(args.journeyDB_folder, "data/train/train_caption_only.jsonl"), split="train", num_proc=64)
        #     train_dataset = train_dataset.add_column('type', len(train_dataset) * ['journeyDB_T2I'])
        #     train_dataset = train_dataset.add_column('image', len(train_dataset) * [None])
        #     train_dataset = train_dataset.rename_column("caption", "txt")
        #     train_dataset = train_dataset.rename_column("img_path", "image_path")
        #     train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if not col in (
        #         ["txt", "image", "type", "image_path"])])
        #     print(f"finish loading journeyDB {len(train_dataset)}")
        


        ###################################### text to image ####################################### 
        data_files = glob.glob(os.path.join(self.data_args.image_folder, "*.tar"))
        ## text to image
        train_dataset = load_dataset("webdataset", data_files=data_files, split="train", num_proc=128)
        train_dataset = train_dataset.rename_column("jpg", "image")
        train_dataset = train_dataset.add_column('type', len(train_dataset) * ['T2I'])
        train_dataset = train_dataset.add_column('image_path', len(train_dataset) * [None])
        train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if not col in (
            ["image", "txt", "type", "image_path"])])
        print(f"finish loading image {len(train_dataset)}")
        list_data_dict.append(train_dataset)
            

        if len(list_data_dict) > 1:
            list_data_dict = concatenate_datasets(list_data_dict)
        else:
            list_data_dict = list_data_dict[0]
        list_data_dict = list_data_dict.shuffle(seed=42)

        rank0_print(f"Totoal number of training instance: {len(list_data_dict)}")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        while True:
            sources = self.list_data_dict[i]

            if sources["type"] == "T2I" or sources["type"] == "journeyDB_T2I":
                sources["conversations"] = [
                    {"from": "human", "value": f"Please generate image based on the following caption: {sources['txt']}"},
                    {"from": "gpt", "value": "<image>"},
                ]


            elif sources["type"] == "I2I" or sources["type"] == "journeyDB_I2I":
                sources["conversations"] = [
                    {
                        "from": "human",
                        "value": f"<image>\nPlease reconstruct the given image.",
                    },
                    {"from": "gpt", "value": ""},
                ]

            else:
                raise ValueError("Unknown source type. Please check the 'type' in 'sources'.")

            if "image" in sources:

                def img_process(images, processor, image_aspect_ratio):
                    if image_aspect_ratio == "pad":

                        def expand2square(pil_img, background_color):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result

                        images = [expand2square(img, tuple(int(x * 255) for x in processor.image_mean)) for img in images]
                        images = processor.preprocess(images, return_tensors="pt")["pixel_values"]
                    else:
                        images = processor.preprocess(images, return_tensors="pt")["pixel_values"]
                    return images

                if sources["type"] == "T2I" or sources["type"] == "I2I":
                    image_files = self.list_data_dict[i]["image"]
                else:
                    image_files = self.list_data_dict[i]["image_path"]

                if not isinstance(image_files, list):
                    image_files = [image_files]

                images = []

                def read_bin_as_bytesio(bin_file_path):
                    with open(bin_file_path, "rb") as f:
                        return io.BytesIO(f.read())

                for img in image_files:
                    try:
                        if sources["type"] == "T2I" or sources["type"] == "I2I":
                            img = img.convert("RGB")
                        elif sources["type"] == "journeyDB_T2I" or sources["type"] == "journeyDB_I2I":
                            if sources["type"] == "journeyDB_T2I" or sources["type"] == "journeyDB_I2I":
                                image_path = os.path.join(args.journeyDB_folder, "data", "train", "imgs", img)
                                # image_path = os.path.join('/fsx/sfr/data/jiuhai/hub/datasets--JourneyDB--JourneyDB/snapshots/e191aa61ca37e5e4418707ade4df5deb5c6d5d8f/data/train/imgs', img)
                            else:
                                raise ValueError("Unknown source type. Please check the 'type' in 'sources'.")
                            img = Image.open(image_path).convert("RGB")
                        images.append(img)
                    except Exception as e:
                        print(f"Error opening image {img}: {e}")
                        images = None
                        break  # Skip to the next image if there's an error

                if not images is None:
                    try:
                        temp = img_process(
                            images,
                            self.data_args.gen_image_processor,
                            self.data_args.image_aspect_ratio,
                        )
                    except Exception as e:
                        print(f"Error wrong number of channels: {e}")
                        images = None


                # If no valid images were found, randomly pick another item
                if images is None:
                    print(sources)
                    print(f"warning false image!!!!!!")
                    i = random.randint(0, len(self.list_data_dict) - 1)
                    continue


                sources, inst_type = preprocess_multimodal(copy.deepcopy([sources["conversations"]]), self.data_args)
            else:
                sources = copy.deepcopy([sources["conversations"]])
            data_dict = preprocess(sources, self.tokenizer, has_image=("image" in self.list_data_dict[i]))
            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

            # image exist in the data
            if "image" in self.list_data_dict[i]:
                if inst_type == "gen":
                    data_dict["gen_image"] = img_process(
                        images,
                        self.data_args.gen_image_processor,
                        self.data_args.image_aspect_ratio,
                    )

                elif inst_type == "und":

                    resized_images = [transform_und_images(img) for img in images]

                    image_inputs = self.data_args.image_processor(resized_images, return_tensors="pt")

                    data_dict["und_image"] = image_inputs.pixel_values
                    data_dict["grid_thw"] = image_inputs.image_grid_thw
                    data_dict["gen_image"] = img_process(
                        resized_images,
                        self.data_args.gen_image_processor,
                        self.data_args.image_aspect_ratio,
                    )

            elif self.data_args.is_multimodal:
                crop_size = self.data_args.image_processor.crop_size
                data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])

            data_dict["ids"] = self.list_data_dict[i]["id"] if "id" in self.list_data_dict[i] else "unk"
            return data_dict

def _img_process_eval(images: List[Image.Image], processor, image_aspect_ratio: str):
    if image_aspect_ratio == "pad":
        raise NotImplementedError("pad aspect not used in this eval")
    return processor.preprocess(images, return_tensors="pt")["pixel_values"]

class CocoT2IDataset(Dataset):
    def __init__(self, hf_split, gen_image_processor, tokenizer, data_args: DataArguments, image_aspect_ratio="square"):
        self.ds = hf_split
        self.proc = gen_image_processor
        self.tok = tokenizer
        self.data_args = data_args
        self.aspect = image_aspect_ratio
        self.data_args.is_multimodal = True
    def __len__(self): return len(self.ds)
    def __getitem__(self, i: int) -> Dict[str, Any]:
        item = self.ds[i]
        with Image.open(item["image_path"]) as im:
            pil = im.convert("RGB")
        gen_px = _img_process_eval([pil], self.proc, self.aspect)
        txt = item["txt"]
        conv = [
            {"from": "human", "value": f"Please generate image based on the following caption: {txt}"},
            {"from": "gpt",   "value": "<image>"},
        ]
        sources, _ = preprocess_multimodal([conv], self.data_args)
        tokd = preprocess(sources, self.tok, has_image=True)
        return {
            "input_ids": tokd["input_ids"][0],
            "labels":    tokd["labels"][0],
            "gen_image": gen_px,
            "ids": int(item["id"]),
            "image_path": item["image_path"],
        }

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, n_query=64, tokenizer: transformers.PreTrainedTokenizer = None):
        self.n_query_and_index = n_query+1
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "ids"))
        multi_input_ids = []
        multi_labels = []
        i_s_pos = []
        for input_id, label in zip(input_ids, labels):
            input_id = input_id[: self.tokenizer.model_max_length - self.n_query_and_index]
            label = label[: self.tokenizer.model_max_length - self.n_query_and_index]
            i_s_pos.append(input_id.shape[0]+1)
            img_id = torch.full((self.n_query_and_index,), IMAGE_TOKEN_IDX, dtype=input_id.dtype, device=input_id.device)
            img_id[0] = 151665
            input_id = torch.cat([input_id, img_id])
            img_label = torch.full((self.n_query_and_index,), IMAGE_TOKEN_IDX, dtype=label.dtype, device=label.device)
            img_label[0] = 151665
            label = torch.cat([label, img_label])
            multi_input_ids.append(input_id)
            multi_labels.append(label)

        input_ids = multi_input_ids
        labels = multi_labels

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        if input_ids.shape[1] > self.tokenizer.model_max_length:
            print(f"Warning input with length {input_ids.shape[1]} is longer than max length {self.tokenizer.model_max_length}")
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        batch_gen_images = []
        batch_und_images = []
        batch_grid_thw = []

        for instance in instances:
            if "gen_image" in instance:
                batch_gen_images.append(instance["gen_image"])


        if len(batch_gen_images) > 0:
            if all(x is not None and y.shape == batch_gen_images[0][0].shape for x in batch_gen_images for y in x):
                batch["gen_image"] = torch.cat([images for images in batch_gen_images], dim=0)
            else:
                batch["gen_image"] = batch_gen_images
        else:
            batch["gen_image"] = None


        for instance in instances:
            if "und_image" in instance:
                batch_und_images.append(instance["und_image"].unsqueeze(0))  ## 1*1024*1176
                batch_grid_thw.append(instance["grid_thw"])  ## 1*3


        # print(f"batch_und_images {batch_und_images}")
        if len(batch_und_images) > 0:
            batch["und_image"] = torch.cat([images for images in batch_und_images], dim=0)
            batch["grid_thw"] = torch.cat([images for images in batch_grid_thw], dim=0)
        else:
            batch["und_image"] = None
            batch["grid_thw"] = None

        batch["ids"] = ids

        batch["i_s_pos"] = i_s_pos

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:

    if data_args.data_type == "mix":
        train_dataset = LazySupervisedMixDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    else:
        raise ValueError("Unknown data type. Please check the Dataloader type.")

    eval_dataset = None


    image_retrieval_eval_dataset = None
    if data_args.eval_coco_root:
        hf = load_coco_as_hfds(data_args.eval_coco_root, data_args.eval_coco_split).with_format("python")
        image_retrieval_eval_dataset = CocoT2IDataset(
            hf_split=hf,
            gen_image_processor=data_args.gen_image_processor,
            tokenizer=tokenizer,
            data_args=data_args,
            image_aspect_ratio=data_args.image_aspect_ratio,
        )

    mapper_eval_dataset = None
    if data_args.eval_mapper_image_folder:
        da_map = copy.deepcopy(data_args)
        da_map.image_folder = data_args.eval_mapper_image_folder
        mapper_eval_dataset = LazySupervisedMixDataset(tokenizer=tokenizer, data_path=None, data_args=da_map)
        if data_args.eval_mapper_num_samples:
            n = min(int(data_args.eval_mapper_num_samples), len(mapper_eval_dataset))
            mapper_eval_dataset = torch.utils.data.Subset(mapper_eval_dataset, list(range(n)))

    eval_dataset = torch.utils.data.Subset(mapper_eval_dataset, list(range(16)))

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, n_query=data_args.n_query)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator, mapper_eval_dataset=mapper_eval_dataset, image_retrieval_eval_dataset=image_retrieval_eval_dataset)


def unlock_vit(training_args, model_args, vision_tower):
    for n, p in vision_tower.named_parameters():
        p.requires_grad = True


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args, data_args, training_args)
    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )
        
    ## if there exists vision tower for image understanind, we will load LLaMA LLM, otherwise will load Qwen-VL
    if model_args.vision_tower is not None:
        model = blip3oLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args,
        )
    else:
        model = blip3oQwenForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args,
        )

    model.config.use_cache = False

    if model_args.freeze_backbone:
        for (n, p) in model.get_model().named_parameters():
            p.requires_grad = False
        for (n, p) in model.visual.named_parameters():
            p.requires_grad = False
        for (n, p) in model.lm_head.named_parameters():
            p.requires_grad = False
    
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    try:
        tokenizer = AutoProcessor.from_pretrained(model_args.model_name_or_path).tokenizer
    except Exception as e:
        tokenizer = AutoProcessor.from_pretrained(model_args.model_name_or_path)
        
    tokenizer.model_max_length = training_args.model_max_length

    # tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(
                pad_token="<pad>",
                additional_special_tokens=["[IMG]", "[/IMG]", "<image>"],
            ),
            tokenizer=tokenizer,
            model=model,
        )
    elif not "<image>" in tokenizer.get_added_vocab():
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(additional_special_tokens=["[IMG]", "[/IMG]", "<image>"]),
            tokenizer=tokenizer,
            model=model,
        )
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["llama3"]
    rank0_print(f"Using conversation format: {conversation_lib.default_conversation.version}")



    # if model_args.vision_tower is not None:
    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
    model.predict_summary_token = model_args.predict_summary_token
    model.predict_dino_grid = model_args.predict_dino_grid
    model.config.num_register_tokens = model_args.num_register_tokens

    ## generation vision tower
    gen_vision_tower = model.get_gen_vision_tower()
    gen_vision_tower.to(
        dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        device=training_args.device,
    )
    gen_vision_tower.requires_grad_(False)

    data_args.gen_image_processor = gen_vision_tower.image_processor
    data_args.image_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct").image_processor

    data_args.is_multimodal = True
    data_args.n_query = model_args.n_query
    data_args.n_und_query = model_args.n_und_query

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter

    # Calculate total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.get_model().parameters())
    trainable_params = sum(p.numel() for p in model.get_model().parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")


    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    mapper_eval_dataset = data_module.pop("mapper_eval_dataset", None)
    image_retrieval_eval_dataset = data_module.pop("image_retrieval_eval_dataset", None)

    trainer = blip3oTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )
    from tabulate import tabulate

    if trainer.is_world_process_zero():
        stat = []
        for i, (n, p) in enumerate(trainer.model.named_parameters()):
            stat.append([i, n, p.shape, p.requires_grad, p.dtype])
        print(tabulate(stat, headers=["idx", "name", "shape", "trainable", "dtype"]))

    cb = RetrievalEvalCallback(
    trainer,
    model_args=model_args,
    data_args=data_args,
    dataset=image_retrieval_eval_dataset,
    use_memmap=getattr(training_args, "eval_use_memmap", True),
    mmap_dir=getattr(training_args, "eval_mmap_dir", os.environ.get("KNN_MMAP_DIR", ".")),
    recompute_db_cache=getattr(training_args, "eval_recompute_db_cache", False),
    cache_dir=getattr(training_args, "eval_cache_dir", os.environ.get("ENCODER_CACHE_DIR", ".encoder_cache")),
    q_chunk=getattr(training_args, "eval_q_chunk", 4096),
    db_chunk=getattr(training_args, "eval_db_chunk", 65536),
    image_aspect_ratio=getattr(training_args, "eval_image_aspect_ratio", "square"),
    )

    
    cb2 = MapperEvalCallback(trainer, data_args=data_args, dataset=mapper_eval_dataset)
    trainer.add_callback(cb2)
    # trainer.add_callback(cb)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
        vision_tower=model_args.vision_tower,
    )


class RetrievalEvalCallback(TrainerCallback):
    """
    Runs COCO Qwen-DiT→Vision Encoder retrieval at eval points
    Uses trainer.eval_dataset + trainer.data_collator.
    """

    def __init__(
        self,
        trainer,        
        model_args,
        data_args,
        dataset: Dataset = None,
        use_memmap: bool = True,
        mmap_dir: str = None,
        recompute_db_cache: bool = False,
        cache_dir: str = None,
        q_chunk: int = 4096,
        db_chunk: int = 65536,
        image_aspect_ratio: str = "square",
    ):
        self.trainer = trainer
        self.model_args = model_args
        self.data_args  = data_args
        self.use_memmap = use_memmap
        self.mmap_dir = mmap_dir or os.environ.get("KNN_MMAP_DIR", ".")
        self.recompute_db_cache = recompute_db_cache
        self.cache_dir = cache_dir or os.environ.get("ENCODER_CACHE_DIR", ".encoder_cache")
        self.q_chunk = q_chunk
        self.db_chunk = db_chunk
        self.image_aspect_ratio = image_aspect_ratio
        self.dataset = dataset

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Skip if no eval set
        eval_ds = self.dataset
        if eval_ds is None:
            return

        self.trainer.model.eval()

        is_dist = dist.is_available() and dist.is_initialized()
        world   = dist.get_world_size() if is_dist else 1
        rank    = dist.get_rank() if is_dist else 0
        device  = args.device

        # Rebuild a fresh distributed eval loader
        sampler = DistributedSampler(
            eval_ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False
        ) if is_dist else None

        eval_loader = DataLoader(
            eval_ds,
            batch_size=args.per_device_eval_batch_size,
            sampler=sampler,
            shuffle=False,
            collate_fn=self.trainer.data_collator,
            num_workers=getattr(args, "eval_workers", 8),
            pin_memory=True,
        )

        if getattr(args, "bf16", False):
            float_dtype = torch.bfloat16
        elif getattr(args, "fp16", False):
            float_dtype = torch.float16
        else:
            float_dtype = torch.float32

        # ----- DB cache/memmap keys (per-rank when memmap; shared for npy) -----
        cache_tag_base = {
            "split": getattr(args, "eval_coco_split", "val2017"),
            "aspect": self.image_aspect_ratio,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "rank": int(os.environ.get("RANK", "0")),
            "model": self.model_args.model_name_or_path,
        }

        have_db_mm = memmaps_exist("db", world, self.mmap_dir)

        # ----- Compute or Load DB (vision) latents -----
        db_vec_local = db_sum_local = db_ids_local = None

        if (not self.use_memmap) and (not self.recompute_db_cache):
            p_db = _cache_prefix("coco_db", cache_tag_base)
            db_vec_local, db_sum_local, db_ids_local = try_load_cached(self.cache_dir, p_db)
            if db_vec_local is not None and rank == 0:
                print(f"[cache] Loaded DB latents from {self.cache_dir}")

        if self.use_memmap and (not self.recompute_db_cache) and have_db_mm:
            if is_dist: dist.barrier()
            if rank == 0:
                print("[memmap] Found existing DB memmaps. Reusing.")
        elif db_vec_local is None:
            gt_vecs, gt_summ, gt_ids = [], [], []
            for batch in eval_loader:
                lat, summary = batch_ground_truth_latents(self.trainer.model, batch, device, float_dtype)
                pooled = lat.mean(dim=(2,3)).float() if lat.dim()==4 else lat.float()
                gt_vecs.append(torch.nn.functional.normalize(pooled, p=2, dim=1).to(torch.float16).cpu())
                gt_summ.append(torch.nn.functional.normalize(summary.float(), p=2, dim=1).to(torch.float16).cpu())
                gt_ids.extend(batch["ids"])
                del lat, summary, pooled
            db_vec_local = torch.cat(gt_vecs, 0)
            db_sum_local = torch.cat(gt_summ, 0)
            db_ids_local = torch.tensor(gt_ids, dtype=torch.long)

            if self.use_memmap:
                write_memmap("db", db_vec_local, db_sum_local, db_ids_local, rank, self.mmap_dir)
                if is_dist: dist.barrier()  # ensure all shards flushed
            else:
                save_cache(self.cache_dir, _cache_prefix("coco_db", cache_tag_base), db_vec_local, db_sum_local, db_ids_local)

        if is_dist: dist.barrier()

        # ----- Queries: always recompute -----
        sampler_q = DistributedSampler(eval_ds, world, rank, shuffle=False, drop_last=False) if is_dist else None
        eval_loader_q = DataLoader(
            eval_ds,
            batch_size=args.per_device_eval_batch_size,
            sampler=sampler_q,
            shuffle=False,
            collate_fn=self.trainer.data_collator,
            num_workers=getattr(args, "eval_workers", 8),
            pin_memory=True,
        )
        n_query = self.model_args.n_query
        q_vec_local, q_sum_local, q_ids_local = per_image_text_predicted_latents_via_loader(
            self.trainer.model, eval_loader_q, device=device, n_query=n_query,
        )

        # ----- Metrics (in-memory vs memmap) -----
        def _gather_np_to_root(t: torch.Tensor):
            if world == 1:
                return [t] if rank == 0 else None
            obj = t.numpy()
            if rank == 0:
                bucket = [None for _ in range(world)]
                dist.gather_object(obj, object_gather_list=bucket, dst=0)
                return [torch.from_numpy(x.copy()) for x in bucket]
            else:
                dist.gather_object(obj, dst=0)
                return None

        if not self.use_memmap:
            g_db_vec = _gather_np_to_root(db_vec_local)
            g_db_sum = _gather_np_to_root(db_sum_local)
            g_db_ids = _gather_np_to_root(db_ids_local)
            g_q_vec  = _gather_np_to_root(q_vec_local)
            g_q_sum  = _gather_np_to_root(q_sum_local)
            g_q_ids  = _gather_np_to_root(q_ids_local)
            if rank != 0:
                return

            db_vec = torch.cat(g_db_vec, 0); db_sum = torch.cat(g_db_sum, 0); db_ids = torch.cat(g_db_ids, 0)
            q_vec  = torch.cat(g_q_vec,  0); q_sum  = torch.cat(g_q_sum,  0); q_ids  = torch.cat(g_q_ids,  0)

            rec_pool = retrieval_recall_at_k(q_vec, db_vec, q_ids, db_ids, device=device,
                                             ks=(1,5,10), q_bs=self.q_chunk, db_bs=self.db_chunk)
            rec_sum  = retrieval_recall_at_k(q_sum, db_sum, q_ids, db_ids, device=device,
                                             ks=(1,5,10), q_bs=self.q_chunk, db_bs=self.db_chunk)

            print("\n=== COCO Text→Image Retrieval (IN-MEMORY, DB cached only) ===")
            print("[POOLED]  R@1: {:.2f}% | R@5: {:.2f}% | R@10: {:.2f}%".format(rec_pool["R@1"], rec_pool["R@5"], rec_pool["R@10"]))
            print("[SUMMARY] R@1: {:.2f}% | R@5: {:.2f}% | R@10: {:.2f}%".format(rec_sum["R@1"],  rec_sum["R@5"],  rec_sum["R@10"]))

            # Log to Trainer
            metrics = {**{f"eval/retrieval/pooled_{k}": v for k, v in rec_pool.items()},
                       **{f"eval/retrieval/summary_{k}": v for k, v in rec_sum.items()}}
            self.trainer.log(metrics)
            return

        # memmap path: gather queries only to rank 0, open DB shards there
        g_q_vec = _gather_np_to_root(q_vec_local)
        g_q_sum = _gather_np_to_root(q_sum_local)
        g_q_ids = _gather_np_to_root(q_ids_local)
        if rank != 0:
            return
        q_vec = torch.cat(g_q_vec, 0)
        q_sum = torch.cat(g_q_sum, 0)
        q_ids = torch.cat(g_q_ids, 0)

        db_vec_shards, db_sum_shards, db_id_shards, Dp, Ds = open_memmap_shards("db", world, self.mmap_dir)
        rec_pool = retrieval_q_vs_db_shards(q_vec, q_ids, db_vec_shards, db_id_shards, device=device,
                                            ks=(1,5,10), q_bs=self.q_chunk, db_bs=self.db_chunk)
        rec_sum  = retrieval_q_vs_db_shards(q_sum, q_ids, db_sum_shards, db_id_shards, device=device,
                                            ks=(1,5,10), q_bs=self.q_chunk, db_bs=self.db_chunk)

        print("\n=== COCO Text→Image Retrieval (MEMMAP: DB cached only) ===")
        print("[POOLED]  R@1: {:.2f}% | R@5: {:.2f}% | R@10: {:.2f}%".format(rec_pool["R@1"], rec_pool["R@5"], rec_pool["R@10"]))
        print("[SUMMARY] R@1: {:.2f}% | R@5: {:.2f}% | R@10: {:.2f}%".format(rec_sum["R@1"],  rec_sum["R@5"],  rec_sum["R@10"]))

        self.trainer.log({**{f"eval/retrieval/pooled_{k}": v for k, v in rec_pool.items()},
                          **{f"eval/retrieval/summary_{k}": v for k, v in rec_sum.items()}})
        
        self.trainer.model.train()

       
class MapperEvalCallback(TrainerCallback):
    def __init__(self, trainer, data_args, dataset):
        self.trainer = trainer
        self.data_args = data_args
        self.dataset = dataset

    def on_evaluate(self, args, state, control, **kwargs):
        if self.dataset is None:
            return
        is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        world = torch.distributed.get_world_size() if is_dist else 1
        rank  = torch.distributed.get_rank() if is_dist else 0

        # shard
        sampler = DistributedSampler(self.dataset, num_replicas=world, rank=rank, shuffle=False) if is_dist else None

        # optional global cap -> local cap
        N_total = self.data_args.eval_mapper_num_samples
        max_local = None if (N_total is None or not is_dist) else max(1, (N_total + world - 1)//world)

        dl = DataLoader(
            self.dataset,
            batch_size=getattr(args, "eval_mapper_batch_size", 4),
            sampler=sampler,
            shuffle=False,
            num_workers=getattr(args, "eval_mapper_workers", 8),
            pin_memory=True,
            collate_fn=self.trainer.data_collator,
        )

        with torch.no_grad():
            metrics_local = self.trainer.model.evaluate_mapper_dino(
                dl,
                device=args.device,
                num_inference_steps=getattr(args, "eval_mapper_steps", 30),
                k_neigh=getattr(args, "eval_mapper_k_neigh", 10),
                max_eval_samples=max_local,
            )

        # ---- DDP reduction: weighted mean by image count ----
        n_local = float(metrics_local.get("n_images", 0))
        # use a deterministic order across ranks
        avg_keys = sorted(k for k in metrics_local.keys() if k != "n_images")

        # convert to tensors on the right device/dtype
        vals = torch.tensor([float(metrics_local[k]) for k in avg_keys],
                            device=args.device, dtype=torch.float64)
        cnt  = torch.tensor([n_local], device=args.device, dtype=torch.float64)

        # multiply local means by local count -> local sums
        vals = vals * cnt

        if is_dist:
            torch.distributed.all_reduce(vals, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(cnt,  op=torch.distributed.ReduceOp.SUM)

        n_global = float(cnt.item())
        if n_global > 0:
            vals = vals / n_global  # back to global means
            metrics_global = {k: float(vals[i].item()) for i, k in enumerate(avg_keys)}
            metrics_global["n_images"] = n_global
        else:
            metrics_global = dict(metrics_local)


        if (not is_dist) or (rank == 0):
            # self.trainer.log({f"eval/mapper/{k}": v for k, v in metrics_global.items()})

            to_log = {f"mapper/{k}": v for k, v in metrics_global.items()}
            self.trainer.log_metrics("eval", to_log)   # shows as eval_* in logs/W&B
            self.trainer.save_metrics("eval", to_log)  # optional: also dumps to metrics.json
            self.trainer.log({f"eval/mapper/{k}": v for k, v in metrics_global.items()})


if __name__ == "__main__":
    train()

