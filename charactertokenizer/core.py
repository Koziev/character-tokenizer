""" CharacterTokenzier for Hugging Face Transformers.

This is heavily inspired from CanineTokenizer in transformers package.

19.08.2023 Доработал для использования в проектах char-level GPT и LLaMa - токены <pad>, <s> etc
20.08.2023 Добавил вызов add_special_tokens в from_pretrained
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import (
    cached_file,
    copy_func,
    download_url,
    extract_commit_hash,
    is_remote_url,
)



class CharacterTokenizer(PreTrainedTokenizer):
    def __init__(self, characters: Sequence[str], model_max_length: int, **kwargs):
        """Character tokenizer for Hugging Face transformers.

        Args:
            characters (Sequence[str]): List of desired characters. Any character which
                is not included in this list will be replaced by a special token called
                <unk> with id=3. Following are list of all of the special tokens with
                their corresponding ids:
                    "<pad>": 0
                    "<s>": 1
                    "</s>": 2
                    "<unk>": 3
                an id (starting at 7) will be assigned to each character.

            model_max_length (int): Model maximum sequence length.
        """
        self.characters = characters
        self.model_max_length = model_max_length

        self._vocab_str_to_int = {
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
            "<sep>": 4,
            "<cls>": 5,
            "<mask>": 6,
            **{ch: i for i, ch in enumerate(characters, start=7)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        self.ascii_2_token = {'\x00': '<pad>', '\x02': '<s>', '\x03': '</s>', '\x18': '<unk>'}

        # Heavily using ASCII control codes instead of multichar tokens
        pad_token = AddedToken("<pad>", lstrip=False, rstrip=False)
        bos_token = AddedToken("<s>", lstrip=False, rstrip=False)
        eos_token = AddedToken("</s>", lstrip=False, rstrip=False)
        unk_token = AddedToken("<unk>", lstrip=False, rstrip=False)
        sep_token = AddedToken("<sep>", lstrip=False, rstrip=False)
        cls_token = AddedToken("<cls>", lstrip=False, rstrip=False)
        mask_token = AddedToken("<mask>", lstrip=True, rstrip=False)
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        #cx = list(text.replace('<pad>', '\x00').replace('<s>', '\x02').replace('</s>', '\x03'))
        #return [self.ascii_2_token.get(c, c) for c in cx]
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["<unk>"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def get_config(self) -> Dict:
        return {
            "name": "CharacterTokenizer",
            "char_ords": [ord(ch) for ch in self.characters],
            "model_max_length": self.model_max_length,
            "size": len(self.characters)
        }

    @classmethod
    def from_config(cls, config: Dict) -> "CharacterTokenizer":
        cfg = {}
        cfg["characters"] = [chr(i) for i in config["char_ords"]]
        cfg["model_max_length"] = config["model_max_length"]
        return cls(**cfg)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        #cfg_file = Path(save_directory) / "tokenizer_config.json"
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        subfolder = kwargs.pop("subfolder", "")
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        commit_hash = kwargs.pop("_commit_hash", None)

        is_local = os.path.isdir(save_directory)
        if os.path.exists(save_directory):
            resolved_config_file = os.path.join(save_directory, "tokenizer_config.json")
            is_local = True
        else:
            configuration_file = "tokenizer_config.json"

            try:
                # Load from local folder or from cache or download from model Hub and cache
                resolved_config_file = cached_file(
                    save_directory,
                    configuration_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    #user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _commit_hash=commit_hash,
                )
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load the configuration of '{save_directory}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the same"
                    f" name. Otherwise, make sure '{save_directory}' is the correct path to a directory"
                    f" containing a {save_directory} file"
                )

        with open(resolved_config_file) as f:
            cfg = json.load(f)
        instance = cls.from_config(cfg)
        instance.add_special_tokens({'pad_token': '<pad>', 'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>'})
        return instance

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int

    def encode_plus(
        self,
        text,
        add_special_tokens=True,
        padding=False,
        truncation=None,
        max_length=None,
        stride=0,
        is_split_into_words=False,
        pad_to_multiple_of=None,
        return_tensors=None,
        return_token_type_ids=False,
        return_attention_mask=None,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
        return_length=False,
        verbose=True,
        **kwargs,
    ):
        return super().encode_plus(text=text,
                                   text_pair=None,
                                   add_special_tokens=add_special_tokens,
                                   padding=padding,
                                   truncation=truncation,
                                   max_length=max_length,
                                   stride=stride,
                                   is_split_into_words=False,
                                   pad_to_multiple_of=pad_to_multiple_of,
                                   return_tensors=return_tensors,
                                   return_token_type_ids=False,
                                   return_attention_mask=return_attention_mask,
                                   return_overflowing_tokens=return_overflowing_tokens,
                                   return_special_tokens_mask=return_special_tokens_mask,
                                   return_offsets_mapping=return_offsets_mapping,
                                   return_length=return_length,
                                   verbose=verbose,
                                   **kwargs
                                   )



class SyllabicTokenizer(PreTrainedTokenizer):
    def __init__(self, model_max_length: int, vocabulary: Sequence[str] = None, vocab_str_to_int=None, **kwargs):
        """Syllable-level tokenizer for Hugging Face transformers."""
        self.model_max_length = model_max_length

        pad_token = AddedToken("<pad>", lstrip=False, rstrip=False)
        bos_token = AddedToken("<s>", lstrip=False, rstrip=False)
        eos_token = AddedToken("</s>", lstrip=False, rstrip=False)
        unk_token = AddedToken("<unk>", lstrip=False, rstrip=False)
        sep_token = AddedToken("<sep>", lstrip=False, rstrip=False)
        cls_token = AddedToken("<cls>", lstrip=False, rstrip=False)
        mask_token = AddedToken("<mask>", lstrip=True, rstrip=False)

        if vocab_str_to_int:
            self._vocab_str_to_int = dict(vocab_str_to_int)
        else:
            self._vocab_str_to_int = {
                "<pad>": 0,
                "<s>": 1,
                "</s>": 2,
                "<unk>": 3,
                "<sep>": 4,
                "<cls>": 5,
                "<mask>": 6,
                **{t: i for i, t in enumerate(vocabulary, start=7)},
            }

        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            **kwargs,
        )


    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        return list(text.split(' '))

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["<unk>"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def get_config(self) -> Dict:
        return {
            "name": "SyllabicTokenizer",
            "vocab_str_to_int": self._vocab_str_to_int,
            "model_max_length": self.model_max_length,
            "size": len(self._vocab_str_to_int)
        }

    @classmethod
    def from_config(cls, config: Dict) -> "CharacterTokenizer":
        cfg = {}
        cfg["vocab_str_to_int"] = config['vocab_str_to_int']
        cfg["model_max_length"] = config["model_max_length"]
        return cls(**cfg)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        #cfg_file = Path(save_directory) / "tokenizer_config.json"
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        subfolder = kwargs.pop("subfolder", "")
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        commit_hash = kwargs.pop("_commit_hash", None)

        is_local = os.path.isdir(save_directory)
        if os.path.exists(save_directory):
            resolved_config_file = os.path.join(save_directory, "tokenizer_config.json")
            is_local = True
        else:
            configuration_file = "tokenizer_config.json"

            try:
                # Load from local folder or from cache or download from model Hub and cache
                resolved_config_file = cached_file(
                    save_directory,
                    configuration_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    #user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _commit_hash=commit_hash,
                )
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load the configuration of '{save_directory}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the same"
                    f" name. Otherwise, make sure '{save_directory}' is the correct path to a directory"
                    f" containing a {save_directory} file"
                )

        with open(resolved_config_file) as f:
            cfg = json.load(f)
        instance = cls.from_config(cfg)
        instance.add_special_tokens({'pad_token': '<pad>', 'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>'})
        return instance

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int
