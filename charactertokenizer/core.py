""" CharacterTokenzier for Hugging Face Transformers.

This is heavily inspired from CanineTokenizer in transformers package.

19.08.2023 Доработал для использования в проектах char-level GPT и LLaMa - токены <pad>, <s> etc
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer


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


    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        cx = list(text.replace('<pad>', '\x00').replace('<s>', '\x02').replace('</s>', '\x03'))
        return [self.ascii_2_token.get(c, c) for c in cx]

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["<unk>"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    # def build_inputs_with_special_tokens(
    #     self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    # ) -> List[int]:
    #     sep = [self.sep_token_id]
    #     cls = [self.cls_token_id]
    #     result = cls + token_ids_0 + sep
    #     if token_ids_1 is not None:
    #         result += token_ids_1 + sep
    #     return result

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

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        result = len(cls + token_ids_0 + sep) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    def get_config(self) -> Dict:
        return {
            "name": "CharacterTokenizer",
            "char_ords": [ord(ch) for ch in self.characters],
            "model_max_length": self.model_max_length,
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
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file) as f:
            cfg = json.load(f)
        return cls.from_config(cfg)
