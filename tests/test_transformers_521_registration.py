"""Regression for mlx-lm's tokenizer registration with current transformers."""
import json


def test_mlx_lm_registered_tokenizer_loads(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoTokenizer
    from transformers.models.auto.tokenization_auto import tokenizer_class_from_name
    from mlx_lm.tokenizer_utils import NewlineTokenizer

    assert tokenizer_class_from_name("NewlineTokenizer") is NewlineTokenizer
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "NewlineTokenizer", "unk_token": "[UNK]"
    }))
    tokenizer = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    assert isinstance(tokenizer, NewlineTokenizer)
    assert tokenizer.get_vocab()["hello"] == 1
