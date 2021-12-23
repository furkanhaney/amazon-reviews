import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="processed/raw0.txt",
    model_type="bpe",
    model_prefix="processed/m",
    vocab_size=2048,
    normalization_rule_name="nmt_nfkc_cf",
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=3,
)
