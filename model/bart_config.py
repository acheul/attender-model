from transformers import BartConfig

d = {
  "activation_dropout": 0.0,
  "activation_function": "gelu",
  "add_bias_logits": False,
  "add_final_layer_norm": False,
  "architectures": [
    "BartModel"
  ],
  "attention_dropout": 0.0,
  "author": "Heewon Jeon(madjakarta@gmail.com)",
  "bos_token_id": 0,
  "classif_dropout": 0.1,
  "classifier_dropout": 0.1,
  "d_model": 768,
  "decoder_attention_heads": 16,
  "decoder_ffn_dim": 3072,
  "decoder_layerdrop": 0.0,
  "decoder_layers": 6,
  "decoder_start_token_id": 1,
  "do_blenderbot_90_layernorm": False,
  "dropout": 0.1,
  "encoder_attention_heads": 16,
  "encoder_ffn_dim": 3072,
  "encoder_layerdrop": 0.0,
  "encoder_layers": 6,
  "eos_token_id": 1,
  "extra_pos_embeddings": 2,
  "force_bos_token_to_be_generated": False,
  "forced_eos_token_id": 1,
  "gradient_checkpointing": False,
  "id2label": {
    "0": "NEGATIVE",
    "1": "POSITIVE"
  },
  "init_std": 0.02,
  "is_encoder_decoder": True,
  "kobart_version": 1.0,
  "label2id": {
    "NEGATIVE": 0,
    "POSITIVE": 1
  },
  "max_position_embeddings": 1026,
  "model_type": "bart",
  "normalize_before": False,
  "normalize_embedding": True,
  "num_hidden_layers": 6,
  "pad_token_id": 3,
  "scale_embedding": False,
  "static_position_embeddings": False,
  "tokenizer_class": "PreTrainedTokenizerFast",
  "transformers_version": "4.21.2",
  "use_cache": True,
  "vocab_size": 30000
}
bart_config = BartConfig.from_dict(d)