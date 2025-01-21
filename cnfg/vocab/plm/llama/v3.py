#encoding: utf-8

# <|begin_of_text|>: 128000
# <|start_header_id|>: 128006
# <|end_header_id|>: 128007
# <|eot_id|>: 128009

# template: "<|start_header_id|>system<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n%s<|eot_id|>" % (system, user, assistant,)
# <|begin_of_text|> was inserted to the beginning automatically by tokenizer.encode

# pad_id: <|finetune_right_pad_id|>, sos_id: <|begin_of_text|>, eos_id: <|eot_id|>
pad_id, sos_id, eos_id, unk_id = 128000, 128000, 128009, None
# soh_id: <|start_header_id|>, eoh_id: <|end_header_id|>
soh_id, eoh_id = 128006, 128007
vocab_size = 128256# 128256 for 1B
