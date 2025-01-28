#encoding: utf-8

# <|begin_of_text|>: 128000
# <|start_header_id|>: 128006
# <|end_header_id|>: 128007
# <|eot_id|>: 128009
# \n: 198

# instruction template: "<|start_header_id|>system<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n%s<|eot_id|>" % (system, user, assistant,)
# <|begin_of_text|> was inserted to the beginning automatically by tokenizer.encode
# \n may be used for pure language modeling with template: "%s\n" % user

# pad_id: <|finetune_right_pad_id|>, sos_id: <|begin_of_text|>, eos_id: <|eot_id|>
pad_id, sos_id, eos_id, unk_id = 128000, 128000, 128009, None
# soh_id: <|start_header_id|>, eoh_id: <|end_header_id|>
soh_id, eoh_id = 128006, 128007
vocab_size = 128256# 128256 for 1B

instruct_template = lambda system, user: "<|start_header_id|>system<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" % (system, user)
instruct_lm_template = lambda system, user, assistant: "<|start_header_id|>system<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n%s<|eot_id|>" % (system, user, assistant,)
assistant_template = lambda system, assistant: "%s<|eot_id|>" % assistant
lm_template = lambda system, user: "%s\n" % user
templated = {"instruct": instruct_template, "instruct_lm": instruct_lm_template, "assistant": assistant_template, "lm": lm_template}
