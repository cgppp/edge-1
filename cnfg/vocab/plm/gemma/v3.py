#encoding: utf-8

# <pad>: 0
# <eos>: 1
# <bos>: 2
# <unk>: 3
# <mask>: 4
# [multimodal]: 5
# <unused0-98>: 6-104
# <start_of_turn>: 105
# <end_of_turn>: 106
# \n: 107
# <start_of_image>: 255999
# <end_of_image>: 256000
# <unused99-6241>: 256001-262143
# <image_soft_token>: 262144

# instruction template: "<bos><start_of_turn>user\n%s\n\n%s<end_of_turn>\n<start_of_turn>model\n%s<end_of_turn>" % (system, user, assistant,)
# <bos> was inserted to the beginning automatically by tokenizer.encode
# \n may be used for pure language modeling with template: "%s\n" % user

# pad_id: <pad>, sos_id: <bos>, eos_id: <end_of_turn>
pad_id, sos_id, eos_id, unk_id = 0, 2, 106, 3
vocab_size = 262144

instruct_template = lambda system, user: "<bos><start_of_turn>user\n%s\n\n%s<end_of_turn>\n<start_of_turn>model\n" % (system, user)
instruct_lm_template = lambda system, user, assistant: "<bos><start_of_turn>user\n%s\n\n%s<end_of_turn>\n<start_of_turn>model\n%s<end_of_turn>" % (system, user, assistant,)
assistant_template = lambda system, assistant: "%s<end_of_turn>" % assistant
lm_template = lambda system, user: "%s\n" % user
instruct_task_template = lambda system, user: instruct_template(system, "Translate the input into German, do not generate the other contents apart from the translation. Input: %s" % user)
templated = {"instruct": instruct_template, "instruct_lm": instruct_lm_template, "assistant": assistant_template, "lm": lm_template, "instruct_task": instruct_task_template}
