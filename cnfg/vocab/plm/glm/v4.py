#encoding: utf-8

# [gMASK]: 151331
# <sop>: 151333
# <|system|>: 151335
# <|user|>: 151336
# <|assistant|>: 151337
# <|endoftext|>: 151329
# [MASK]: 151330
# [sMASK]: 151332
# <eop>: 151334
# <|observation|>: 151338
# <|begin_of_image|>: 151339
# <|end_of_image|>: 151340
# <|begin_of_video|>: 151341
# <|end_of_video|>: 151342
# \n: 198

# instruction template: "[gMASK]<sop><|system|>\n%s<|user|>\n%s<|assistant|>\n%s<|user|>" % (system, user, assistant,)
# \n may be used for pure language modeling with template: "[gMASK]<sop>%s\n" % user

# pad_id: <|endoftext|>, sos_id: <sop>, eos_id: <|user|>
pad_id, sos_id, eos_id, unk_id = 151329, 151333, 151336, None
gmask_id, sop_id = 151331, 151333
vocab_size = 151552# 151552 for 9B

instruct_template = lambda system, user: "<|system|>\n%s<|user|>\n%s<|assistant|>\n" % (system, user,)
instruct_lm_template = lambda system, user, assistant: "<|system|>\n%s<|user|>\n%s<|assistant|>\n%s<|user|>" % (system, user, assistant,)
assistant_template = lambda system, assistant: "%s<|user|>" % assistant
lm_template = lambda system, user: "%s\n" % user
instruct_task_template = lambda system, user: instruct_template(system, "Translate the input into German, do not generate the other contents apart from the translation. Input: %s" % user)
templated = {"instruct": instruct_template, "instruct_lm": instruct_lm_template, "assistant": assistant_template, "lm": lm_template, "instruct_task": instruct_task_template}
