#encoding: utf-8

# <unk> (pad_token_id): 0
# <s>: 1
# </s>: 2
# <reserved_106> (user_token_id): 195
# <reserved_107> (assistant_token_id): 196

# instruction template: "%s<reserved_106>%s<reserved_107>%s</s>" % (system, user, assistant,)
# pure language modeling do not have either <sos> or <eos> tokens with template: "%s" % user

# pad_id: using <unk> for padding, sos_id: <s>, eos_id: </s>
pad_id, sos_id, eos_id, unk_id = 0, 1, 2, 0
vocab_size = 125696

instruct_template = lambda system, user: "%s<reserved_106>%s<reserved_107>" % (system, user,)
instruct_lm_template = lambda system, user, assistant: "%s<reserved_106>%s<reserved_107>%s</s>" % (system, user, assistant,)
assistant_template = lambda system, assistant: "%s</s>" % assistant
lm_template = lambda system, user: user
templated = {"instruct": instruct_template, "instruct_lm": instruct_lm_template, "assistant": assistant_template, "lm": lm_template}
