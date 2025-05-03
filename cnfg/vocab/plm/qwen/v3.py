#encoding: utf-8

# <|endoftext|>: 151643
# <|im_start|>: 151644
# <|im_end|>: 151645
# <|object_ref_start|>: 151646
# <|object_ref_end|>: 151647
# <|box_start|>: 151648
# <|box_end|>: 151649
# <|quad_start|>: 151650
# <|quad_end|>: 151651
# <|vision_start|>: 151652
# <|vision_end|>: 151653
# <|vision_pad|>: 151654
# <|image_pad|>: 151655
# <|video_pad|>: 151656
# <tool_call>: 151657
# </tool_call>: 151658
# <|fim_prefix|>: 151659
# <|fim_middle|>: 151660
# <|fim_suffix|>: 151661
# <|fim_pad|>: 151662
# <|repo_name|>: 151663
# <|file_sep|>: 151664
# <tool_response>: 151665
# </tool_response>: 151666
# <think>: 151667
# </think>: 151668

# instruction template: "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n%s\n</think>\n\n%s<|im_end|>" % (system, user, think, assistant,)
# pure language modeling do not have either <sos> or <eos> tokens with template: "%s" % user

# pad_id: using <|endoftext|> for padding, sos_id: <|im_start|>, eos_id: <|im_end|>
pad_id, sos_id, eos_id, unk_id = 151643, 151644, 151645, None
sot_id, eot_id = 151667, 151668
vocab_size = 151936# 151936 for 0.6/1.7/4/8/14B

instruct_template = lambda system, user: ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" % (system, user,)) if system else ("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" % user)
instruct_lm_template = lambda system, user, assistant: ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n%s<|im_end|>" % (system, user, assistant,)) if system else ("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n%s<|im_end|>" % (user, assistant,))
instruct_auto_template = lambda system, user: ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" % (system, user,)) if system else ("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" % user)
instruct_think_template = lambda system, user, think: ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n%s\n</think>\n\n" % (system, user, think,)) if system else ("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n%s\n</think>\n\n" % (user, think,))
instruct_think_lm_template = lambda system, user, assistant: ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n%s\n</think>\n\n%s<|im_end|>" % (system, user, think, assistant,)) if system else ("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n%s\n</think>\n\n%s<|im_end|>" % (user, think, assistant,))
assistant_template = lambda system, assistant: "%s<|im_end|>" % assistant
lm_template = lambda system, user: user
templated = {"instruct": instruct_template, "instruct_lm": instruct_lm_template, "assistant": assistant_template, "lm": lm_template, "instruct_auto": instruct_auto_template, "instruct_think": instruct_think_template, "instruct_think_lm": instruct_think_lm_template}
