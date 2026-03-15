#encoding: utf-8
"""
Qwen3 词表与指令模板配置。

本模块提供：
- 特殊 token 的 id（pad、sos、eos、<think>/</think> 等），与 Qwen2/Qwen3 tokenizer 一致。
- 词表大小 vocab_size，供模型 embedding/classifier 维度使用。
- 若干「指令格式」模板函数（lambda），将 system/user/assistant 等拼成 Qwen 对话格式字符串；
  tools/plm/map/qwen/v3.py 等会按模板名（如 instruct_auto、assistant）调用，生成 src/tgt 文本再 tokenize。
"""

# ----- 特殊 token 在词表中的 id（与 Qwen tokenizer 对齐） -----
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

# 训练/推理常用：pad 用 <|endoftext|>，对话开始用 <|im_start|>，结束用 <|im_end|>；BPE 词表无 <unk> 故为 None
pad_id, sos_id, eos_id, unk_id = 151643, 151644, 151645, None
# 思考链：<think> / </think> 的 id，用于 think 类模板
sot_id, eot_id = 151667, 151668

# 词表大小，与 Qwen 0.6/1.7/4/8/14B 一致；embedding 与 classifier 的 vocab 维度依此配置
vocab_size = 151936  # 151936 for 0.6/1.7/4/8/14B

# ----- 指令模板（返回拼接好的字符串，供 map 脚本填内容后 tokenize） -----
# 参数含义：system=系统提示（可选），user=用户输入，assistant=助手回复，think=思考内容（可选）

# 只拼到 assistant 开头、不包含回复内容（用于「输入」侧，模型要生成的是后面的 assistant 部分）
instruct_template = lambda system, user: ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" % (system, user,)) if system else ("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" % user)

# 含完整 assistant 回复，并以 <|im_end|> 结尾（用于已标注的「指令+回答」整段，如做 loss 时目标序列）
instruct_lm_template = lambda system, user, assistant: ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n%s<|im_end|>" % (system, user, assistant,)) if system else ("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n%s<|im_end|>" % (user, assistant,))

# 拼到 assistant 后、不含 <think> 占位与回复内容，结尾无 <|im_end|>（本复现 map 时 src 常用此模板名 instruct_auto）
instruct_auto_template = lambda system, user: ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" % (system, user,)) if system else ("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" % user)

# 含 think 占位，不包含 assistant 回复（思考链输入侧）
instruct_think_template = lambda system, user, think: ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n%s\n</think>\n\n" % (system, user, think,)) if system else ("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n%s\n</think>\n\n" % (user, think,))

# 含 think 与完整 assistant，并以 <|im_end|> 结尾（思考链完整序列）；参数 (system, user, think, assistant)
instruct_think_lm_template = lambda system, user, think, assistant: ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n%s\n</think>\n\n%s<|im_end|>" % (system, user, think, assistant,)) if system else ("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n%s\n</think>\n\n%s<|im_end|>" % (user, think, assistant,))

# 仅助手回复片段加结尾符（如只取「回答」部分时用）
assistant_template = lambda system, assistant: "%s<|im_end|>" % assistant

# 纯语言建模：不包任何对话符，直接返回 user 文本
lm_template = lambda system, user: user

# 翻译任务示例：在 user 外包一层固定指令
instruct_task_template = lambda system, user: instruct_template(system, "Translate the input into German, do not generate the other contents apart from the translation. Input: %s" % user)

# 模板名 -> 模板函数，供 tools/plm/map/qwen/v3.py 等通过名字选用（如 instruct_auto、assistant）
templated = {"instruct": instruct_template, "instruct_lm": instruct_lm_template, "assistant": assistant_template, "lm": lm_template, "instruct_auto": instruct_auto_template, "instruct_think": instruct_think_template, "instruct_think_lm": instruct_think_lm_template, "instruct_task": instruct_task_template}
