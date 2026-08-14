import re, sys
from mlx_vlm import load, generate
B=sys.argv[1]
m,proc=load(B)
# Render through the model's OWN chat template, which handles {"type":"video"}
# by inserting <|vision_start|><|video_pad|><|vision_end|> (jinja line 29).
msgs=[{"role":"user","content":[
    {"type":"video"},
    {"type":"text","text":"Describe what happens in this video. What moves, and in which direction?"}]}]
p=proc.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
assert "<|video_pad|>" in p, f"template did not insert video pad: {p[-200:]!r}"
o=str(generate(m,proc,p,video=["vidprobe.mp4"],max_tokens=220,temperature=0.6,verbose=False))
g=re.search(r"text='(.*?)', token=", o, re.S)
print("VIDEO:", (g.group(1) if g else o)[:320].replace("\\n"," "))
