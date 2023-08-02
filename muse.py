from transformers.generation import LogitsWarper, LogitsProcessorList
import torch

class MuseLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs dampening of the k highest probability elements.

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        damp (`float`, *optional*, defaults to 0.98):
            How much less likely should the top_k most likely tokens be made. If set to 0, they become impossible.
    """

    def __init__(self, top_k: int, damp: float = 0.98, damp_initial: float = 1.0, damp_ramp_tokens: int = 0, min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        self.top_k = max(top_k, min_tokens_to_keep)
        self.damp = damp
        self.damp_initial = damp_initial
        self.damp_ramp_tokens = damp_ramp_tokens
        self.input_ids_length = None

    def reset(self):
        print('the-muse has reset')
        self.token_num = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # reset if input suddenly decreases in length
        if self.input_ids_length is None or self.input_ids_length > input_ids.size(-1):            
            self.reset()
        self.input_ids_length = input_ids.size(-1)

        top_k = min(self.top_k, scores.size(-1))  # Safety check

        ratio = 1.0 if self.damp_ramp_tokens == 0 else min(self.token_num/self.damp_ramp_tokens, 1.0)        
        linear_damp = self.damp_initial + ratio*(self.damp - self.damp_initial) if ratio < 1.0 else self.damp

        topk_values, topk_indices = torch.topk(scores, top_k)
        dampened_values = topk_values * linear_damp
        scores.scatter_(-1, topk_indices, dampened_values)

        self.token_num += 1

        return scores


from transformers import AutoTokenizer, AutoModelForCausalLM

# init
tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1", torch_dtype=torch.float16).to('cuda')

# init custom logit processor chain
from transformers.generation import TopPLogitsWarper, TopKLogitsWarper, TemperatureLogitsWarper

logits_processor = LogitsProcessorList()
logits_processor.append(TemperatureLogitsWarper(temperature=0.7))
the_muse = MuseLogitsWarper(top_k=3, damp=0.9, damp_ramp_tokens=0)
logits_processor.append(the_muse)
logits_processor.append(TopPLogitsWarper(top_p=0.7))
logits_processor.append(TopKLogitsWarper(top_k=50))

# infer
prompt = "<human>: Please write a story about polar bears.\n<bot>:"
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
import tensorflow as tf

for x in range(3):
    outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.7, top_k=50, return_dict_in_generate=True)
    input_length = inputs.input_ids.shape[1]
    token = outputs.sequences[0, input_length:]
    output_str = tokenizer.decode(token)

    print("---reference",x,"---")
    print(output_str)

for x in range(5):
    outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, logits_processor=logits_processor, return_dict_in_generate=True)

    input_length = inputs.input_ids.shape[1]
    token = outputs.sequences[0, input_length:]
    output_str = tokenizer.decode(token)
    print("---muse",x,"---")
    print(output_str)
