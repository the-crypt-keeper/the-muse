from transformers.generation import LogitsWarper, LogitsProcessorList
import torch
import math

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

# from https://github.com/oobabooga/text-generation-webui/blob/main/modules/sampler_hijack.py#L80
class MirostatLogitsWarper(LogitsWarper):
    def __init__(self, mirostat_mode: int, mirostat_tau: float, mirostat_eta: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if mirostat_mode not in [2]:
            raise ValueError(f"`mirostat` has to be a an integer 2, but is {mirostat_mode}")
        self.mirostat_mode = mirostat_mode
        self.mirostat_eta = mirostat_eta
        self.mirostat_tau = mirostat_tau
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep
        self.mu = 2 * self.mirostat_tau
        self.e = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        logits = scores[0]
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        prob_original = torch.softmax(sorted_logits, dim=-1).tolist()  # candidates

        # Truncate the words with surprise values greater than mu
        for i, candidate in enumerate(prob_original):
            if candidate > 0 and -math.log2(candidate) > self.mu:
                if (i == 0):
                    sorted_logits = sorted_logits[:1]
                else:
                    sorted_logits = sorted_logits[:i]
                break

        # Normalize the probabilities of the remaining words
        prob_topk = torch.softmax(sorted_logits, dim=0)

        prev_i = torch.multinomial(prob_topk, num_samples=1, replacement=True).to('cuda')

        observed_surprise = -math.log2(prob_topk[prev_i])
        self.e = observed_surprise - self.mirostat_tau

        # Update mu using the learning rate and error
        self.mu -= self.mirostat_eta * self.e

        sorted_indices_to_remove = torch.ones_like(scores[0], dtype=torch.bool)
        sorted_indices_to_remove[prev_i] = False

        indices_to_remove = sorted_indices_to_remove.unsqueeze(0).scatter(1, sorted_indices.unsqueeze(0), sorted_indices_to_remove.unsqueeze(0))
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores

# init custom logit processor chain
from transformers.generation import TemperatureLogitsWarper

temperature = TemperatureLogitsWarper(temperature=0.7)
mirostat = MirostatLogitsWarper(mirostat_mode=2, mirostat_tau=0.1, mirostat_eta=5.0)

mirostat_pipe = LogitsProcessorList()
mirostat_pipe.append(temperature)
mirostat_pipe.append(mirostat)

the_muse = MuseLogitsWarper(top_k=3, damp=0.9, damp_ramp_tokens=0)

miromuse_pipe = LogitsProcessorList()
miromuse_pipe.append(the_muse)
miromuse_pipe.append(temperature)
miromuse_pipe.append(mirostat)

def test_inference(model, reference_gens = 3, muse_gens = 5):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # init
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map="auto")

    # infer
    prompt = "<human>: Please write a story about polar bears.\n<bot>:"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    import tensorflow as tf

    for x in range(reference_gens):
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, logits_processor=mirostat_pipe, return_dict_in_generate=True)
        input_length = inputs.input_ids.shape[1]
        token = outputs.sequences[0, input_length:]
        output_str = tokenizer.decode(token)

        print("---reference",x,"---")
        print(output_str)

    for x in range(muse_gens):
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, logits_processor=miromuse_pipe, return_dict_in_generate=True)

        input_length = inputs.input_ids.shape[1]
        token = outputs.sequences[0, input_length:]
        output_str = tokenizer.decode(token)
        print("---muse",x,"---")
        print(output_str)

if __name__ == "__main__":
    test_inference("togethercomputer/RedPajama-INCITE-Chat-3B-v1")