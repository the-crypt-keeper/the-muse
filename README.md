# The Muse

This project implements `MuseLogitsWarper`, a simple logit processor that makes the top logit propabilities less likely.

To speed up iteration we experiment with [togethercomputer/RedPajama-INCITE-Chat-3B-v1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1)

# How does it work?

The Muse is essentially a top-k constrained temperature processor. It's inserted into the typical logit processing chain, sitting between Temperature and TopK.  It's job is to make the output a little more creative by reducing the propabilities of the most probable tokens (parameter `top_k`) by a little bit (parameter `damp`).  To make the result more coherent we ramp the peantly from `damp_initial` (usually 1.0 but could be higher!) to `damp` over the span of `damp_ramp_tokens` tokens.

## Examples without Muse 

```
Once upon a time, there was a polar bear named Paddington. He was a big bear with a big heart and a big appetite. He loved to eat marmalade sandwiches and watch the snow fall outside his window.

One day, Paddington woke up and felt a little different. His fur was a little bit thicker and his paws were a little bit bigger. He thought it was just a winter cold, but as the days went by, he realized that something was wrong.

Paddington started to feel very sad and lonely. He couldn't find his family anywhere. He searched all over the
```

```
Once upon a time, there was a polar bear named Paddington. He lived in the Arctic with his mother and father, who were polar bears like him. Paddington was a very curious bear, always asking questions and wanting to learn more about the world around him.

One day, Paddington heard a noise outside his den and went out to investigate. He saw a small group of humans, dressed in colorful clothes and carrying strange-looking instruments. They were exploring the Arctic and had stumbled upon Paddington's den.

Paddington was surprised to see humans in the Arctic, but he was even more
```

```
The polar bear was a polar bear, but he was not like other polar bears. He was the only polar bear who could swim. He could swim in the sea, and he could swim in the river. He could swim in the sea, and he could swim in the river. He could swim in the sea, and he could swim in the river. He could swim in the sea, and he could swim in the river. He could swim in the sea, and he could swim in the river. He could swim in the sea, and he could swim in the river. He could swim in the sea, and he could swim
```


```
Once upon a time, there was a polar bear named Pikachu. Pikachu was a friendly bear who loved to play with his friends, the penguins. One day, Pikachu decided to take a trip to the North Pole. He packed his bags and set off on his adventure.

As Pikachu walked through the snowy mountains, he saw a group of penguins standing in front of a large, mysterious building. The penguins were wearing thick coats and had hats on their heads, making them look like they were from another world.

Pikachu walked up to
```


```
 The polar bear is a symbol of strength and endurance. It is known for its ability to survive in the harshest of conditions. Despite its impressive physical attributes, the polar bear is a gentle creature that only attacks when provoked.

One day, a polar bear was walking through the forest when he noticed a small group of deer grazing in the meadows. The bear approached the deer cautiously, but as he got closer, he realized that they were not the normal deer he was used to. These deer were much smaller than he was used to, and they were running away from him.

The bear was confused and unsure of
```

## Examples with Muse (top_k=1, damp=0.99, damp_ramp_tokens=32)

```
 It started as just any ordinary morning for the Polar bear, but soon, he found that he couldn’ t wake his friend the walruss.
The polar bears had always lived together in harmony. But today was a day of first for the two of the them, they had never had a disagreement. 
 The walrus woke the bear, saying, "We have a new baby, it’ s so small and it looks like a dot."
 The walruses first statement caused a disagreement. "That is a silly statement. A walruses are never small, we’ re the largest in our group, we’
```

``` Polar Bears: A Story
Once, in a land called Antarctisia (which was the cold and icy frozen continent that surrounded all the other lands of Earth) there was once lived two bears. One bear, called Bear One was the son and grandson to two previous bears of that same species, who lived there for generations before them, as all of their family members did, with no notable event to mark them as special or exceptional, save perhaps their tendency for having very thick, furry, and very large, round ears, and a distinct lack in having a chin, as most polar bear bears have. The second of these
```

``` It is winter and a group polar bear has been living their life by a small island. The weather has become very bad, the sea ice is starting melting. They have to move north and start hunting seals again to keep them from getting starving to die in this new environment, and to give their babies the chance of growing. But it's so hard for the bear cub to cross through this new sea.

One night the bear saw something in front. A man in red jacket, carrying something on the boat with a sail on top of the water, was crossing through this new ice and water, going to this small islet and
```

``` The sun had risen high above a snow-crowned mount and shone down through a veil that was slowly being blown aside. A solitary polar fox stood at its base.
The sun was warming his belly, but the bears' huddle of sleeping bodies around the hot spring had no need for such warmth as the water's heat would provide them with enough. The bears had grown used of this hot water, which was now flowing slower than ever before, as it had become so warm it no more retained heat than water itself, but had now turned to steam instead of a solid. The bear had no more use of this hot spring,
```

See [more examples](https://www.reddit.com/r/LocalLLaMA/comments/15ffzw5/presenting_the_muse_a_logit_sampler_that_makes/) in the reddit post.