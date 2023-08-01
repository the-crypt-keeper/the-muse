# The Muse

This project implements `MuseLogitsWarper`, a simple logit processor that makes the top logit propabilities less likely.

To speed up iteration we experiment with [togethercomputer/RedPajama-INCITE-Chat-3B-v1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1)

# How does it work?

The Muse is inserted into the typical logit processing chain, sitting between Temperature and TopK.  It's job is to make the output a little more creative by reducing the propabilities of the most probable tokens (parameter `top_k`) by a little bit (parameter `damp`).

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

## Examples with Muse (top_k=1, damp=0.99)

```
 It's winter time and a young girl and a little bear were sitting together. They talked for ages. They told about how polar ice meltr was shrinking. And it is affecting them a great.
The polar ice was the main food supply to many bears, seals. They needed to catch the seal for them and feed the cub, and keep their little home alive! 

 
  - So they talked a bit longer about what could happen in their life without that source to survive from

They started talking in low tones so they won`'´ t disturb others around

And the cub asked the bear a great and very
```

```
Polar Bear Storytime: The Snow is Melts! We all knew the ice wouldn´´´ melt eventually but the end is always hard for some bears, and I was afraid I was the unoved polar. My best buddy is Bear 2. She has her very long, shiny and flaky tongue, that I envy because I am the second in command in her group and have less special skills, and my best friends have a shiny and beautiful belly! Bear 3 doesn’ttendto share our sadness. The snow melters are just beautiful for the other two but I don't want the world without my beloved icy land! But
```

```
 One winter night a young Polar bears wandered outside the village, they had heard about bears who would dance on a pole outside. So, one young polarbear wandered to this place to find if there are polarbats here! When the bear reached there the bear found the place very empty and there are not many people in that area so it decided it's a perfect opportunity and danced with a joyful sound on that tree, but the bears soon felt tired so they took rest in some branches, then next thing that bear felt, a little bird landed at there! And soon a fight began! the little bear asked if you dance for us
```

```
 It's cold, it’d been snow for weeks now and every morning I see polar Bears on their feet walking through our forest on a sunny winter afternoon! They have no fear in them! It was the time to come home! We have to have the warm clothes in order not get cold in our home, so the next step we take was the most dangerous, to take our children with. I can see their little face are smiling but we can hear them cry, they can hear them, I have a strong heart, it was so tough! We are exhausted when I reach their doorstep!
They have to stay warm!
```