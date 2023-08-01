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
 Polar Bears in Space! The story starts on the icy surface where two bears, a mother polar and a son, a young bear are resting in their summer dens, just a couple weeks away of giving birt.
<bot> Polar Bears on Earth were a staple in many people’ s childhood. The adorable, lovably googly eyes, and big round ears of these adorable creatures would bring joy and happiness into the lives. However this story takes a different twist, as polar bear parents take to space. A small group, a father, a son and two other polar bear families are launched to space on the first commercial polar flight
```

``` The Arctic is warming, polar bear families and habitats shrinking
<bot> Once, in a frozen Arctic far away from the bustling city lights of modern day civilization lived polar bear family, with one mom and her two adorable little ones, the young ones named Hibiki, a male and Maki a young girl, the mother, whose fur was as thick and soft, was named Yuko, the bear was getting hungry and so she decided that they should hunt some prey for their next feast, they would go hunting in a small cave nearby. They were able a long walk from the main land to get there. The bears entered
```

``` Polar Bears
Once there lived polar bear in a land called Antarctis, he had been living with other polar bear friends. He lived a simple but comfortable lifestyle in a land of snow, sea ice and frozen land masses, and had been a polar bears best friends. He was known as a quiet and friendly polar bears and lived in the company with his best friend. One fine morning he was walking through his home, and noticed a stranger, he did know but was a stranger. The two of then had been walking through a large open space and saw many different species, and he noticed a different one, and saw that the new
```

``` The Arctic
Once, in a remote and icy region, there were a mother and a baby bear, and the bear mother decided that it would do well for the two to venture into a neighboring forest to hunt.
The mother was in the lead and was walking along, with her cub on the end, and the forest seemed quite welcoming to her, as it seemed full with food, as the other animals would surely have to have left the place. The bear was starting on her prey when suddenly a loud and powerful roar was emitted by an unknown beast.

As she was startled by this noise and the fact of being so far
```

``` The Arctic is warming faster and becoming a more hostile environment, forcing the bears into new habitats and challenging them with unfamiliar conditions and predators

One day a group called Polar Bear Watch found themselves stranded in a small village.  The villagers, afraid and unsure, tried their hardest not only not harm them but to also offer help and support, which was much needed in this new and strange environment
<bot> Once there was once an arcticle called Polar Bears.

It began in a village.
A village in a far off place, where people live and breathe for their livelihoods, for the next generation, but for```