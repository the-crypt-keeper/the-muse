# The Muse

This project implements `MuseLogitsWarper`, a simple logit processor that makes the top logit propabilities less likely.

To speed up iteration we experiment with [togethercomputer/RedPajama-INCITE-Chat-3B-v1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1)

# How does it work?

The Muse is a stateful, top-k constrained temperature processor. It's inserted into the typical logit processing chain, sitting between Temperature and TopK.

It's job is to make the output a little more creative by reducing the propabilities of the most probable tokens (parameter `top_k`) by a little bit (parameter `damp`).  To make the result more coherent we ramp the penalty from `damp_initial` (usually 1.0 but could be higher!) to `damp` over the span of the first `damp_ramp_tokens` tokens.

# Some ideas on tuning the parameters

`damp` is effectively `1/temperature` but applied to only the `top_k` logits.

Keep `top_k` fairly low, probably single digits otherwise this logit processor essentially degenerates into Temperature.

`damp_ramp_tokens` = 0 will skip straight to peak creativity which causes some really interesting generations but with a high chance of going off the rails.

`damp_initial` > 1.0 may be interesting to explore.

Theres no particular reason that damp > damp_initial, the ramp could very well run the other direction (start crazy and then get boring)

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

## Examples with Muse (top_k=3, damp=0.9, damp_ramp_tokens=32)

```
Once upon a time, in a far-off land, there lived a polar bear named Yogi. Yogi was a very friendly bear. He lived alone in a beautiful Arctic land and was known for his kindness, his sense of humor, and his great intelligence. One day, Yogi had a vision: he saw the icebergs melting into the sea and he was terrified at how much ice and icebergs would disappear. This meant the ice would eventually break into tiny pieces and fall into the sea and become dangerous icebergs called glaciers. Glaciers could not melt into the sea. It needed to go in
```

```
Once upon a time, in the icy tundra of the North Pole, a polar bear and her cub were exploring the snowy mountains and hills near their Arctic habitat. As they moved slowly over the snow, they could sense something moving just ahead in the deep powder, as if someone had hidden beneath their own paws. As they cautiously advanced, their senses began to register an eerie feeling in the air and an icy gust blowing in their direction. It didn't take long until they discovered their mystery visitor - a large group of reindeer! As one of the adults turned toward them, their cub was terrified.
```

```
Once upon a time, there lived a polar bear named Nana. She was the oldest bear in the Arctic, with an age of 500 years, making her feel wiser than anyone else her age. As an expert polar bear, Nana could not understand what had led so many young bears to the city. It was because many had become disenchanted with her nature in this world where there are always options and compromises that make the most rational bear think he had become a polar bear trapped in a polar bear world.

The polar bears had longed to go on adventures to new lands where bears of their nature roamed
```

```
Once upon a time, there was a polar bear named Paul. Paul was a big and strong polar bear with a shiny fur coat, large flippers for swimming, and a special white patch on his face. Paul had spent many years swimming through the cold, arctic seas, exploring for new prey and finding a cozy den to call home. He lived a carefree life among his Arctic companions and enjoyed hunting seals with his spear or bumbling into his favorite berry patch. He was a big teddy bear and his soft fur kept him warm on a winter morning in Alaska. However, the harshness of the polar environment would sometimes
```

```
Once upon a time, there was a polar bear named Misha. Misha was a big and strong polar bear and he had a lot of experience on ice floes. His name is short for "migratory ice breaker." His favorite hobby was exploring the frozen ocean in search of fish to eat. One day, a young female polar bear was spotted wandering through the frozen landscape of the arctic.

She was thin and weak, barely able to stand, her fur grayish-green from being trapped for so long, and the marks left from starvation. The sight of her caused an overwhelming rush of feelings inside M
```

```
Once upon a time, there lived polar bears deep in the heart of the Arctic. These bears, who were the strongest in the icy land, spent the winter in a hidden ice cave and returned to the shore each spring for a summer of food and fresh water.

Their domain stretched far into the ice fields that stretched far above, with a clear blue sky stretching over it and snow on its highest points. The sun shone brightly above their backs in an eternal and gentle dance across the frozen fields. Yet even on these bright sunny days, their cave was often covered in a thick fog of snow. This allowed them to see through these
```

```
Once upon a time, there was a polar bear named Baloo. Baloo was a very brave bear. He would go anywhere and do anything he pleased. However, sometimes his actions got him into trouble. For example, when he ate some magic mushrooms that made him invisible! Now all he saw were his reflections and all around him was darkness and gloom. Baloo was confused and didn't know which way to go or which direction was up.

After several days in the pitch dark, Baloo suddenly found himself face to face with something very scary - a very angry witch with burning hat brim! "YOU ARE NOW E
```

## Examples with Muse (top_k=3, damp=0.9, damp_ramp_tokens=0)

```
In a world without climate change, polar bears lived peacefully. Their arctic homes were safe from melting ice. Then a group of radical environmentalists got together and proposed the idea of banning fossil fuels in a move called "degrowth." Polar bears who once depended on melting ice for a hearty seal hunt had nothing but seaweed, algae, and lichen to survive. With nothing else to hunt, some polar bears starved or froze to death. Some had to swim long distances, carrying heavy blubber and fur for days to get to an island of green trees with plentiful fruits and leaves that made
```

```
A young polar bear, Jens, grew up in a harsh Arctic habitat and knew all too well about the difficulties of his existence. In an effort to learn and improve, Jens trained himself on all aspects of Arctic life. As Jens aged, he gained an understanding of what life was like outside of his native region. One day, he met a man in New York, an anthropologist by the name of Edward. He and Edward began traveling across continents together, sharing their unique view of their shared habitat. During these adventures, Edward asked Jens about his knowledge of life on his northern continent. The two discussed the
```

```
One chilly night, polar bears huddled near each other on the icy shores of their polar habitat. Their ears pricked for any noise or disturbance in the air above or under the water surrounding them. Their leader noticed movement from above the horizon in the direction of his home and raised their alarm. Soon enough, their keen ears caught the faint sound of an aircraft in the distance approaching the area. The bears slowly shifted from the position they had settled on. As one by one, their bodies began to lift above the snow. Soon a team of biologists landed nearby to capture one of the bears in the name of
```