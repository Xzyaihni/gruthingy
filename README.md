# gruthingy
my lil autodiff with its own little ssa ir that also tries to reuse tensor memory

doesnt actually use gru units by default, default is star units but it can be changed!

architecture options can be changed in **src/neural\_network/neural\_network\_config.rs**


# random info
biases 0 initialized, weights with orthogonal initialization

loops are not unrolled so the ir supports variable sized inputs, idk if other compiled thingies do that (it took me SO long to figure out how to differentiate a loop though)

has A LOT of options that you can access with **--help**, by default its a medium sized stacked star rnn with dropout regularization and scaffolded dropout bpe as the tokenizer, also has a small-ish embeddings layer and uses adamx as the optimizer

batches are just randomly picked sequences from the input file cuz i didnt wanna assume any kind of input structure, so no sentence end tokens or anything like that

during inference does monte carlo dropout with the --batch amount of dropped out runs (--dropout and --input-dropout have to be the same as during training for it to give accurate approximations)



# exe. where. what are these? codes?
these are called bash commands btw, cuz i use arch btw, linux user btw

```
git clone --depth 1 https://github.com/Xzyaihni/gruthingy
cd gruthingy
cargo b -r
```

then do
```
./target/release/gruthingy -m train -i ~/path/to/some/cool/text/file.txt -I 100
```
to train it for 100 minibatches!!

then run with this
```
./target/release/gruthingy -m run -i 'test text'
```
