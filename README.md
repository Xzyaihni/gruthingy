# gruthingy
a gru neural network. cool.

i have no clue why it segfaults for me randomly sometimes after adding arrayfire and im so done
is it some nvidia stupidity?

# how to do
```
git clone https://github.com/Xzyaihni/gruthingy
cd gruthingy
cargo b -r
```

then do
```
./target/release/gruthingy train_new ~/path/to/some/cool/text/file.txt -e 100
```
to train it for 100 epochs (not actually epochs lol)

then run with this
```
./target/release/gruthingy run 'test text'
```