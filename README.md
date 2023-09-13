# gruthingy
a gru neural network. cool.

# how to do
```
git clone https://github.com/Xzyaihni/gruthingy
cd gruthingy
cargo b -r
```

u can change network parameters and stuff in src/neural\_network.rs

then do
```
./target/release/gruthingy train_new ~/path/to/some/cool/text/file.txt ascii_dictionary.txt -e 100
```
to train it for 100 minibatches!!

then run with this
```
./target/release/gruthingy run 'test text'
```
