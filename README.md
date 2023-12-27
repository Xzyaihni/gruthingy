# gruthingy
a gru neural network. cool.

# how to do
```
git clone https://github.com/Xzyaihni/gruthingy
cd gruthingy
cargo b -r
```

u can change network parameters and stuff in src/neural\_network/neural\_network\_config.rs

then do
```
./target/release/gruthingy -m train -i ~/path/to/some/cool/text/file.txt -I 100
```
to train it for 100 minibatches!!

then run with this
```
./target/release/gruthingy -m run -i 'test text'
```
