# Rnnoise window

RNNoise is a noise suppression library based on a recurrent neural network. Rnnoise provides high quality real-time noise suppression that can run smoothly on mobiles and browsers. 

## Demo

Get Wsl on window
Results in files are removed from /mnt/wsl so you can do somewhere /mnt/c or d
```bash
cd /mnt/wsl
```

```bash
git clone https://github.com/xiph/rnnoise.git
```

```bash
sudo apt install autoconf
sudo apt-get install libtool
```

```bash
./autogen.sh
./configure
make
```

```bash
explorer.exe .
```
Replace below  farend.wav with yout noisy speech and rnnoise_farend.wav as output denoised

```bash
./examples/rnnoise_demo farend.wav rnnoise_farend.wav
```
The output is a 16-bit raw PCM file

<img src="/img/output.png" width="1000" height="200">

## Training

For training we need a clean speech and noise only file. 

```bash
cd ..
cd src
./compile.sh
./denoise_training clean_speech.wav noise.wav 50000 > training.f32
```

```bash
cd ..
cd training
python3 ./bin2hdf5.py ../src/training.f32 50000 87 training.h5
python3 ./rnn_train.py
```

<img src="/img/train.png" width="500" height="200">

We used our own dataset gathered from phones. For example dataset you can use Microsoft MS-SNSD to test [here][https://github.com/microsoft/MS-SNSD]


```bash
./dump_rnn.py weights.hdf5 ../src/rnn_data.c ../src/rnn_data.h orig
```

if you are facing issue of .h files being changed then revert that change then proceed furthur

```bash
cd ..
./autogen.sh
./configure
make
```

Results

<img src="/img/output2.png" width="1000" height="200">




