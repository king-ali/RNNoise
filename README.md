# Rnnoise window demo

RNNoise is a noise suppression library based on a recurrent neural network

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

![](./Img/output.png)

Below in our case nearend.wav is speech signal and farend.wav is noise

```bash
cd ..
cd src
./compile.sh
./denoise_training nearend.wav farend.wav 50000 > training.f32
```

```bash
cd ..
cd training
python3 ./bin2hdf5.py ../src/training.f32 50000 87 training.h5
python3 ./rnn_train.py
```

![](./Img/train.png)


