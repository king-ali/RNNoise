# Rnnoise

This is the most comprehensive detail for RNNoise, a noise suppression library built upon a recurrent neural network. RNNoise delivers top-notch real-time noise reduction, ensuring a seamless audio communication experience on mobile devices by eliminating background noises and echoes. While the official implementation and advanced versions of RNNoise yield impressive results, they still face challenges in effectively suppressing echoes and various types of noise while preserving speech quality. We trained and fine-tuned RNNoise to address these challenges successfully.

## Demo window

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
./denoise_training ../combined_clean_segments.raw ../combined_echo_segments.raw 500000 > training.f32
```

```bash
cd ..
cd training
python3 ./bin2hdf5.py ../src/training.f32 500000 87 training.h5
python3 ./rnn_train.py
```

<img src="/img/train.png" width="500" height="200">

We used our own dataset gathered from phones. For example dataset you can use Microsoft MS-SNSD to test [https://github.com/microsoft/MS-SNSD]


```bash
python3 ./dump_rnn.py weights.hdf5 ../src/rnn_data.c ../src/rnn_data.h orig
```

if you are facing issue of .h files being changed then revert that change then proceed furthur

```bash
cd ..
make
```


Results

<img src="/img/output2.png" width="1000" height="200">



## Using pretrained model

Upon inference we found that Hogwash variant performs better in echo and noise suppression than Vanila, so we used pretrained model of hogwash for training on our dataset.

Results


<img src="/img/hog3.png" width="1000" height="400">

<img src="/img/hog2.png" width="1000" height="400">



## Data preprocessing

We gathered dataset of speech and echo in respective folders, the data is those folders is combined to make single speech and one echo file. Next we divided them to 30 second segemnt as required by rnnoise for feature extraction and training. Finally we combined all files into PCM 16-bit, 48000 Hz raw file as clean.raw and echo.raw.

Within the pre-processing folder, the "clean" folder contains the speech data, while the "echo" folder have the echo data. To initiate the pre-processing, execute the "p1.py" script, which combne all the files and saves them in the output folder. Following this, execute the "p2.py" script, which splits the combined file into 30-second segments. Subsequently, execute the "p3.py" script to combine all the segmented files. Finally, Audacity to convert these files into the raw format.



## Data Augmentation

To enhance our dataset, we performed data augmentation by collecting various impulse responses from different environments. We then applied convolution to these impulse responses with our echo and noise data.











