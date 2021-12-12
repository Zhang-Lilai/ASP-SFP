# Arbitrary-oriented Spatial and Scalable Frequency Pooling for Lightweight Facial Expression Recognition

## Requirements

- Python >= 3.6
- PyTorch >= 1.2
- torchvision >= 0.4.0

## Training

- Step 1: download basic emotions dataset of [FER2013]
- Step 2: transform it into png

```txt

./datasets/
     FER2013/
         train/
               0/
                 train_09748.jpg
                 ...
                 train_12271.jpg
               1/
               ...
               6/
         test/
              0/
              ...
              6/

[Note] 0: Neutral; 1: Happiness; 2: Sadness; 3: Surprise; 4: Fear; 5: Disgust; 6: Anger
```

- Step 2: change the ***--data*** in *run.sh* to your path
- Step 3: run ``` sh run.sh ```