# Data Preparation
## ScanRefer, Scan2CAD:

For simplicity, all related data regarding to Scan2CAD and ScanRefer dataset are already prepared and you can use them directly

## ScanNet v2:

1.Please download and put your scannet data in this folder. Your folder should look like this:
```
data
├── scannet
│   ├── scans
│       ├── scene0000_00
│            ├── scene0000_00_vh_clean_2.ply
│            ├── scene0000_00_vh_clean_2.labels.ply
│            ├── scene0000_00_vh_clean_2.0.010000.segs.json
│            ├── scene0000_00.aggregation.json
│            ├── scene0000_00.txt
│       ├── .......
│   ├── scan_test
│       ├── ......
```

2.Split and preprocess data
```
cd SoftCap/data/scannet
bash prepare_data.sh
```

3.After running the script the scannet dataset structure should look like below.
```
SoftCap
├── data
│   ├── scannetv2
│   │   ├── scans
│   │   ├── scans_test
│   │   ├── train
│   │   ├── val
│   │   ├── test
│   │   ├── val_gt
```

## glove:
please also put the "glove.p" file in this data folder
