## Converting Ego4D dataset to RLDS


#### Step.0 Prepare Pre-training Dataset
Download [Ego4D](https://ego4d-data.org/docs/start-here/) Hand-and-Object dataset:
```
# Download the CLI
pip install ego4d
# Select Subset Of Hand-and-Object
python -m ego4d.cli.cli --output_directory=<path-to-save-dir> --datasets clips annotations  --metadata --version v2 --benchmarks FHO
```

Your directory tree should look like this:
```
$<path-to-ego4d-save-dir>
├── ego4d.json
└── v2
    |—— annotations
    └── clips
```


#### :one: Install necessary dependencies

First create a conda environment using the provided environment.yml file (use `environment_ubuntu.yml` or `environment_macos.yml` depending on the operating system you're using):
```
conda env create -f environment_ubuntu.yml
```

Then activate the environment using:
```
conda activate rlds_env
cd vla-scripts/extern/ego4d_rlds_dataset_builder
pip install -e .
```

Then, download all necessary dependencies form [huggingface](https://huggingface.co/datasets/qwbu/univla-ego4d-rlds-dependencies) and put them under ```vla-scripts/extern/ego4d_rlds_dataset_builder```.


#### :two: We first extract the interaction frames (video clips within ```pre_frame``` and ```post_frame```) with a FPS of 2 and save them as ```.npy``` files.

We first process the citical information about the interaction clips and key frames (```pre_frame```, ```pnr_frame```, and ```post_frame```) into a json file (```info_clips.json```) with [this script](https://github.com/OpenDriveLab/MPI/blob/79798d0d6c40919adcf3263c6df7e86758fdd59a/prepare_dataset.py), or you can directly download the json file from [here](https://huggingface.co/datasets/qwbu/univla-ego4d-rlds-dependencies).

```bash
python preprocess_ego4d.py \
    --denseclips_dir /path/to/output/denseclips \           # output dir for processed clips
    --info_clips_json /path/to/info_clips.json \            # metadata of keyframes
    --source_videos_dir <path-to-ego4d-save-dir>/v2/clips \       # ego4d videos path
    --frame_interval 15                                     # downsample Ego4D to 2 fps
```


#### :three: We then create episodes with according to desirable format with:

```bash
mkdir ../ego4d_rlds_dataset_builder/ego4d/data
mkdir ../ego4d_rlds_dataset_builder/ego4d/data/train

python create_episode_ego4d.py \
    --source_dir /path/to/output/denseclips \                       # processed clips from the step.2
    --target_dir ../ego4d_rlds_dataset_builder/ego4d/data/train \     # path to save episodes
    --annotation_file /path/to/output/denseclips/annotations.json \ # processed meta-info from step.2
    --processes 64                                                  # multi-processing
```

#### :four: Create ego4d rlds dataset

```bash
cd vla-scripts/extern/ego4d_rlds_dataset_builder/ego4d
tfds build --overwrite --beam_pipeline_options="direct_running_mode=multi_processing,direct_num_workers=16"
```

The default save path for the dataset is `/root/tensorflow_datasets/ego4d_dataset`. Directly process the whole dataset may face memory limit issue, we can split the dataset into several parts and process them separately:

```bash
cd vla-scripts/extern/ego4d_rlds_dataset_builder/ego4d
mkdir data/val
rsync -av --files-from=<(printf "episode_%d.npy\n" {0000..9999}) data/train/ data/val/
tfds build --overwrite --beam_pipeline_options="direct_running_mode=multi_processing,direct_num_workers=4"
mkdir /root/tensorflow_datasets/ego4d_dataset/ego4d_split_1
mv /root/tensorflow_datasets/ego4d_dataset/1.0.0 /root/tensorflow_datasets/ego4d_dataset/ego4d_split_1/1.0.0
rm -r data/val

rsync -av --files-from=<(printf "episode_%d.npy\n" {10000..19999}) data/train/ data/val/
tfds build --overwrite --beam_pipeline_options="direct_running_mode=multi_processing,direct_num_workers=4"
mkdir /root/tensorflow_datasets/ego4d_dataset/ego4d_split_2
mv /root/tensorflow_datasets/ego4d_dataset/1.0.0 /root/tensorflow_datasets/ego4d_dataset/ego4d_split_2/1.0.0
rm -r data/val

# repeat until all data is processed
```
