# Key_detection

Key_detection is a library that (as the names says) detects keys
in images. It proposes the uses of several deep neural networks structures.
It can also be used to detect keys in video files.

## Installation

No installation needed. Clone the repository and enjoy.

## Usage

Key detection in a image 
```bash
detect_key.py --model model-name --image --path path-to-image --output path-to-result-image
```

Key detection in a video
```bash
detect_key.py --model model-name --video --path path-to-video --output path-to-result-video
```

Retrain weights
```bash
train.py --model model-name --initial_epoch inep --epoch ep --batch_size bsiz --annotations path-to-annotations --classes path-to-classes --anchors path-to-anchors
```