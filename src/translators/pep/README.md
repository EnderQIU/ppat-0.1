# Predict English Pronunciations

You should **only** visit this "pep" submodule by api.py from outer scope.

## Generate Dictionary

Select `src/` as the source of the project and run `python translators/pep/dictionary_generator.py`.

## Train model

Select `src/` as the source of the project and run `python translators/pep/trainer.py`.
Note this will take up to **2 hours** (i7-6700 with GTX1070).

## Use PEP API from Outer Scope

Python interactive shell example:

```python
>>> from translators.pep.api import PhoneticDictionary
Using TensorFlow backend.
>>> p = PhoneticDictionary()
2019-XX-XX XX:XX:XX.XXXXXX: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2019-XX-XX XX:XX:XX.XXXXXX: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: GeForce GTX 1070 major: 6 minor: 1 memoryClockRate(GHz): 1.759
pciBusID: 0000:01:00.0
totalMemory: 8.00GiB freeMemory: 6.63GiB
2019-XX-XX XX:XX:XX.XXXXXX: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2019-XX-XX XX:XX:XX.XXXXXX: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-XX-XX XX:XX:XX.XXXXXX: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0
2019-XX-XX XX:XX:XX.XXXXXX: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N
2019-XX-XX XX:XX:XX.XXXXXX: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6384 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1070, pci bus id: 0000:01:00.0, compute capability: 6.1)
>>> p.lookup('green')
'G R IY1 N'
>>> p.predict('green')
'G R IY1 N'
```

## Introduction of Training Model

### Dataset Preparation

1. `cmudict-0.7b` and `cmudict-0.7b.symbols` stored in `dataset/` are downloaded from
 [The CMU Pronouncing Dictionary]http://www.speech.cs.cmu.edu/cgi-bin/cmudict which has more than 100,000 words'
 pronunciations.

2. Filtering: Since some words are abbreviations, or have multiple correct pronunciations, we should pre-process those
 into normal features. See details in [dictionary_generator.py](dictionary_generator.py).

3. Tokenizing: Transform each char and symbol into ids (positive integers).

4. One-hot-encoding the words' chars and pronunciations' symbols. Sequence the verticals to the same dimension.