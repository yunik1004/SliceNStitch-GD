# SliceNStitch-GD

Source code for SliceNStitch-GD, which is a gradient descent version of [SliceNStitch: Continuous CP Decomposition of Sparse Tensor Streams](https://arxiv.org/pdf/2102.11517.pdf).

## Datasets

All parsed datasets are available at this [link](https://www.dropbox.com/sh/lha0oevqos6jxn9/AAAz3Xkql2aKwcnKmX3kt357a?dl=0).
The source of each dataset is listed below.
| Name          | Structure                                       | Size                   | # Non-zeros | Source   |
| ------------- |:-----------------------------------------------:| :---------------------:| :----------:| :-------:|
| Divvy Bikes   | sources x destinations x time (minutes)         | 673 x 673 x 525594     | 3.82M       | [Link](https://www.divvybikes.com/system-data) |
| Chicago Crime | communities x crime types x time (hours)        | 77 x 32 x 148464       | 5.33M       | [Link](http://frostt.io/) |
| New York Taxi | sources x destinations x time (seconds)         | 265 x 265 x 5184000    | 84.39M      | [Link](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |
| Ride Austin   | sources x destinations x colors x time (minutes)| 219 x 219 x 24 x 285136| 0.89M       | [Link](https://data.world/andytryba/rideaustin) |

## Requirements

* C/C++17 compiler
* CMake
* Git

## Used Libraries

* Eigen (<https://gitlab.com/libeigen/eigen>)
* yaml-cpp (<https://github.com/jbeder/yaml-cpp>)

## Download

```bash
git clone --recursive https://github.com/yunik1004/SliceNStitch-GD.git
```

## Generate Build System

To generate the build system, run the following command on your terminal:

```bash
cmake . -DCMAKE_BUILD_TYPE=RELEASE
```

After that, you can build the program using the build automation software (e.g. Make, Ninja).

## Execution

To test the algorithms in the paper, run the following command on your terminal:

```bash
./SliceNStitch [config_file]
```

## Config File

### Example config file

```yaml
# test.yaml
data:
    filePath: "nyt_2019.csv"  # Path of the data stream file
    nonTempDim: [265, 265]  # Non-temporal dimension
    tempDim: 5184000  # Temporal length of data stream

tensor:
    unitNum: 10  # The number of indices in the time mode (W)
    unitSize: 3600  # Period (T)
    rank: 20  # Rank of CPD (R)

algorithm:
    name: "SGD"
    settings:  # Details are described in the next section
        numSample: 100
        learningRate: 0.01

test:
    outputPath: "out.txt"  # Path of the output file
    startTime: 0  # Starting time of the input data to be processed
    numEpoch: 180000  # The number of epochs
    checkPeriod: 1  # Period of printing the error
    updatePeriod: 1  # Period of updating the tensor
    random: true  # Randomness of the initial points
```

### Examples of Possible Algorithms

```yaml
algorithm:
    name: "GD"
    settings:
        learningRate: 0.0001
```

```yaml
algorithm:
    name: "SGD"
    settings:
        numSample: 100
        learningRate: 0.01
```

```yaml
algorithm:
    name: "Momentum"
    settings:
        numSample: 100
        learningRate: 0.01
        momentum: 0.9
        momentumNew: 0.1
```

```yaml
algorithm:
    name: "RMSProp"
    settings:
        numSample: 100
        learningRate: 0.01
        decay: 0.9
```

```yaml
algorithm:
    name: "Adam"
    settings:
        numSample: 100
        learningRate: 0.001
        beta1: 0.9
        beta1New: 1.0
        beta2: 0.9
```

## Input & Output Format

Input (data:filePath in a config file) must be a CSV file that consists of a multi-aspect data stream.
Each row of the file is a single event and the file should be formatted as follows.

For a CSV file with N columns,

* First (N-2) columns represent the non-temporal indices of events
* The (N-1)th column represents the time indices of events
* The last column represents the values of events

The output (test:outputPath) of the code will be:

```text
-----Test Results-----
RMSE    Fitness
0.33438 0.675436
0.329659 0.639608
...
0.37203 0.692263

-----The Total Number of Updates-----
4030721

-----Total Elapse Time-----
330.706 (sec)

-----Elapse Time per each Update-----
8.20463e-05 (sec)
```
