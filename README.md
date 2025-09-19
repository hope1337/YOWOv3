# YOWOv3: An Efficient and Generalized Framework for Human Action Detection and Recognition

This is an implementation of paper : [YOWOv3: An Efficient and Generalized Framework for Human Action Detection and Recognition](https://arxiv.org/abs/2408.02623).

# Newsest update
Hi, I have just gotten some free time for this project.
Since many people have encountered issues with the model's performance on custom datasets, I'm skeptical about its true capabilities. After dissecting the AVA and UCF101-24 datasets, it seems they have some problems—for example, the distribution between the validation and training sets is too similar, and there's a lack of class balance. I'm currently working on gaining deeper insights and making modifications to improve the model's robustness. You can help me by giving your insights or statistics of your experiments, it would be really helpful and I truly appreciate your support.

I'm working on adding new features based on the feedback in #issues—stay tuned for updates!
Thank you all, with all my heart <3

---
## Preface 

Hello, thank you everyone for your attention to this study. If you find it valuable, please consider leaving a star, as it would greatly encourage me.

If you intend to use this repository for your own research, please consider to cite:
```latex
@misc{dang2024yowov3efficientgeneralizedframework,
      title={YOWOv3: An Efficient and Generalized Framework for Human Action Detection and Recognition}, 
      author={Duc Manh Nguyen Dang and Viet Hang Duong and Jia Ching Wang and Nhan Bui Duc},
      year={2024},
      eprint={2408.02623},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.02623}, 
}
```
---
## About asking question
I am very pleased that everyone has shown interest in this project. There are many questions being raised, and I am more than willing to answer them as soon as possible. However, if you have any doubts about the code or related matters, please provide me with context (config file, some samples that you couldn't detect, a checkpoint, etc.). Also, please use **English**.


---
## Structure of Instruction

In this Instruction, I will divide it into smaller sections, with each section serving a specific purpose. I will provide a summary of this Instruction structure in order right below. Please read carefully to locate the information you are looking for.

* **Preparation**: Environment setup and dataset preparation guide. 
* **Basic Usage**: Guide on using the pre-existing train, evaluate, detect, and live code in the repository.
* **Customization**: Guide on customizing datasets and components within the model.
* **Pretrained Resources**: Pretrained resources for 2D backbone, 3D backbone, and model checkpoints.
* **Limitations and Future Development**: In this section, I will outline the limitations that I have identified in this project. I did not have enough time and resources to experiment and do everything, so I will leave it for other research in the future.
* **Some Notes**: I will add some points that I think may cause difficulties or misunderstandings in using the repository. I will also update it based on the issues created.
* **References**: This section cites the repositories that I primarily relied on throughout the development of this project. These repositories have been incredibly helpful, and I am very grateful to the authors for providing them to the research community.
 


---
## Preparation
### Environment setup 
Clone this repository
```cpp 
git clone https://github.com/AakiraOtok/YOWOv3.git
```

**Update**: I found that reconstructing env with conda is easier to do, thus I provided an ```environment.yml``` file so that you can use ```conda env create -f environment.yml``` to reconstruct the env <3 (without doing the remaining steps below. Btw, I tested it and it worked well on ubuntu but not with window).

Use Python 3.8 or Python 3.9, and then download the dependencies:

```powershell
pip install -r requirements.txt
```

**Note**:
On my system, I use Python 3.7 with slightly different dependencies, specifically for torch:

```python
torch==1.13.1+cu117
torchaudio==0.13.1+cu117
torchvision==0.14.1+cu117
```
However, when testing on another system, it seems that these versions have been deprecated. I have updated the requirements.txt file and tested it again on systems using Python 3.8 and Python 3.9, and everything seems to be working fine. If you encounter any errors during the environment setup, please try asking in the "issues" section. Perhaps someone has faced a similar issue and has already found a solution.


### Datasets

#### UCF101-24
- Download from (as in YOWOv2): https://drive.google.com/file/d/1Dwh90pRi7uGkH5qLRjQIFiEmMJrAog5J/view

#### AVAv2.2 
- Follow the instructions at: https://github.com/yjh0410/AVA_Dataset.
- If you find that video downloading takes too long, you can consider downloading them from my Hugging Face repository and then proceed with the remaining steps as instructed above: https://huggingface.co/datasets/manh6054/avav2.2_train_set/tree/main.


---

## Basic Usage

### About config file
The project is designed in such a way that almost every configuration can be adjusted through the config file. In the repository, I have provided two sample config files: ucf_config.yaml and ava_config.yaml for the UCF101-24 and AVAv2.2 datasets, respectively. The **Basic Usage** section will not involve extensive modifications of the config file, while the customization of the config will be covered in the **Customization** section.

<span style="color:red">
    
**Warning!**: Since all configurations are closely related to the config file, please carefully read the part **Modify Config file** in the **Customization** section to be able to use the config file correctly.
    
</span>

### Simple command line

We have the following command template:
```powershell
python main.py --mode [mode] --config [config_file_path]
```
Or the shorthand version:

```powershell
python main.py -m [mode] -cf [config_file_path]
```

For ```[mode] = {train, eval, detect, live, onnx}``` for training, evaluation, detection (visualization on the current dataset), live (camera usage) or export to onnx and inference respectively. The```[config_file_path]``` is the path to the config file.

Example of training a model on UCF101-24:
```powershell
python main.py --mode train --config config/ucf_config.yaml
```
Or try evaluating a model on AVAv2.2:
```powershell
python main.py -m eval -cf config/ava_config.yaml
```


---

## Customization

### Modify Config file

There are some notes about the config file:

- If you want to train from scratch, set **pretrain_path = null**. Otherwise, if you want to use pretrain model, please set **pretrain_path** to the path of the checkpoint file.
- Because the project is developed over time, some options that were not present in the past might be added later, such as **image_size**. Some config files in old checkpoint folder may lack these options and cause errors. You just need to add the missing options. I have prepared two sample config files (with all options) in the config folder of this project. Please refer to them for corrections.
- All options used must remain consistent during both training and testing. For example, if the **image_size** option is set to **224** during training, changing it to **512** during inference will severely impact performance. Similarly, if you set **backbone3D = i3d** during training but infer using **resnext101**, the model will still run but with warnings and, of course, the detection results will be incorrect.
- The config file is automatically loaded through the ```build_config``` function in ```utils/build_config.py```.

```python
import yaml

def build_config(config_file='config/ucf_config.yaml'):
    with open(config_file, "r") as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    if config['active_checker']:
        pass
    
    return config
```

## Custom Dataset

I know why you are here, my friend =)))))

You can build a custom dataset for yourself, however, make sure to carefully read the notes below to do it correctly.

Firstly, every time you want to use a dataset, simply call the ```build_dataset``` function as shown in the example code:

```python
    dataset = build_dataset(config, phase='train')
```

The ```build_dataset``` function is defined in ```datasets/build_dataset.py``` as follows:

 

```python
from datasets.ucf.load_data import build_ucf_dataset
from datasets.ava.load_data import build_ava_dataset

def build_dataset(config, phase):
    dataset = config['dataset']

    if dataset == 'ucf':
        return build_ucf_dataset(config, phase)
    elif dataset == 'ava':
        return build_ava_dataset(config, phase)
```

To accommodate your needs, you simply need to define the ```build_custom_dataset``` function for your specific purpose and modify the above ```build_dataset``` function accordingly.

The model is generalized to train on **multi-action** datasets, meaning that each box may have multiple actions simultaneously. However, metrics for **one box - one action** are more common than **one box - multi-action**. Therefore, I will guide you on evaluation as **one box - one action**, while training as **one box - multi-action** for generalization.

The ```build_dataset``` function returns custom-defined dataset classes. There are two important parameters to consider: ```config``` and ```phase```. ```config``` is a dictionary containing options as in the config file (loaded beforehand), with nothing particularly special. On the other hand, ```phase``` has two values: **train** or **test**. **train** is used for training, and **test** is used for detection/evaluation/live stages.

Let:

- $C$: the number of channels
- $D$: the number of frames (default = 16)
- $H$, $W$: the height and width of the image
- $N$: the number of truth boxes
- $num\\_class$: the number of classes


You need to return:

- **clip**: a tensor with shape $[C, D, H, W]$ representing the clip to be detected.

- **boxes**: a tensor with shape $[N, 4]$, containing the coordinates x_top_left, y_top_left, x_bottom_right, y_bottom_right of the truth boxes (normalized in range $0 ... 1$).
- **labels**: 
    - For **phase = train**, return a tensor with shape $[N, num\\_class]$ where num_class is a vector with the $i$ th element being $1$ if class $i$ is present and $0$ otherwise. 
    - For **phase = test**, return a tensor with shape $[N]$, where $i$ th element is the class index of the $i$ th truth box.

- Additionally, to use detect.py, the get_item function also needs an additional parameter get_origin_image. If this parameter is set to True, it should return the original unaltered image.

Please note that index class start at $0$.

To evaluate, use ```ucf_eval.py```.

---

## Pretrained Resources

All pre-trained models for backbone2D, backbone3D and model checkpoints are publicly available on [my Hugging Face repo](https://huggingface.co/manh6054/YOWOv3/tree/main).

Regarding the model checkpoints, I have consolidated them into an Excel file that looks like this:


![Untitled](https://github.com/user-attachments/assets/f47de4b7-2d35-4f22-9bb3-8aee59320867)


Each cell represents a model checkpoint, displaying information such as mAP, GLOPs, and # param in order. The checkpoints are stored as folders named after the corresponding cells in the Excel file (e.g., O27, N23, ...). Each folder contains the respective config file used for training that model. Please note that both the regular checkpoint and the exponential moving average (EMA) version of the model are saved.

<span style="color:red">
    
**Warning!**: Since all configurations are closely related to the config file, please carefully read the part **Modify Config file** in the **Customization** section to be able to use the config file correctly.
    
</span>

## Limitations and Future Development

- The architecture of YOWOv3 does not differ much from YOWOv2, although initially I had planned to try a few things. Firstly, the feature map from the 3D branch does not go through path aggregation but merges with the 2D branch and is used by the model for predictions directly. This makes the architecture look quite simple, and I believe it will have a significant impact on performance. Another thing is that in [this](https://arxiv.org/abs/2108.07755) paper, the authors propose an alternative method to replace the decoupled head called Task-aligned head, which can avoid repeating attention modules in YOWOv3 and make the model much lighter.

- The project was developed gradually through stages and gradually expanded, so there are still some areas that are not comprehensive enough. For example, evaluating on AVA v2.2 took a lot of time because I did not parallelize this process (batch_size = 1). The reason for this is that the format required by AVA v2.2 demands an additional identity for the bounding box, which I did not set up in the initial evaluation code as it was only serving experiments on UCF101-24 at that time.



---

## Some notes:
- In the commit history, there are commits named after the best checkpoints. The code in that commit is what I used to train the model, but it's not the config file! This is because during the training process, I opened another terminal window to experiment with a few things, so the config file changed and I didn't revert it back to the original. The original configs are saved in my notes, not in the commit files.


---

## References

I would like to express my sincere gratitude to the following amazing repositories/codes, which were the primary sources I heavily relied on and borrowed code from during the development of this project:

- [A neat implementation of YOLOv8 using PyTorch](https://github.com/jahongir7174/YOLOv8-pt)
- [3D CNN backbones: MobileNet/v2, ShuffleNet/v2, ResNet, ResNeXt](https://github.com/okankop/Efficient-3DCNNs)
- [Implementation of the I3D model on PyTorch](https://github.com/piergiaj/pytorch-i3d)
- [YOWO model](https://github.com/wei-tim/YOWO?tab=readme-ov-file)
- [YOWOv2 model](https://github.com/yjh0410/YOWOv2)
- [AVAv2.2 evaluation code from the organizers](https://github.com/activitynet/ActivityNet/tree/master/Evaluation)
