# Synthetic Faces High Quality (SFHQ) dataset  

![SFHQ dataset sample images](https://raw.githubusercontent.com/SelfishGene/SFHQ-dataset/main/images/SFHQ_sample_4x8.jpg)

## **Update**: A new and higher quality dataset of synthetic face images: [SFHQ-T2I-dataset](https://github.com/SelfishGene/SFHQ-T2I-dataset)

The original SFHQ dataset consists of 4 parts, totaling ~425,000 curated high quality 1024x1024 synthetic face images.  
It was created by "bringing to life" and turning to photorealistic face images from multiple "inspiration" sources (paintings, drawings, 3D models, text to image generators, etc) using a process similar to what is described [in this short twitter thread](https://twitter.com/DavidBeniaguev/status/1376020024511627273?s=20&t=kH9J5mV9hL8e3y8PruuB5Q). The process involves encoding the images into StyleGAN2 latent space and performing a small manipulation that turns each image into a photo-realistic image. These resulting candidate images are then further curated using a semi-manual semi-automatic process with the help of the lightweight [visual taste aprroximator](https://github.com/SelfishGene/visual_taste_approximator) tool

The dataset also contains facial landmarks (an extended set of 110 landmark points) and face parsing semantic segmentation maps. An example script (`explore_dataset.py`) is provided ([live kaggle notebook here](https://www.kaggle.com/code/selfishgene/explore-synthetic-faces-hq-sfhq-dataset)) and demonstrates how to access landmarks, segmentation maps, and textually search withing the dataset (with CLIP image/text feature vectors), and also performs some exploratory analysis of the dataset.

Example illustation of landmarks and segmentation maps below:  
![SFHQ dataset landmarks and segmentation](https://raw.githubusercontent.com/SelfishGene/SFHQ-dataset/main/images/SFHQ_sample_landmarks_segmentation.jpg)

## Download
The dataset can be downloaded via kaggle:
- [Part 1](https://www.kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-1) consists of 89,785 HQ 1024x1024 curated face images. It uses "inspiration" images from [Artstation-Artistic-face-HQ dataset (AAHQ)](https://github.com/onion-liu/aahq-dataset), [Close-Up Humans dataset](https://opensynthetics.com/dataset/close-up-humans-dataset-by-synthesis-ai/) and [UIBVFED dataset](http://ugivia.uib.es/uibvfed/).  
- [Part 2](https://www.kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-2) consists of 91,361 HQ 1024x1024 curated face images. It uses "inspiration" images from [Face Synthetics dataset](https://github.com/microsoft/FaceSynthetics) and by sampling from the [Stable Diffusion v1.4](https://github.com/CompVis/stable-diffusion) text to image generator using varied face portrait prompts. 
- [Part 3](https://www.kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-3) consists of 118,358 HQ 1024x1024 curated face images. It uses "inspiration" images by sampling from [StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch) mapping network with very high truncation psi coefficients to increase diversity of the generation. Here, the [e4e](https://github.com/omertov/encoder4editing) encoder is basically used a new kind of truncation trick.
- [Part 4](https://www.kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-4) consists of 125,754 HQ 1024x1024 curated face images. It uses "inspiration" images by sampling from the [Stable Diffusion v2.1](https://github.com/Stability-AI/stablediffusion) text to image generator using varied face portrait prompts. 


## More Details about dataset generation and collection
1. The original inspiration images are taken from:
    - [Artstation-Artistic-face-HQ Dataset (AAHQ)](https://github.com/onion-liu/aahq-dataset) which contains mainly painting, drawing and 3D models of faces (part 1)
    - [Close-Up Humans Dataset](https://opensynthetics.com/dataset/close-up-humans-dataset-by-synthesis-ai/) that contains 3D models of faces (part 1)
    - [UIBVFED Dataset](http://ugivia.uib.es/uibvfed/) that also contain 3D models of faces (part 1)
    - [Face Synthetics Dataset](https://github.com/microsoft/FaceSynthetics) which contains 3D models of faces (part 2)
    - generated images using [stable diffusion v1.4 model](https://github.com/CompVis/stable-diffusion) using various face portrait prompts that span a wide range of ethnicities, ages, expressions, hairstyles, etc. (part 2)
    - StyleGAN2 mapping network sampled with larger that 1 truncation psi values, and then using a new truncation trick in which instead of moving towards the average w_avg vector, we move towards the encoded w_e4e vector to correct the example. illustation is provided in the section below (part 3)
    - generated images using [stable diffusion v2.1 model](https://github.com/Stability-AI/stablediffusion) using various face portrait prompts that span a wide range of ethnicities, ages, expressions, hairstyles, etc. (part 4)
1. Each inspiration image was encoded by [encoder4editing (e4e)](https://github.com/omertov/encoder4editing) into [StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch) latent space (StyleGAN2 is a generative face model tained on [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset)) and multiple candidate images were generated from each inspiration image
1. These candidate images were then further curated and verified as being photo-realistic and high quality by a single human (me) and a machine learning assistant model that was trained to approximate my own human judgments and helped me scale myself to asses the quality of all images in the dataset. The code for the tool used for this purpuse [can be found here](https://github.com/SelfishGene/visual_taste_approximator)
1. Near duplicates and images that were too similar were removed using CLIP features (no two images in the dataset have CLIP similarity score of greater than ~0.92)
1. From each image various pre-trained features were extracted and provided here for convenience, in particular CLIP features for fast textual query of the dataset, the feaures are under `pretrained_features/` folder
1. From each image, semantic segmentation maps were extracted using [Face Parsing BiSeNet](https://github.com/zllrunning/face-parsing.PyTorch) and are provided in the dataset under under `segmentations/` folder
1. From each image, an extended landmark set was extracted that also contain inner and outer hairlines (these are unique landmarks that are usually not extracted by other algorithms). These landmarks were extracted using [Dlib](https://github.com/davisking/dlib), [Face Alignment](https://github.com/1adrianb/face-alignment) and some post processing of [Face Parsing BiSeNet](https://github.com/zllrunning/face-parsing.PyTorch) and are provided in the dataset under `landmarks/` folder
1. NOTE: semantic segmentation and landmarks were first calculated on scaled down version of 256x256 images, and then upscaled to 1024x1024

Example of dataset generation process on artistic illustations and paintings taken from [AAHQ](https://github.com/onion-liu/aahq-dataset) (part 1):  
![SFHQ dataset paintings](https://raw.githubusercontent.com/SelfishGene/SFHQ-dataset/main/images/bring_to_life_process_paintings.jpg)

Example of dataset generation process on 3D models taken from [Face Synthetics](https://github.com/microsoft/FaceSynthetics), [Close-Up Humans](https://opensynthetics.com/dataset/close-up-humans-dataset-by-synthesis-ai/), and [UIBVFED](http://ugivia.uib.es/uibvfed/) (parts 1 & 2):  
![SFHQ dataset 3D models](https://raw.githubusercontent.com/SelfishGene/SFHQ-dataset/main/images/bring_to_life_process_3D_models.jpg)

Example of dataset generation process of correcting faults in face images generated by [Stable Diffusion](https://github.com/CompVis/stable-diffusion) (parts 2 & 4):  
![SFHQ dataset stable diffusion](https://raw.githubusercontent.com/SelfishGene/SFHQ-dataset/main/images/bring_to_life_process_stable_diffusion.jpg)

Example of dataset generation process of using the StyleGAN2 mapping network samples with high truncation psi and correcting with e4e encoder (part 3):  
![SFHQ dataset encoder based truncation](https://i.ibb.co/NT7BJy5/Figure-1-brief-stages-psi-08-to-20-1.jpg)


## Demonstation of variability in the dataset 
we deomonstate the variability of the images in the dataset by textual query of the dataset with [CLIP](https://github.com/openai/CLIP) ViT-L/14@336 model embeddings (NOTE: these demonstation images are only of parts 1&2, the dataset with parts 3&4 is much more varied, please try out for yourself or check the [script on kaggle](https://www.kaggle.com/code/selfishgene/explore-synthetic-faces-hq-sfhq-dataset-2)):  
- Hair color:  
![SFHQ dataset variability haircolor](https://raw.githubusercontent.com/SelfishGene/SFHQ-dataset/main/images/SFHQ_variability_hair_color.jpg)

- Age:  
![SFHQ dataset variability age](https://raw.githubusercontent.com/SelfishGene/SFHQ-dataset/main/images/SFHQ_variability_age.jpg)

- Ethnicity:  
![SFHQ dataset variability ethnicity](https://raw.githubusercontent.com/SelfishGene/SFHQ-dataset/main/images/SFHQ_variability_ethnicity.jpg)

- Facial expression:  
![SFHQ dataset variability expression](https://raw.githubusercontent.com/SelfishGene/SFHQ-dataset/main/images/SFHQ_variability_expression.jpg)  

Additional variability demonstations can be found under `images/`

## Privacy
Since all images in this dataset are synthetically generated there are no privacy issues or license issues surrounding these images.  

## Summary
Overall the 4 parts of this dataset contain ~425,000 high quality and curated synthetic face images that have no privacy issues or license issues surrounding them.  

This dataset contains a high degree of variability on the axes of identity, ethnicity, age, pose, expression, lighting conditions, hair-style, hair-color, facial hair. It lacks variability in accessories axes such as hats or earphones as well as various jewelry. It also doesn't contain any occlusions except the self-occlusion of hair occluding the forehead, the ears and rarely the eyes. This dataset naturally inherits all the biases of it's original datasets (FFHQ, AAHQ, Close-Up Humans, Face Synthetics, LAION-5B) and the StyleGAN2 and Stable Diffusion models.  

The purpose of this dataset is to be of sufficiently high quality that new machine learning models can be trained using this data alone or provide meaningful augmentation to other data sources, including the training of generative face models such as StyleGAN. The dataset may be extended from time to time with additional supervision labels (e.g. text descriptions), but no promises.

Hope this is helpful to some of you, feel free to use as you see fit...

## Citation

```
@misc{david_beniaguev_2022_SFHQ,
	title={Synthetic Faces High Quality (SFHQ) dataset},
	url={https://github.com/SelfishGene/SFHQ-dataset},
	DOI={10.34740/kaggle/dsv/4737549},
	publisher={GitHub},
	author={David Beniaguev},
	year={2022}
}
```


