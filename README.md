
# DeepMAO-Deep Multi-scale Aware Overcomplete Network for Building Segmentation in Satellite Imagery

In this work, we propose to tackle the issue of detecting and segmenting small and complex-shaped buildings in Electro-Optical (EO) and SAR satellite imagery. A novel architecture Deep Multi-scale Aware Overcomplete Network (DeepMAO), is proposed that comprises an overcomplete branch that focuses on fine structural features and an undercomplete (U-Net) branch tasked to focus on coarse, semantic-rich features. Additionally, a novel self-regulating augmentation strategy, “Loss-Mix,” is proposed to increase pixel representation of misclassified pixels.

![DeepMAO architecture](https://github.com/Sumanth181099/DeepMAO/blob/main/pngs/git_deepmao.png)

![Sample Results](https://github.com/Sumanth181099/DeepMAO/blob/main/pngs/git_deepmao_results.png)

We advise you to use conda environment to run the package. Run the following command to install all the necessary modules:

```sh
conda env create -f environment.yml 
conda activate solaris_new
```
You can download the MSAW(Multi-Sensor All Weather Mapping) dataset [here](https://spacenet.ai/sn6-challenge/).
Update the data paths in the 'DeepMAO.py' in the command argparser. 
Our training and validation split is based of the official SN6 challenge [repo](https://github.com/SpaceNetChallenge/SpaceNet_SAR_Buildings_Solutions/tree/master/1-zbigniewwojna).  To be precise, you can find the validation files in the [val_masks.csv file](https://drive.google.com/file/d/1pccsbwxtUeJoLCKgkTAxrYs6wLAi7mQo/view?usp=sharing).
To train the model from scratch, run the following command:
```sh
./train.sh
```
To get an additional bump on the result, run the post-processing file using:
```sh
./postprocessing.sh
```
To only run the inference, use the ckpt provided [here](https://drive.google.com/drive/folders/111UQvS-vkjjRRCdkzZHGUTlhY9HDKHlc?usp=sharing). In the train.sh file, remove the '--train' parameter. The provided checkpoint is only for EO modality of MSAW dataset.
To view the predictions, run the following commands:
```sh
mkdir plots_of_predictions
python visual.py
```
## Citation

If you find this repo useful for your work, please cite our paper:

```shell
@inproceedings{sikdar2023deepmao,
  title={DeepMAO: Deep Multi-Scale Aware Overcomplete Network for Building Segmentation in Satellite Imagery},
  author={Sikdar, Aniruddh and Udupa, Sumanth and Gurunath, Prajwal and Sundaram, Suresh},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={487--496},
  year={2023}
}
```



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This repo is based on [Spacenet6 challenge](https://github.com/SpaceNetChallenge/SpaceNet_SAR_Buildings_Solutions/tree/master/1-zbigniewwojna).



