# Dataset

## DeepFashion
![Example images of DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/retrieval_inshop.png)
Download dataset in [here](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)

## CelebA
![Example images of CelebA](http://mmlab.ie.cuhk.edu.hk/projects/celeba/overview.png)

Download dataset in [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## Crawled

Crawled from sites of apparel brands with [crawler](https://github.com/shygiants/ChangeGAN/tree/master/crawler)

## TFRecord Builder

It converts images that has specified attributes to TFRecord format.

### Preliminary
* Download dataset you want
* Install [TensorFlow](https://www.tensorflow.org/install/).

### Usage

Execute following.
```bash
$ python builder.py \
--attributes=blond_hair \
--output_dir=/dir/output
```
It converts images that have attribute of `blond_hair`. Its format of output file is `blond_hair-00000-of-00032`.

If you want to build multiple datasets, you can use comma separated list of attributes in `--attributes`.
```bash
$ python builder.py \
--attributes=clothes, models \
--output_dir=/dir/output
```
It builds two datasets of `clothes` and `models` respectively.


Each arguments represents as follows

#### attributes
There are a set of attributes available.
 
Currently, following is predefined
 
* **CelebA**
  * blond_hair
  * black_hair
  * male
  * female
* **DeepFashion + Crawled**
  * clothes
  * models
  
If you want to add more attributes, edit `_PREDEFINED_ATTR` and request pull. 

#### output_dir
Directory to save TFRecord files