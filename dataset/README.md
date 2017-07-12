# Dataset

## DeepFashion
![Example images of DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/retrieval_inshop.png)
Download dataset in [here](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)

## CelebA
![Example images of CelebA](http://mmlab.ie.cuhk.edu.hk/projects/celeba/overview.png)

Download dataset in [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## TFRecord Builder

It converts images that has specified attributes to TFRecord format.

### Preliminary
* Download [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
* Install [TensorFlow](https://www.tensorflow.org/install/).

### Usage
Change directory to `celeba`

Execute following.
```bash
$ python builder.py \
--attributes=blond_hair \
--attributes_file=/celebA/list_attr_celeba.txt \
--image_dir=/dir/images \
--output_dir=/dir/output
```
It converts images that have attribute of `blond_hair`. Its format of output file is `blond_hair-00000-of-00032`.

If you want to build multiple datasets, you can use comma separated list of attributes in `--attributes`.
```bash
$ python builder.py \
--attributes=blond_hair,black_hair \
--attributes_file=/celebA/list_attr_celeba.txt \
--image_dir=/dir/images \
--output_dir=/dir/output
```
It builds two datasets of `black_hair` and `blond_hair` respectively.


Each arguments represents as follows
##### attributes
 There are a set of attributes available.
 
 Currently, following is predefined
  * blond_hair
  * black_hair
  * male
  * female
  
 If you want to add more attributes, edit `_PREDEFINED_ATTR` in `selector.py` and request pull. 
##### attributes_file
Path of file that has attributes annotations

It's included in dataset.
##### image_dir
Directory of images
##### output_dir
Directory to save TFRecord files