# ChangeGAN

## ChangeGAN

![Abstract image of ChangeGAN](https://github.com/shygiants/ChangeGAN/blob/master/static/abstract.png)

Change part of image with other image. For instance, change clothes that a model wearing with other clothes.

## AutoConverter

Autoencoder + DiscoGAN

Validate if Autoencoder and DiscoGAN can share encoders and decoders.

Domain A: Black hair
Domain B: Blond hair

### Usage

There are two options.

1. Run on Google CloudML (Recommended)
1. Run locally

#### Google CloudML (Recommended)

To use CloudML, you need to install [gcloud](https://cloud.google.com/ml-engine/docs/quickstarts/command-line).

You need to create a project on Google Cloud and edit `train.sh`.

Submit training job to CloudML.
```bash
$ bash train.sh
```

It requests the configuration as specified in `config.yaml`.

If you have preferable configuration, edit `config.yaml` and execute `train.sh`.

#### Local

Change directory to `change-gan`.

```bash
$ python main.py \
--dataset-dir /dataset/dir \
--job-dir /job/dir
```