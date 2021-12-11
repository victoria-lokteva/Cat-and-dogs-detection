import click
from skimage.io import imread
import os
from numpy import random
from PIL import Image

dataset_path = '/home/wildkatze/cats_dogs_dataset'


@click.group()
def cli():
    pass


@cli.command()
@click.argument('name')
def listimages(name):
    """List the dataset content"""
    if name == 'train':
        p = os.path.join(dataset_path, 'train')
    elif name == 'test':
        p = os.path.join(dataset_path, 'train')
    lst = os.listdir(p)
    images = [image for image in lst if '.jpg' in image]
    files = [file for file in lst if '.txt' in file]
    click.echo("Images: ")
    click.echo(images)
    click.echo("Files with description: ")
    click.echo(files)


@cli.command()
@click.argument('name')
def imageshape(name):
    """Print the shape of the image"""
    p = os.path.join(dataset_path, 'train')
    im = os.path.join(p, name)
    image = imread(im)
    click.echo(image.shape)


@cli.command()
@click.argument('name')
def printimage(name):
    """Show the image from dataset"""
    p = os.path.join(dataset_path, 'train')
    im = os.path.join(p, name)
    img = Image.open(im)
    img.show()


@cli.command()
def randomimage():
    """Show a random image from dataset"""
    p = os.path.join(dataset_path, 'train')
    lst = os.listdir(p)
    images = [image for image in lst if '.jpg' in image]
    n = len(images) - 1
    ind = random.randint(n)
    name = images[ind]
    im = os.path.join(p, name)
    img = Image.open(im)
    img.show()


if __name__ == "__main__":
    cli()
