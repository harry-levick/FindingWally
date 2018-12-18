Finding Wally

I have used the [Mask-RCNN](https://github.com/matterport/Mask_RCNN) to solve the problem of finding Wally. On 
evaluation, my model does not perform all that well on the validation set. This could be due to the small training set,
and the general complexity of the images in the evaluation set. Overall this model performs well on images similar to
the ones provided in the training set, although extremely complex images it struggles on.

![](datasets/train/23.jpg)
![](results/tmpdznyosjd.png)

## Data
I have used a dataset found online, because the data was more accurately mapped, this improvement was seen because I 
initially created my own train data set. To create a new dataset, use [via-via](http://www.robots.ox.ac.uk/~vgg/software/via/via.html)
tool to annotate the images with masks.

## Training
To train this model, we use transfer learning. By initially loading the coco weights, we then build on these weights when
training our model. This helps to speed up the training process, as we can just build on our previous knowledge (coco weights).

```bash

python3 src/wally.py train --dataset=datasets/wally --weights=coco

```

To train this model, I have used [Floydhub](https://www.floydhub.com). Initially uploading my dataset to my account, then
using the following command to train my model on Floydhub:

```bash

floyd run --gpu --data <your-account-name>/datasets/wally/1:datasets "python3 src/wally.py train --dataset=/datasets/wally --weights=coco"

```

## Evaluation
In order to then use this model to detect wally on a new image, use the command

```bash

python3 src/find_wally.py <your-image>

```