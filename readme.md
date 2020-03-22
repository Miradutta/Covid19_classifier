
# Covid19 Classifier

## Aim 
To detect and classify various types of pneumonias from chest X-ray using a Convolutional neural network

## Exploring the data

Dataset : https://github.com/ieee8023/covid-chestxray-dataset

Our image dataset is stored as .jpg files with labels stored in metadata.csv. We use ImageDataBunch.from_df
to load the data and assign the labels. Then we normalize our data based on the stats of the RGB channels from the ImageNet dataset

We have 6 classes 'ARDS', 'COVID-19', 'No Finding', 'Pneumocystis', 'SARS', 'Streptococcus'

## Training the model

We now use a pre-trained Resnet50 Convolutional Neural Net model and use transfer learning to leanrn the weights of only the last layer of the network

Initially we fine tune our model with size of 512 

We fit one cycle of 4 epochs to see how our model performs on this dataset as an attempt to benchmark. We had an error rate of approximately 44% which is 56% accuracy

We then unfreeze the weights of our entire network and plot a loss vs learning rate to find the proper learning rate. Then we fit one cyle with 5 more epochs where we get an error rate of 20%

Now that we have finetuned our model for images with resolution of 512, we now finetune it for 1024 res images. We fit one cycle with 5 epochs and get an error rate of 17% which is an accuracy of 83%

## Conclusion

There are a number of experiments we could further do to see if the accuracy can be improved, such as increasing the number of epochs, data augmentation techniques, etc.
