# Neural Network Based Handwriting Recognition Tool

## Objective
To enable a machine learning-driven handwriting recognition application that can convert the handwritten script into a computerized legible script, and continuously learn in the process.

## About the Dataset
Dataset used was created by the National Institute of Standards and Technology (NIST). The NIST Special Database 19 consists of roughly 0.7 million sample png images. The current model has been trained only for uppercase letters (A-Z).
Dataset Link: https://www.nist.gov/srd/nist-special-database-19

## Preprocessing
While each character in the original dataset occupies 128x128 pixels per raster, in order to avoid heavy computation, first the size of the image was reduced to 56 x 56 pixels, and then further the canvas size was reduced to 28x28 pixels by removing the padding and thus creating a 784 feature configuration dataset. Each character is labeled sequentially from “A”- “Z”.

![Alt text](/assets/img/PreProcessing1.png?raw=true "")
Figure 2: (a) Original 128 x 128-pixel raster as obtained from the NIST Database, (b) 56 x 56- pixel raster upon resizing the image, (c) 28 x 28-pixel raster upon removing the padding from the resized image

The package ‘tkinter’ enables the programming of the user interface. The implementation of this package created a canvas platform where-in the user is able to enter their character or word inputs. The pop-up canvas will convert the image into a NumPy array, traveling column-wise through the canvas and looking for a filled pixel to mark the beginning of a letter. For words, the model continues to traverse and look for a column where there is significant relative blank space to mark the beginning of the second character. The tool is intelligent enough to differentiate a break in letters versus the beginning of a second letter.

![Alt text](/assets/img/PreProcessing2.png?raw=true "")

## Model Configuration
For this classification, Multi Layer Perceptron (MLP) classifier with Adam solver and sigmoid function has been used to achieve significant results.

## Run the tool
To run the GUI, kindly run the "final_code.py".

<strong>Note:</strong> To use the pre trained model in the GUI, "MLP_Adam_100_SelectedData_sigmoid_earlystop.sav", please ensure you have sklearn version 0.22.1 installed in Anaconda. Otherwise, kindly follow the below steps to run the tool:
1. Unzip the final_data_a_z.csv.zip.
2. Run the MLP_Classifier.py
3. Run the final_code.py to run the GUI.
