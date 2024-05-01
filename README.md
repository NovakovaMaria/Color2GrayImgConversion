# Color2GrayImgConversion

Implementation of color-to-gray image conversion using salient colors and radial basis functions.

## Short Description

This method employs quantization of an image’s salient colors combined with radial basis functions (RBFs) to convert color images to grayscale. It optimizes contrast retention by mapping a small set of dominant colors, identified through k-means clustering, to corresponding grayscale intensities. This ensures the preservation of important visual contrasts in the resultant grayscale images, making it effective for applications where contrast fidelity is crucial.

**Three main steps** of this color-to-gray image conversion:

    1. Quantization Process
    2. Assigning Gray Values
    3. Final Image Rendering

## Acknowledgements

 - [Implemented according to the paper by Zhang L., Wan Y.; published Feb. 22, 2024](https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-33/issue-1/013047/Color-to-gray-image-conversion-using-salient-colors-and-radial/10.1117/1.JEI.33.1.013047.full#_=_)

## Usage

### Compilation

```
make
```

### Execution

```
./ZhangWan24 [<input_image>] [<max_k>] [<sigma>] 
```

- ```input_image``` is color image input for conversion,
- ```max_k``` is maximum number of quantized colors (clusters),
- ```sigma``` controls the spread of the Laplace kernel's influence.

## Examples Of The Conversion

![Natural image before conversion](https://github.com/NovakovaMaria/Color2GrayImgConversion/blob/main/results/natural/parots_sigma25/parots.png)
![Natural image after conversion](https://github.com/NovakovaMaria/Color2GrayImgConversion/blob/main/results/natural/parots_sigma25/gray_withstep3.png)

![Synthetic image before conversion](https://github.com/NovakovaMaria/Color2GrayImgConversion/blob/main/results/synthetic/geometrypalete_sigma45/geometrypalete.png)
![Synthetic image after conversion](https://github.com/NovakovaMaria/Color2GrayImgConversion/blob/main/results/synthetic/geometrypalete_sigma45/gray_withstep3.png)
