# Color2GrayImgConversion

Implementation of color-to-gray image conversion using salient colors and radial basis functions.

## Short Description

This method employs quantization of an image’s salient colors combined with radial basis functions (RBFs) to convert color images to grayscale. It optimizes contrast retention by mapping a small set of dominant colors, identified through k-means clustering, to corresponding grayscale intensities. This ensures the preservation of important visual contrasts in the resultant grayscale images, making it effective for applications where contrast fidelity is crucial.

Additionally, this method adapts differently when converting natural versus synthetic images. For natural images, which often have a wider range of colors and subtler gradients, the process focuses on preserving the richness and depth of the original scene. For synthetic images, which typically feature more defined and fewer colors, the conversion emphasizes clarity and accuracy in replicating the distinct colors and sharp contrasts. Examples of conversions for both natural and synthetic images are provided below.

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
./ZhangWan24 [<input_image>] [<max_k>] [<sigma>] [<ordering>]
```

- ```input_image``` is color image input for conversion,
- ```max_k``` is maximum number of quantized colors (clusters),
- ```sigma``` controls the spread of the Laplace kernel's influence. All colour values are normalised to [0,1] as in the paper, so sigma is on that scale too (e.g. 0.1; the paper gives no value).
- ```ordering``` (optional) selects how gray values are assigned to the quantized colors: 1 = by rgb2gray value (paper Sec. 3.2.1), 2 = by weighted Lab distance (paper Sec. 3.2.2, default).

## Examples Of The Conversion

![Natural image before conversion](https://github.com/NovakovaMaria/Color2GrayImgConversion/blob/main/results/natural/parots_sigma25/parots.png)
![Natural image after conversion](https://github.com/NovakovaMaria/Color2GrayImgConversion/blob/main/results/natural/parots_sigma25/gray_withstep3.png)

![Synthetic image before conversion](https://github.com/NovakovaMaria/Color2GrayImgConversion/blob/main/results/synthetic/geometrypalete_sigma45/geometrypalete.png)
![Synthetic image after conversion](https://github.com/NovakovaMaria/Color2GrayImgConversion/blob/main/results/synthetic/geometrypalete_sigma45/gray_withstep3.png)
