# stanford_bunny
This project seeks to create a two-dimensional image that looks three-dimensional using python and OpenCV.

## Background
The main inspiration stems from Matthew Gamber's piece *Stanford Bunny*, a continuation of his *Any Color You Like* series where he photographs items in color against a white background, convert the photos to black and white negatives, and prints them without color as a way to test color vision. His *Stanford Bunny* is created in a way to evaluate three-dimensionality. 

## Process
This program takes an input photo of a subject against a plain background and produces a binary outline and mask of the subject. The outline and mask are then used to perform a series of alpha blending and warping to eventually produce the metallic overlay and shadow. The isolated subject is centered and resized on a new canvas and blended with the generated overlay and shadow to produce the final result.

### removeNoise(image)
Applies a Gaussian blur to remove any noise that may produce undesired edges

### apply_binary(image)
Reduced-noise image is put through an adaptive threshold to isolate the subject from its background. For this function, we used Adaptive Gaussian Thresholding after experimenting with different edge detection and image segmentation techniques.

### make_mask(image, threshold)
Created subject's binary mask by using the altered image and a threshold value set to 232; this function creates a kernel and uses OpenCV's morphologyEx function with its second param set to MORPH_CLOSE to fill any holes present in the mask

### make_metallic(metal, outline, mask, alpha=0.9, gamma=1.5)
Alpha-blends a grayscale texture image (for the purpose of this project we chose a metallic texture), the outline created with the apply-binary function, and the binary mask created with the binary_mask function.

       **gamma_correction(image, correction)**
       Called in the make_metallic function; it takes the alpha-blended image and a treshold value to output a new contrast-adjusted result 

### generate_shadow(image, mask, x=0, y=5)
Generates a shadow using the binary mask. This function uses a Gaussian blur and OpenCV's warpAffine function to produce and warp the shadow so it can be translated and fitted to the original subject.

### create_canvas(image)
Creates a blank canvas with set height and width (600x800) and scales the subject to a width of 300, keeping the aspect ratio the same

### create_color_overlay(b, g, r) and alpha_blend(image1, image2, alpha)
Color overlay is created and alpha-blended with the canvas image to produce the effect of a 3D object that has been washed over with a coat of paint
