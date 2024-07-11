# Zocket-Product-Photography-with-generative-A.I

Instructions and flow please refer bwlo doc link

[Doc for the run the product pipeline](https://docs.google.com/document/d/1UIt0CoKVBC8GZNI6NjhI45pRJ82lFrrtpRpLLYf2a_Q/edit?usp=sharing)

The code you've provided for a background change generator using the FastSAM model and the Stable Diffusion method. Let's break down the different components of the code and explain its functionality:

1. Import Statements:
   - The script starts with a series of import statements to bring in various Python libraries and modules, including image processing libraries (PIL, OpenCV), machine learning libraries (PyTorch), and other utilities.

2. Class Definition: BackgroundChangeGenerator
   - This class appears to encapsulate the functionality for generating background changes. It contains several methods for different tasks related to this process.

3. Overlay Mask:
   - `overlay_mask` is a method that takes an original image and a mask image, and it overlays the mask on the original image with a specified opacity.

4. Get Masks:
   - `get_masks` method processes annotations to obtain masks and create a unified mask image.

5. Get Image:
   - `get_image` method seems to extract and process an image from annotations.

6. Generate Background Change:
   - The class contains two methods for generating background changes, one using an API (`generate_background_change_api`) and another using a custom pipeline (`generate_background_change_custom`). These methods take a prompt, an image, a mask, and some other parameters to produce a background-changed image.

7. File Handling and User Interaction:
   - The class also has methods to handle file selection and drawing rectangles on images.

8. Concatenating Images:
   - `get_concat_h_multi_resize` method takes a list of images and concatenates them horizontally while resizing them to have the same height.

9. Mask the Region:
   - `mask_the_region` method seems to be used for interactively selecting a region in an image, processing it with a model, and displaying the results.

In the main part of the script:
- An instance of the `BackgroundChangeGenerator` class is created.
- The FastSAM model is loaded.
- An image, annotated image, and mask are obtained by selecting a region in the input image.
- The user is prompted to enter a prompt for background change.
- The script generates a background change using the `generate_background_change_api` method and displays the result.

The script also includes an option for using the `generate_background_change_custom` method, which is a GPU-intensive operation and requires certain GPU capabilities.

Results of the prompt is below:

prompt :  Change background to hotel

![myimage-alt-tag](https://github.com/rakshit176/Zocket-Product-Photography-with-generative-A.I/blob/main/results.png)

In summary, this script provides an interactive way to select a region in an image, apply a background change using either an API or a custom method, and display the resulting image. It combines several libraries and models for this purpose. To use the script, you would need the required models and API access tokens. Additionally, you might need to install the necessary Python packages. The script is designed for background change tasks and can be useful in creative applications or image editing workflows.
