# Soft Agar Colony Counter
This program quantifies the colones grown in a 3D agar matrix from a greyscale image, by recurising through directories with images.
It outputs both the number of colonies in each image as well as its cross sectional area in a .csv. In addition it makes a smaller quality control image to visually verify performance

Since this program is much older than the beta-gal counter, you have to fine tune performance via the global variables in the top of the script itself

# Parameters  
*-m --magnification:* Microscope magnification - 4x or 10x. Default is 4x  
*-s --stained:* Stained -s flag will use an algorithm optimized for stained colonies. No flag uses the unstained algorithm  
*-i --input:* Directory where images are located. Defaults to current directory  
*-t --testing:* Will display intermediate QC images for fine tuning parameters.

# Required Packages  
**Python 2.7.14**  
- cv2 v3.4.0 (opencv)
- numpy v1.14.2
- matplotlib v2.2.3
- optparse v1.5.3