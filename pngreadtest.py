#%% this script looks to test the input formats of Pil and imageio

from PIL import Image
import imageio

file = "D:\\DAS\\FK\\DS3_x512_20250530T093645\\FK\\20220821T180027Z\\FK256_T0_X256_20220821T180027Z.png"

tmp1 = Image.open(file)
tmp2 = imageio.imread(file)

# %%
