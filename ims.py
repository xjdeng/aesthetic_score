from Easy_Image import imagesearch
try:
    from . import score_image
except ImportError:
    import score_image
default_columns = ['file','mtime','timestamp','score']
import time
from PIL import Image 
Image.MAX_IMAGE_PIXELS = 10000000000000

    
def calc_score(f, fpath, mtime):
    try:
        score = score_image.run(fpath)
    except IOError:
        score = 0
    return [[fpath, mtime, time.time(), score]]

def run(start = "./", outfile = "image_score.csv", batch = 10000):
    imagesearch.run_meta(calc_score, default_columns, outfile, start, batch)