import subprocess
from PIL import Image
from pillow_heif import HeifImagePlugin

# Uses the EXIF data to get the year the image/video was taken. Returns the year
# as a string or None if not available.
def get_exif_creation_year(file):
    creation_year = None

    # Start by calling out to exiftool (most reliable and supporting)
    EXIFTOOL_CREATION_DATE_TAG = "Creation Date"
    EXIFTOOL_CREATE_DATE_TAG = "Create Date"
    process = subprocess.Popen(["exiftool", file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    lines = out.decode("utf-8").split("\n")
    for line in lines:
        if line.startswith(EXIFTOOL_CREATION_DATE_TAG):
            creation_year = line.split(":")[1].strip() # Prefer this entry
            break
        elif line.startswith(EXIFTOOL_CREATE_DATE_TAG) and creation_year is None:
            creation_year = line.split(":")[1].strip() # Fallback to this entry

    # If creation_year is still None, then try the Pillow library.
    if creation_year is None:
        image = Image.open(file)
        exif = image.getexif()
        if exif is not None:
            dt_original = exif.get(306)
            if dt_original is not None:
                creation_year = dt_original[0:4]
    
    return creation_year
