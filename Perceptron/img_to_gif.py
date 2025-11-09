import os
from PIL import Image

def images_to_gif(folder_path, output_path="output.gif", duration=200, loop=0):
    """
    Create an animated GIF from all images in a folder.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing image files.
    output_path : str
        Path (including filename) for the resulting GIF.
    duration : int
        Duration of each frame in milliseconds.
    loop : int
        Number of times the GIF should loop. 0 = infinite.
    """
    # Allowed image file extensions
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"}

    # Collect and sort image file paths
    files = sorted(
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f.lower())[1] in exts
    )

    if not files:
        raise ValueError("No image files found in the given folder.")

    # Open images
    frames = [Image.open(f) for f in files]

    # Optionally ensure all frames are the same size as the first
    first_frame = frames[0]
    frames = [im.resize(first_frame.size) for im in frames]

    # Save as GIF
    first_frame.save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
    )

    return output_path

# Example usage
# if __name__ == "__main__":
#     folder = r"/path/to/your/image/folder"
#     gif_path = r"/path/to/save/animation.gif"
#     images_to_gif(folder, gif_path, duration=150, loop=0)
#     print(f"GIF saved to: {gif_path}")
