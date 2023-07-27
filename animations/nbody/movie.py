import os
import shutil
import yaml
import argparse


def movie(fps: int,
          remove_frames: bool,
          frames_dir: str = "animations/nbody/movies/frames/",
          output_dir: str = "animations/nbody/movies/",
          filename: str = "movie") -> None:
    """
    Create a movie from the images in `frames_dir`.

    Parameters
    ----------
    fps : int
        The frames per second of the movie.
    remove_frames : bool
        If True, remove frame directory.
    frames_dir : str, optional
        The directory of the images, by default "movies/frames/".
    output_dir : str, optional
        The directory to save the output file, by default "movies/".
    filename : str, optional
        The name of the output file, by default "movie".
    """
    os.system(
        f"ffmpeg -r {fps} -i {frames_dir}frame%01d.png "
        f"-vcodec mpeg4 -y {output_dir}{filename}.mp4")
    if remove_frames:
        shutil.rmtree(frames_dir)


def main():
    # Get simulation file name
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation",
                        type=str,
                        required=True,
                        help="The name of the simulation.")
    parser.add_argument("--delete_frames",
                        type=str,
                        choices=["yes", "no"],
                        required=True,
                        help="The name of the simulation.")
    args = parser.parse_args()

    # Read configuration file
    config = yaml.safe_load(
        open(f"animations/nbody/configs/{args.simulation}.yml"))

    # Create movie
    movie(fps=config["fps"],
          remove_frames=args.delete_frames == "yes",
          filename=config["filename"])


if __name__ == "__main__":
    main()
