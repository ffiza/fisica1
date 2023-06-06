import subprocess
import os


def get_files_with_extension(dir_path: str, ext: str) -> list:
    """
    Get a list of files with a given extension in a specified folder.

    Parameters
    ----------
    dir_path : str
        The path to the folder where the files are located.
    ext : str
        The file extension to filter the files.

    Returns
    -------
    files : list
        A list of file names that match the given extension.
    """
    files = [os.path.join(dir_path, my_file) for my_file
             in os.listdir(dir_path) if my_file.endswith(ext)]
    return files


def compile_tex_list(files: str) -> None:
    """
    Compile a list of LaTeX files into individual PDFs.

    Parameters
    ----------
    files : list
        A list with the paths to the LaTeX files.

    Returns
    -------
    bool
        True if the compilation was successful, False otherwise.
    """
    os.environ['TEXINPUTS'] = f'.:{"docs/tex/"}:'
    for my_file in files:
        try:
            for _ in range(2):
                # Compile two times to get the references correctly
                subprocess.run(['pdflatex', '-interaction=batchmode', my_file],
                               check=True)
            return True
        except subprocess.CalledProcessError:
            return False


def clean_aux_files(aux_exts: tuple) -> None:
    """
    Clean auxiliary files from the LaTeX compilation.

    Parameters
    ----------
    aux_exts : tuple
        A tuple with the auxiliary files extensions.
    """
    files = os.listdir(os.getcwd())
    for my_file in files:
        if my_file.endswith(aux_exts):
            os.remove(my_file)


def move_pdfs(source_path: str, target_path: str) -> None:
    """
    Move PDFs from a given path another path.

    Parameters
    ----------
    source_path : str
        The source path.
    target_path : str
        The destination path.
    """
    files = os.listdir(os.getcwd())
    for my_file in files:
        if my_file.endswith("pdf"):
            os.rename(my_file, f"docs/out/{my_file}")


def main() -> None:
    """
    Find LaTeX files to compile, compile then, remove auxiliary files and
    move PDFs to output folder.
    """
    files = get_files_with_extension(dir_path="docs/tex/", ext="tex")
    compile_tex_list(files)
    clean_aux_files(aux_exts=("aux", "log", "out"))
    move_pdfs(source_path=os.getcwd(), target_path="docs/out/")


if __name__ == "__main__":
    main()
