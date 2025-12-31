import os
import subprocess
import shutil
import zipfile

def code_dumper(args):
    """
    Dumps all the git tracked code in the current directory into a folder called code_dump, and zip it.

    Args:
        args: Command line arguments, expected to have a 'work_dir' attribute.

    Returns:
        None
    """
    save_dir = args.work_dir
    dump_dir = os.path.join(save_dir, 'code_dump')

    # Create the dump directory if it doesn't exist
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    # Get the list of git tracked files
    result = subprocess.run(['git', 'ls-files'], stdout=subprocess.PIPE)
    files = result.stdout.decode('utf-8').split('\n')

    # Copy the git tracked files to the dump directory
    for file in files:
        if file:  # Ignore empty strings
            # Create the directory structure in the dump directory
            os.makedirs(os.path.join(dump_dir, os.path.dirname(file)), exist_ok=True)
            # Copy the file to the dump directory, preserving the directory structure
            shutil.copy(file, os.path.join(dump_dir, file))

    # Zip the dump directory
    with zipfile.ZipFile(os.path.join(save_dir, 'code_dump.zip'), 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(dump_dir):
            for file in files:
                # Use the relative path to preserve the directory structure in the zip file
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), dump_dir))

    # Remove the dump directory
    shutil.rmtree(dump_dir)
