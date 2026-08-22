def append_to_file(filename, *args):
    """
    Appends the given variable values to the specified text file.

    Parameters:
    - filename: str - name of the file to append to
    - *args: any number of values to write, one per line
    """
    with open(filename, 'a') as file:
        for value in args:
            file.write(str(value) + '\n')
