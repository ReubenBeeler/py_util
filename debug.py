from typing import Iterable, Iterator
import pdb

def _get_terminal_path() -> str:
    import subprocess
    proc = subprocess.run("/usr/bin/tty", stdout=subprocess.PIPE)
    return proc.stdout[:-1].decode("ascii")

# print to terminal (regardless of whether stdout is redirected)
def tprint(*args, **kwargs):
    assert "file" not in kwargs
    try:
        with open(_get_terminal_path(), "w") as file:
            print(*args, **kwargs, file=file)
    except:
        from sys import stderr
        print(*args, **kwargs, file=stderr)

def set_trace(auto_start:bool=False):
    with open(_get_terminal_path(), "w") as terminal:
        p = CustomPdb(stdout=terminal)
        commands = ["continue"] if auto_start else []
        # p.onecmd("continue")
        p.set_trace(commands=commands)

class CustomPdb(pdb.Pdb):
    # for calling set_trace with initial commands
    def set_trace(self, frame=None, commands:Iterable[str]|Iterator[str]=[]):
        # Initialize the debugger and set the frame
        if frame is None:
            frame = pdb.sys._getframe().f_back

        # Reset the debugger state
        self.reset()
        
        # Set up the frame and local variables
        self.curframe = frame
        self.curframe_locals = frame.f_locals

        for cmd in commands:
            self.onecmd(cmd)

        # Now start the actual debugger session
        super().set_trace(frame)