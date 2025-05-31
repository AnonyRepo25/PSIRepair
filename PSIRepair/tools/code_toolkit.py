import os
import jpype
import jpype.imports
from jpype.types import *


def _init_code_toolkit_backend():
    # Launch the JVM
    ## Set JAVA_HOME
    _old_java_home__ = os.getenv("JAVA_HOME")
    os.environ["JAVA_HOME"] = os.environ["JAVA17_HOME"]
    ## Launch
    backend_src_root_path = os.path.abspath(
        f"{os.path.dirname(__file__)}/CodeToolkitBackend"
    )
    if not jpype.isJVMStarted():
        jpype.startJVM(
            "-enableassertions",
            jvmpath=jpype.getDefaultJVMPath(),
            classpath=[
                f"{backend_src_root_path}/bin",
                f"{backend_src_root_path}/lib/json-20250107.jar",
                f"{backend_src_root_path}/lib/spoon-core-11.2.1-beta-5-jar-with-dependencies.jar",
            ],
        )
    ## Reset JAVA_HOME
    if _old_java_home__ is not None:
        os.environ["JAVA_HOME"] = _old_java_home__
    else:
        del os.environ["JAVA_HOME"]
    del _old_java_home__

    ## Import & Return backend class
    from org.threerepair import CodeToolkitBackend

    return CodeToolkitBackend


CodeToolkitBackend = _init_code_toolkit_backend()
