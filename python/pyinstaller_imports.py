# This is a junk file which imports a bunch of packages, to make it much more easy to deal with pyinstaller
# to package up all the correct dependencies. There are more proper ways to do this, but this is the easiest

import numpy
import traceback
import numpy.core.multiarray
import numpy.core
import numpy.core.overrides

try:
    from pyannote.audio.features import Pretrained as _Pretrained
except:
    pass
try:
    import scipy
    import scipy.linalg
    import scipy.linalg.blas
    import scipy.special
    import scipy.optimize
    # import scipy.optimize.line_search
    import sklearn.utils._cython_blas
    try:
        import sklearn.neighbors.typedefs
        import sklearn.neighbors.quad_tree
    except:
        import sklearn.neighbors
    import sklearn.tree
    import sklearn.tree._utils
    import sklearn.utils._cython_blas
    try:
        import sklearn.neighbors.typedefs
        import sklearn.neighbors.quad_tree
    except:
        import sklearn.neighbors
    import sklearn.tree._utils
except:
    print("==== scipy")
    print(traceback.format_exc())

from websockets import legacy
from websockets.legacy import auth
from websockets.legacy import client
from websockets import client
from websockets.legacy import framing
from websockets.legacy import handshake
from websockets.legacy import http
from websockets.legacy import protocol
from websockets.legacy import server
from websockets import server

try:
    import pyannote.core
    import pyannote.audio
    from pyannote.audio import features
    from pyannote.audio.features.pretrained import Pretrained
    from pyannote.audio import embedding
    from pyannote.audio.embedding import approaches
except:
    print("==== pyannote")

try:
    import librosa
except:
    print("==== librosa")
    print(traceback.format_exc())
