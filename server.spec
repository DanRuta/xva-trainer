# -*- mode: python ; coding: utf-8 -*-

block_cipher = None
excluded_modules = ['torch.distributions']

a = Analysis(['server.py'],
             pathex=['envXVATrainerP310\\lib\\site-packages', 'F:\\Speech\\xVA-Trainer'],
             binaries=[],
             datas=[('envXVATrainerP310\\lib\\site-packages\\torchaudio','envXVATrainerP310\\lib\\site-packages\\torchaudio')],
             hiddenimports=['numpy','librosa', 'sklearn', 'numpy.core', 'scipy', 'scipy.linalg.blas',  'numpy.core.multiarray', 'numpy.random.common', 'numpy.random', 'numpy.random.bounded_integers', 'numpy.core.dtype_ctypes', 'pkg_resources.py2_warn' 'numpy.random.entropy', 'tqdm', "sklearn.utils._cython_blas", "sklearn.neighbors.typedefs", "sklearn.neighbors.quad_tree", "sklearn.tree._utils"],
             hookspath=[],
             runtime_hooks=[],
             excludes=excluded_modules,
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='server',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='server')