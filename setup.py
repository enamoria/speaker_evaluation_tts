# -*- coding: utf-8 -*-

""" Created on 2:21 PM, 4/12/19
    @author: ngunhuconchocon
    @brief: Пролетарии всех стран, соединяйтесь!
"""

from setuptools import setup

setup(
    name="pyEval",
    version="0.0.1",
    author="ngunhuconchocon",
    description="Speaker evaluation for TTS corpus",
    license="BSD",
    url="https://github.com/enamoria/speaker_evaluation_tts",
    install_requires=[
          'tqdm', 'scipy', 'numpy', 'pyworld', 'librosa', 'matplotlib', 'parmap'
      ],
    packages=['pyEval'],
    include_package_data=True
)
