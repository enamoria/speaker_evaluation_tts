# -*- coding: utf-8 -*-

""" Created on 2:21 PM, 4/12/19
    @author: ngunhuconchocon
    @brief: Пролетарии всех стран, соединяйтесь!
"""

import os
from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="pyEval",
    version="0.0.1",
    author="ngunhuconchocon",
    description="Speaker evaluation for TTS corpus",
    license="BSD",
    url="https://github.com/enamoria/speaker_evaluation_tts",
    install_requires=[
          'tqdm', 'scipy', 'numpy', 'pyworld'
      ],
    packages=['pyEval'],
    include_package_data=True
)
