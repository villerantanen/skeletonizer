from distutils.core import setup
import time
setup(
  name = 'skeletonizer',
  packages = ['skeletonizer'], 
  install_requires=[
            'numpy','opencv-python'
      ],
  scripts = [],
  version = str(int(time.mktime(time.localtime()))),
  description = 'Skeletonizer test',
  author = 'Ville Rantanen',
  author_email = 'ville.rantanen@reaktor.com',
  url = 'https://github.com/villerantanen/skeletonizer/',
  download_url = 'https://github.com/villerantanen/skeletonizer/archive/master.zip', 
  keywords = ['opencv'], 
  classifiers = [],
  license = 'GPL',
)
