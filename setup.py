from distutils.core import setup
setup(
  name = 'skeletonizer',
  packages = ['skeletonizer'], 
  install_requires=[
            'numpy','opencv-python'
      ],
  scripts = [],
  version = '0.1',
  description = 'Skeletonizer test',
  author = 'Ville Rantanen',
  author_email = 'ville.rantanen@reaktor.com',
  url = 'https://github.com/villerantanen/skeletonizer/',
  download_url = 'https://github.com/villerantanen/skeletonizer/archive/master.zip', 
  keywords = ['opencv'], 
  classifiers = [],
  license = 'GPL',
)
