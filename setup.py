from setuptools import setup

setup(name='EL',
      version='0.1',
      description='Emergent languages implementation',
      url='https://github.build.ge.com/el/EL.git',
      author='Aritra Chowdhury',
      author_email='aritra.chowdhury@ge.com',
      license='MIT',
      packages=['EL'],
      install_requires=['torch',
                        'torchvision',
                        'sacred'
                        ],
      zip_safe=False)
