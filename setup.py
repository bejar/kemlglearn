from distutils.core import setup

setup(
    name='kemlglearn',
    version='0.1',
    packages=['kemlglearn', 'kemlglearn.cluster', 'kemlglearn.metrics', 'kemlglearn.datasets',
              'kemlglearn.feature_selection', 'kemlglearn.feature_selection.unsupervised'],
    url='http://www.cs.upc.edu/~bejar',
    license='GPL 2.0',
    author='Javier Bejar',
    author_email='bejar@lsi.upc.edu',
    description='Machine Learning algorithms/functions '
)
