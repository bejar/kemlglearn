from distutils.core import setup

setup(
    name='kemlglearn',
    version='0.3.13',
    packages=['kemlglearn',
              'kemlglearn.cluster',
              'kemlglearn.cluster.consensus',
              'kemlglearn.cluster.border',
              'kemlglearn.metrics',
              'kemlglearn.datasets',
              'kemlglearn.feature_selection',
              'kemlglearn.feature_selection.unsupervised',
              'kemlglearn.preprocessing',
              'kemlglearn.time_series',
              'kemlglearn.time_series.discretization'],
    url='http://www.cs.upc.edu/~bejar',
    license='GPL 2.0',
    author='Javier Bejar',
    author_email='bejar@lsi.upc.edu',
    description='Machine Learning algorithms/functions '
)
