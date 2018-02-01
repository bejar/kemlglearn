from distutils.core import setup

setup(
    name='kemlglearn',
    version='0.3.15',
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
              'kemlglearn.time_series.discretization',
              'kemlglearn.time_series.decomposition',
              'kemlglearn.time_series.smoothing'],
    url='http://www.cs.upc.edu/~bejar',
    license='GPL 2.0',
    author='Javier Bejar',
    author_email='bejar@cs.upc.edu',
    description='Machine Learning algorithms/functions '
)
