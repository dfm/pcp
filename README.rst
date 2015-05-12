Principal Component Pursuit in Python
=====================================

This is a Python implementation of the `Principal Component
Pursuit <http://arxiv.org/abs/0912.3599>`_ algorithm for robust PCA.

This implementation uses the `fbpca <http://fbpca.readthedocs.org/>`_
implementation of approximate partial SVD for speed so you'll need to install
that first.

Usage
-----

TODO


Demo
----

Applied to `the 'Escalator' dataset
<http://perception.i2r.a-star.edu.sg/bk_model/bk_index.html>`_ (using the code
in the ``demo.py`` script, this algorithm produces a video with frames that
look like:

.. image:: https://raw.githubusercontent.com/dfm/pcp/master/demo.png


Author & License
----------------

Copyright 2015 Daniel Foreman-Mackey

This is open source software written by Dan Foreman-Mackey and released under
the terms of the MIT license (see LICENSE).
