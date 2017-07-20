To generate epydoc documentation:

epydoc -v General.py --config=epydoc.config
epydoc -v NLP.py --config=epydoc.config
epydoc -v Poetry.py --config=epydoc.config

To check python code:

pylint General.py --max-line-length=200
pylint NLP.py --max-line-length=200 --disable=all --enable=E

To run the ipython notebook:

ipython notebook

To not check outputs in the notebooks under git control:
http://stackoverflow.com/questions/18734739/using-ipython-notebooks-under-version-control
