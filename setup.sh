# Get pip
wget https://bootstrap.pypa.io/pip/2.7/get-pip.py
python2 get-pip.py
rm get-pip.py

# Setup virtual environment
python2 -m pip install virtualenv
python2 -m virtualenv .venv
. .venv/bin/activate
pip install -r requirements.txt

# nltk (for text mutations) needs these
sudo apt install build-essential python-dev
