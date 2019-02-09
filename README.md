# Setup

This is the setup for install on RHEL 7.

First get the torch wheel compiled for our setup from mseritan@linkedin.com

Create a venv environment and activate it

/export/apps/python/3.7/bin/python3 -m venv .
. ./bin/activate
pip install --upgrade pip

Install the dependencies

pip install -r requirements.txt
