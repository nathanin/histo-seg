#!/bin/bash

# Do a setting to override the base git message if wanted

set -e

git commit -a -m 'saving work'
git push origin master

exit 0
