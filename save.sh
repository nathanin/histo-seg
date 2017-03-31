#!/bin/bash

# Parse arguments

set -e

doPush=false

while [[ $# -gt 1 ]]
do
    key="$1"

    case $key in
        -p|--push)  # Push or not. Default not push
        doPush=true
        echo "Setting push true"
        shift
        ;; # Done with one case
    *)
        # Unknown option
        ;;
esac
shift # past argument
done

git commit -a

if [ "$doPush"=true ]; then
    git push origin master
fi
