#!/bin/bash

# List of repositories to clone
repositories=(
    "git@github.com:giellalt/corpus-fit.git"
    "git@github.com:giellalt/corpus-fin.git"
    # This repository requires acess from GiellaTekno's Github
    #"git@github.com:giellalt/corpus-fit-x-closed.git"
)

mkdir meankieli/
mkdir finnish/
# Clone repositories
for repo in "${repositories[@]}"	   
do
    echo "Moving files into respective language folder"
    git clone $repo
    if [[ $repo == *"corpus-fit"* ]]; then
	repo_name=$(basename $repo | sed 's/\.git$//')
	mv $repo_name meankieli/
    fi
    if [[ $repo == *"corpus-fin"* ]]; then
	repo_name=$(basename $repo | sed 's/\.git$//')
	mv $repo_name finnish/
    fi
done


