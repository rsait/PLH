#!/bin/bash

PLH_HOME=$HOME/Github/PLH
TOK_HOME=$HOME/Github/ixa-pipe-tok
POS_HOME=$HOME/Github/ixa-pipe-pos

filename="$1"
while read -r line
do
	echo $line | java -jar $TOK_HOME/target/ixa-pipe-tok-1.8.4.jar tok -l eu | java -jar $POS_HOME/target/ixa-pipe-pos-1.5.1.jar tag -m $POS_HOME/target/classes/ud-morph-models-1.5.0/eu/eu-pos-perceptron-ud.bin -lm $POS_HOME/target/classes/ud-morph-models-1.5.0/eu/eu-lemma-perceptron-ud.bin > _aux.xml
	python $PLH_HOME/Code/ixa_pipe_eu.py _aux.xml
	echo ""
done < "$filename"


